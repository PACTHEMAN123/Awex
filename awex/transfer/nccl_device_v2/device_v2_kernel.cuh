// Licensed to the Awex developers under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include "device_v2_primitives.cuh"

namespace awex {
namespace nccl_device_v2 {

__device__ __forceinline__ bool v2SideUsesChannel(const V2WorkSide& side, std::uint32_t channel, std::uint32_t* part) {
  if (!side.enabled || channel < side.channel_base) {
    return false;
  }
  const std::uint32_t relative = channel - side.channel_base;
  if (relative >= side.channel_count) {
    return false;
  }
  *part = relative;
  return true;
}

__device__ __forceinline__ void v2PartBounds(std::uint32_t parts, std::uint32_t part, std::uint64_t bytes,
                                             std::uint64_t* begin, std::uint64_t* end) {
  *begin = (bytes * part) / parts;
  *end = (bytes * (part + 1)) / parts;
}

__device__ __forceinline__ bool v2WaitFreeWarp(volatile V2FifoSlot* slot, unsigned long long step,
                                               const V2KernelArgs& args) {
  bool ready = true;
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    ready = v2WaitFree(slot, step, args.layout.fifo_depth, &reinterpret_cast<V2WindowHeader*>(args.local_window)->error,
                       args.timeout_cycles);
  }
  return __shfl_sync(0xffffffffU, ready, 0);
}

__device__ __forceinline__ bool v2WaitReadyWarp(volatile V2FifoSlot* slot, unsigned long long step,
                                                const V2KernelArgs& args) {
  bool ready = true;
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    ready = v2WaitReady(&slot->ready_step, step, &reinterpret_cast<V2WindowHeader*>(args.local_window)->error,
                        args.timeout_cycles);
  }
  return __shfl_sync(0xffffffffU, ready, 0);
}

__device__ __forceinline__ void v2RunSend(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                          std::uint32_t part, int tid, int nthreads) {
  const V2WorkSide& side = work.send;
  std::uint64_t part_begin = 0;
  std::uint64_t part_end = 0;
  v2PartBounds(side.channel_count, part, side.nbytes, &part_begin, &part_end);
  const std::uint64_t part_bytes = part_end - part_begin;
  std::uint64_t cursor = 0;
  std::uint64_t step = side.step_begin[part];
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;

  while (cursor < part_bytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < part_bytes - cursor ? args.layout.slot_bytes : part_bytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, work.peer, args.local_rank, channel, step, false);
    if (!v2WaitFreeWarp(slot, step, args)) {
      return;
    }
    std::uint8_t* payload = v2FifoPayload(args, work.peer, args.local_rank, channel, step, false);
    const auto* tensor = reinterpret_cast<const std::uint8_t*>(side.tensor_ptr);
    v2CopyTensorToContiguous(payload, tensor, side.tensor_offset + part_begin + cursor, slice_bytes,
                             side.tensor_row_bytes, side.tensor_row_stride, tid, nthreads);
    __syncwarp();
    if (tid == 0) {
      slot->bytes = static_cast<std::uint32_t>(slice_bytes);
      v2Publish(&slot->ready_step, step);
    }
    cursor += slice_bytes;
    ++step;
  }

  // The final consume acknowledgement protects the source tensor lifetime
  // when the caller reuses it as soon as the kernel returns.
  if (part_bytes != 0 && (side.final_parts & (1U << part)) != 0) {
    V2FifoSlot* final_slot = v2FifoSlot(args, work.peer, args.local_rank, channel, step - 1, false);
    bool consumed = true;
    if (tid == 0) {
      consumed = v2WaitConsumed(final_slot, step - 1, error, args.timeout_cycles);
    }
    consumed = __shfl_sync(0xffffffffU, consumed, 0);
    if (!consumed) {
      return;
    }
  }
}

__device__ __forceinline__ void v2RunRecv(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                          std::uint32_t part, int tid, int nthreads) {
  const V2WorkSide& side = work.recv;
  std::uint64_t part_begin = 0;
  std::uint64_t part_end = 0;
  v2PartBounds(side.channel_count, part, side.nbytes, &part_begin, &part_end);
  const std::uint64_t part_bytes = part_end - part_begin;
  std::uint64_t cursor = 0;
  std::uint64_t step = side.step_begin[part];

  while (cursor < part_bytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < part_bytes - cursor ? args.layout.slot_bytes : part_bytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, work.peer, work.peer, channel, step, true);
    if (!v2WaitReadyWarp(slot, step, args)) {
      return;
    }
    const std::uint8_t* payload = v2FifoPayload(args, work.peer, work.peer, channel, step, true);
    auto* tensor = reinterpret_cast<std::uint8_t*>(side.tensor_ptr);
    v2CopyContiguousToTensor(tensor, payload, side.tensor_offset + part_begin + cursor, slice_bytes,
                             side.tensor_row_bytes, side.tensor_row_stride, tid, nthreads);
    __syncwarp();
    if (tid == 0) {
      v2Publish(&slot->consumed_step, step);
    }
    cursor += slice_bytes;
    ++step;
  }
}

struct V2BatchShared {
  std::uint32_t send_mask;
  std::uint32_t recv_mask;
  std::int8_t send_warp[kMaxWorksPerBatch];
  std::int8_t recv_warp[kMaxWorksPerBatch];
};

__device__ __forceinline__ void v2RunBatch(const V2KernelArgs& args, const V2WorkBatch& batch, std::uint32_t channel) {
  __shared__ V2BatchShared shared;
  const int tid = threadIdx.x;
  const int wid = tid / kWarpSize;
  const int lane = tid % kWarpSize;

  if (wid == 0 && lane == 0) {
    shared.send_mask = 0;
    shared.recv_mask = 0;
    for (std::uint32_t index = 0; index < kMaxWorksPerBatch; ++index) {
      shared.send_warp[index] = -1;
      shared.recv_warp[index] = -1;
    }
    int next_warp = 0;
    for (std::uint32_t index = 0; index < batch.work_count; ++index) {
      const V2Work& work = args.works[batch.work_begin + index];
      std::uint32_t part = 0;
      if (v2SideUsesChannel(work.send, channel, &part)) {
        shared.send_mask |= 1U << index;
        shared.send_warp[index] = static_cast<std::int8_t>(next_warp++);
      }
      if (v2SideUsesChannel(work.recv, channel, &part)) {
        shared.recv_mask |= 1U << index;
        shared.recv_warp[index] = static_cast<std::int8_t>(next_warp++);
      }
    }
    if (next_warp > blockDim.x / kWarpSize) {
      atomicExch_system(&reinterpret_cast<V2WindowHeader*>(args.local_window)->error, 1U);
    }
  }
  __syncthreads();

  for (std::uint32_t index = 0; index < batch.work_count; ++index) {
    const V2Work& work = args.works[batch.work_begin + index];
    std::uint32_t part = 0;
    if (shared.send_warp[index] == wid && v2SideUsesChannel(work.send, channel, &part)) {
      v2RunSend(args, work, channel, part, lane, kWarpSize);
    }
    if (shared.recv_warp[index] == wid && v2SideUsesChannel(work.recv, channel, &part)) {
      v2RunRecv(args, work, channel, part, lane, kWarpSize);
    }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreadsPerBlock, 1) device_v2_kernel(V2KernelArgs args) {
  auto* local_header = reinterpret_cast<V2WindowHeader*>(args.local_window);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    atomicExch_system(&local_header->epoch, args.epoch);
  }

  if (blockIdx.x >= args.channel_count) {
    return;
  }
  const V2ChannelQueue queue = args.channels[blockIdx.x];
  for (std::uint32_t index = 0; index < queue.batch_count; ++index) {
    v2RunBatch(args, args.batches[queue.first_batch + index], blockIdx.x);
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

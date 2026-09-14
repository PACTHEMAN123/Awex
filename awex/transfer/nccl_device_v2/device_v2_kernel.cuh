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

__device__ __forceinline__ void v2RunSend(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                          std::uint32_t part, int group, int tid, int nthreads, int* group_ready) {
  const V2WorkSide& side = work.send;
  std::uint64_t part_begin = 0;
  std::uint64_t part_end = 0;
  v2PartBounds(side.channel_count, part, side.nbytes, &part_begin, &part_end);
  const std::uint64_t part_bytes = part_end - part_begin;
  std::uint64_t cursor = 0;
  std::uint64_t step = side.step_begin[part];
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  if (tid == 0) {
    auto* remote_header = reinterpret_cast<V2WindowHeader*>(args.peer_windows[work.peer]);
    *group_ready = v2WaitReady(&remote_header->epoch, args.epoch, error, args.timeout_cycles);
    if (!*group_ready) atomicExch_system(error, 4U);
  }
  v2GroupBarrier(group, nthreads);
  if (!*group_ready) return;

  while (cursor < part_bytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < part_bytes - cursor ? args.layout.slot_bytes : part_bytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, work.peer, args.local_rank, channel, step, false);
    if (tid == 0)
      *group_ready = v2WaitFree(slot, step, args.layout.fifo_depth, error, args.timeout_cycles);
    v2GroupBarrier(group, nthreads);
    if (!*group_ready) return;

    std::uint8_t* payload = v2FifoPayload(args, work.peer, args.local_rank, channel, step, false);
    const auto* tensor = reinterpret_cast<const std::uint8_t*>(side.tensor_ptr);
    v2CopyTensorToContiguous(payload, tensor, side.tensor_offset + part_begin + cursor, slice_bytes,
                             side.tensor_row_bytes, side.tensor_row_stride, tid, nthreads);
    v2GroupBarrier(group, nthreads);
    if (tid == 0) {
      slot->bytes = static_cast<std::uint32_t>(slice_bytes);
      v2Publish(&slot->ready_step, step);
    }
    v2GroupBarrier(group, nthreads);
    cursor += slice_bytes;
    ++step;
  }

  // The final consume acknowledgement protects the source tensor lifetime
  // when the caller reuses it as soon as the kernel returns.
  if (part_bytes != 0 && (side.final_parts & (1U << part)) != 0) {
    V2FifoSlot* final_slot = v2FifoSlot(args, work.peer, args.local_rank, channel, step - 1, false);
    if (tid == 0) *group_ready = v2WaitConsumed(final_slot, step - 1, error, args.timeout_cycles);
    v2GroupBarrier(group, nthreads);
  }
}

__device__ __forceinline__ void v2RunRecv(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                          std::uint32_t part, int group, int tid, int nthreads, int* group_ready) {
  const V2WorkSide& side = work.recv;
  std::uint64_t part_begin = 0;
  std::uint64_t part_end = 0;
  v2PartBounds(side.channel_count, part, side.nbytes, &part_begin, &part_end);
  const std::uint64_t part_bytes = part_end - part_begin;
  std::uint64_t cursor = 0;
  std::uint64_t step = side.step_begin[part];
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  if (tid == 0) {
    auto* remote_header = reinterpret_cast<V2WindowHeader*>(args.peer_windows[work.peer]);
    *group_ready = v2WaitReady(&remote_header->epoch, args.epoch, error, args.timeout_cycles);
    if (!*group_ready) atomicExch_system(error, 4U);
  }
  v2GroupBarrier(group, nthreads);
  if (!*group_ready) return;

  while (cursor < part_bytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < part_bytes - cursor ? args.layout.slot_bytes : part_bytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, work.peer, work.peer, channel, step, true);
    if (tid == 0) *group_ready = v2WaitReady(&slot->ready_step, step, error, args.timeout_cycles);
    v2GroupBarrier(group, nthreads);
    if (!*group_ready) return;

    const std::uint8_t* payload = v2FifoPayload(args, work.peer, work.peer, channel, step, true);
    auto* tensor = reinterpret_cast<std::uint8_t*>(side.tensor_ptr);
    v2CopyContiguousToTensor(tensor, payload, side.tensor_offset + part_begin + cursor, slice_bytes,
                             side.tensor_row_bytes, side.tensor_row_stride, tid, nthreads);
    v2GroupBarrier(group, nthreads);
    if (tid == 0) v2Publish(&slot->consumed_step, step);
    v2GroupBarrier(group, nthreads);
    cursor += slice_bytes;
    ++step;
  }
}

struct V2BatchShared {
  std::uint32_t active_count;
  std::uint32_t work_index[kMaxWorksPerBatch];
  int group_ready[kMaxWorksPerBatch];
};

__device__ __forceinline__ void v2RunBatch(const V2KernelArgs& args, const V2WorkBatch& batch, std::uint32_t channel) {
  __shared__ V2BatchShared shared;
  const int tid = threadIdx.x;
  const int wid = tid / kWarpSize;
  const int lane = tid % kWarpSize;

  if (wid == 0 && lane == 0) {
    shared.active_count = 0;
    for (std::uint32_t index = 0; index < batch.work_count; ++index) {
      const V2Work& work = args.works[batch.work_begin + index];
      std::uint32_t part = 0;
      if (v2SideUsesChannel(work.send, channel, &part) || v2SideUsesChannel(work.recv, channel, &part)) {
        shared.work_index[shared.active_count++] = index;
      }
    }
  }
  __syncthreads();

  if (shared.active_count != 0) {
    const int warps_per_work = kWarpsPerBlock / shared.active_count;
    const int group = wid / warps_per_work;
    if (group < shared.active_count) {
      const std::uint32_t work_index = shared.work_index[group];
      const V2Work& work = args.works[batch.work_begin + work_index];
      const int subtid = (wid - group * warps_per_work) * kWarpSize + lane;
      const int subthreads = warps_per_work * kWarpSize;
      std::uint32_t part = 0;
      if (v2SideUsesChannel(work.send, channel, &part)) {
        v2RunSend(args, work, channel, part, group, subtid, subthreads, &shared.group_ready[group]);
      } else if (v2SideUsesChannel(work.recv, channel, &part)) {
        v2RunRecv(args, work, channel, part, group, subtid, subthreads, &shared.group_ready[group]);
      }
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

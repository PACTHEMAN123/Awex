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

struct V2BatchShared {
  unsigned long long ready_steps[kMaxWorksPerBatch];
  unsigned int completed_worker_warps[kMaxWorksPerBatch];
};

__device__ __forceinline__ int v2WorkerWarps(std::uint32_t group, std::uint32_t work_count) {
  return kWorkerWarpsPerBlock / work_count + (group < kWorkerWarpsPerBlock % work_count ? 1 : 0);
}

__device__ __forceinline__ int v2WorkerWarpBegin(std::uint32_t group, std::uint32_t work_count) {
  const int base = kWorkerWarpsPerBlock / work_count;
  const int remainder = kWorkerWarpsPerBlock % work_count;
  return 1 + static_cast<int>(group) * base +
         (static_cast<int>(group) < remainder ? static_cast<int>(group) : remainder);
}

__device__ __forceinline__ void v2WaitWork(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                           std::uint32_t group, V2BatchShared* shared) {
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  unsigned long long step_cache = 0;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    V2FifoSlot* slot = args.direction == V2Direction::kSend
                         ? v2FifoSlot(args, args.local_rank, work.peer, channel, step, true)
                         : v2FifoSlot(args, work.peer, args.local_rank, channel, step, false);
    const bool ready = args.direction == V2Direction::kSend
                         ? v2WaitFree(slot, step, args.layout.fifo_depth, &step_cache, error, args.timeout_cycles)
                         : v2WaitReady(&slot->ready_step, step, &step_cache, error, args.timeout_cycles);
    if (!ready) break;
    v2FenceSystem();
    atomicExch_block(&shared->ready_steps[group], step);
    cursor += args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    ++step;
  }

  if (args.direction == V2Direction::kSend && work.final && work.nbytes != 0 && v2LoadError(error) == 0) {
    V2FifoSlot* slot = v2FifoSlot(args, args.local_rank, work.peer, channel, step - 1, true);
    v2WaitConsumed(slot, step - 1, &step_cache, error, args.timeout_cycles);
  }
}

__device__ __forceinline__ void v2CopyWork(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                           std::uint32_t group, int subtid, int nworkers, V2BatchShared* shared) {
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    while (atomicAdd_block(&shared->ready_steps[group], 0ULL) < step && v2LoadError(error) == 0) {
    }
    if (v2LoadError(error) != 0) break;
    v2FenceSystem();

    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    if (args.direction == V2Direction::kSend) {
      std::uint8_t* payload = v2FifoPayload(args, args.local_rank, work.peer, channel, step, true);
      v2CopyFragmentsToContiguous(args, work, payload, cursor, slice_bytes, subtid, nworkers);
    } else {
      const std::uint8_t* payload = v2FifoPayload(args, work.peer, args.local_rank, channel, step, false);
      v2CopyContiguousToFragments(args, work, payload, cursor, slice_bytes, subtid, nworkers);
    }
    if (args.direction == V2Direction::kSend) v2FenceSystem();
    __syncwarp();
    if (subtid % kWarpSize == 0) atomicAdd_block(&shared->completed_worker_warps[group], 1U);
    cursor += slice_bytes;
    ++step;
  }
}

__device__ __forceinline__ void v2PostWork(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                           std::uint32_t group, std::uint32_t worker_warps, V2BatchShared* shared) {
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  unsigned int completed_target = worker_warps;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    while (atomicAdd_block(&shared->completed_worker_warps[group], 0U) < completed_target &&
           v2LoadError(error) == 0) {
    }
    if (v2LoadError(error) != 0) break;

    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    V2FifoSlot* slot = args.direction == V2Direction::kSend
                         ? v2FifoSlot(args, args.local_rank, work.peer, channel, step, true)
                         : v2FifoSlot(args, work.peer, args.local_rank, channel, step, false);
    if (args.direction == V2Direction::kSend) slot->bytes = static_cast<std::uint32_t>(slice_bytes);
    v2Publish(args.direction == V2Direction::kSend ? &slot->ready_step : &slot->consumed_step, step);
    cursor += slice_bytes;
    ++step;
    completed_target += worker_warps;
  }
}

__device__ __forceinline__ void v2RunBatch(const V2KernelArgs& args, const V2WorkBatch& batch, std::uint32_t channel) {
  __shared__ V2BatchShared shared;
  const int tid = threadIdx.x;
  const int wid = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  if (tid < kMaxWorksPerBatch) {
    shared.ready_steps[tid] = 0;
    shared.completed_worker_warps[tid] = 0;
  }
  __syncthreads();

  if (wid == 0 && lane < batch.work_count) {
    const V2Work& work = args.works[batch.work_begin + lane];
    v2WaitWork(args, work, channel, lane, &shared);
  } else if (wid == kWarpsPerBlock - 1 && lane < batch.work_count) {
    const V2Work& work = args.works[batch.work_begin + lane];
    v2PostWork(args, work, channel, lane, v2WorkerWarps(lane, batch.work_count), &shared);
  } else if (wid > 0 && wid < kWarpsPerBlock - 1) {
    for (std::uint32_t group = 0; group < batch.work_count; ++group) {
      const int worker_begin = v2WorkerWarpBegin(group, batch.work_count);
      const int worker_warps = v2WorkerWarps(group, batch.work_count);
      if (wid >= worker_begin && wid < worker_begin + worker_warps) {
        const int subtid = (wid - worker_begin) * kWarpSize + lane;
        const V2Work& work = args.works[batch.work_begin + group];
        v2CopyWork(args, work, channel, group, subtid, worker_warps * kWarpSize, &shared);
        break;
      }
    }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreadsPerBlock, 1) device_v2_kernel(V2KernelArgs args) {
  auto* local_header = reinterpret_cast<V2WindowHeader*>(args.local_window);
  if (blockIdx.x == 0 && threadIdx.x == 0) v2Publish(&local_header->epoch, args.epoch);

  __shared__ int peers_ready;
  if (threadIdx.x == 0) {
    peers_ready = 1;
    for (std::uint32_t index = 0; index < args.active_peer_count && peers_ready; ++index) {
      const std::uint32_t peer = args.active_peers[index];
      auto* remote_header = reinterpret_cast<V2WindowHeader*>(args.peer_windows[peer]);
      unsigned long long cache = 0;
      peers_ready = v2WaitReady(&remote_header->epoch, args.epoch, &cache, &local_header->error, args.timeout_cycles);
    }
    if (!peers_ready) atomicExch_system(&local_header->error, 4U);
  }
  __syncthreads();
  if (!peers_ready) return;

  const std::uint32_t channel = args.channel_ids[blockIdx.x];
  const V2ChannelQueue queue = args.channels[blockIdx.x];
  for (std::uint32_t index = 0; index < queue.batch_count; ++index) {
    v2RunBatch(args, args.batches[queue.first_batch + index], channel);
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

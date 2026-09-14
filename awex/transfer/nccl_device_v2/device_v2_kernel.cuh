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
  int ready[kMaxWorksPerBatch];
  unsigned long long step_cache[kMaxWorksPerBatch];
};

__device__ __forceinline__ std::uint32_t v2Roles(V2Direction direction, int tid, int nthreads, int* nworkers) {
  const bool send = direction == V2Direction::kSend;
  *nworkers = nthreads - (nthreads >= 3 * kWarpSize ? kWarpSize : 0);
  std::uint32_t roles = tid < *nworkers ? kRoleWorker : 0;
  if (tid == 0) roles |= send ? kRoleWaitSend : kRoleWaitRecv;
  if (tid == nthreads - 1) roles |= send ? kRolePostSend : kRolePostRecv;
  return roles;
}

// In read mode the producer owns the FIFO payload. The sender only writes its
// local window; the receiver performs the NVLink read and returns credits by
// writing consumed_step back into the sender's window.
__device__ __forceinline__ void v2RunSend(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel, int tid,
                                          int nthreads, int main_barrier, int wait_barrier, int* ready,
                                          unsigned long long* step_cache) {
  int nworkers = 0;
  const std::uint32_t roles = v2Roles(V2Direction::kSend, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, args.local_rank, work.peer, channel, step, true);
    if (roles & kRoleWaitSend) {
      *ready = v2WaitFree(slot, step, args.layout.fifo_depth, step_cache, error, args.timeout_cycles);
    }
    if (roles & kRoleWorker) {
      v2GroupBarrier(wait_barrier, nworkers);
      if (*ready) {
        std::uint8_t* payload = v2FifoPayload(args, args.local_rank, work.peer, channel, step, true);
        v2CopyFragmentsToContiguous(args, work, payload, cursor, slice_bytes, tid, nworkers);
      }
    }

    // Like NCCL SIMPLE send, a wide group reserves its final warp for Post.
    // Workers can begin the next step while Post fences and publishes this one.
    v2GroupBarrier(main_barrier, nthreads);
    if ((roles & kRolePostSend) && v2LoadError(error) == 0) {
      slot->bytes = static_cast<std::uint32_t>(slice_bytes);
      v2Publish(&slot->ready_step, step);
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }

  if (work.final && work.nbytes != 0) {
    V2FifoSlot* slot = v2FifoSlot(args, args.local_rank, work.peer, channel, step - 1, true);
    if (roles & kRoleWaitSend) {
      *ready = v2WaitConsumed(slot, step - 1, step_cache, error, args.timeout_cycles);
    }
    v2GroupBarrier(main_barrier, nthreads);
  }
}

__device__ __forceinline__ void v2RunRecv(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel, int tid,
                                          int nthreads, int barrier, int* ready,
                                          unsigned long long* step_cache) {
  int nworkers = 0;
  const std::uint32_t roles = v2Roles(V2Direction::kRecv, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, work.peer, args.local_rank, channel, step, false);
    if (roles & kRoleWaitRecv) {
      *ready = v2WaitReady(&slot->ready_step, step, step_cache, error, args.timeout_cycles);
    }
    v2GroupBarrier(barrier, nthreads);
    if (*ready && (roles & kRoleWorker)) {
      const std::uint8_t* payload = v2FifoPayload(args, work.peer, args.local_rank, channel, step, false);
      v2CopyContiguousToFragments(args, work, payload, cursor, slice_bytes, tid, nworkers);
    }

    v2GroupBarrier(barrier, nthreads);
    if ((roles & kRolePostRecv) && v2LoadError(error) == 0) v2Publish(&slot->consumed_step, step);
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }
}

__device__ __forceinline__ void v2RunBatch(const V2KernelArgs& args, const V2WorkBatch& batch,
                                           std::uint32_t channel) {
  __shared__ V2BatchShared shared;
  const int tid = threadIdx.x;
  const int wid = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int warps_per_work = kWarpsPerBlock / batch.work_count;
  const int group = wid / warps_per_work;
  if (group < batch.work_count) {
    const int subtid = (wid - group * warps_per_work) * kWarpSize + lane;
    const int subthreads = warps_per_work * kWarpSize;
    const bool extra_send_barrier = args.direction == V2Direction::kSend && subthreads >= 3 * kWarpSize;
    const int barrier_width = extra_send_barrier ? 2 : 1;
    const int main_barrier = 1 + group * barrier_width;
    const int wait_barrier = extra_send_barrier ? main_barrier + 1 : main_barrier;
    if (subtid == 0) {
      shared.ready[group] = 1;
      shared.step_cache[group] = 0;
    }
    v2GroupBarrier(main_barrier, subthreads);

    const V2Work& work = args.works[batch.work_begin + group];
    if (args.direction == V2Direction::kSend) {
      v2RunSend(args, work, channel, subtid, subthreads, main_barrier, wait_barrier, &shared.ready[group],
                &shared.step_cache[group]);
    } else {
      v2RunRecv(args, work, channel, subtid, subthreads, main_barrier, &shared.ready[group],
                &shared.step_cache[group]);
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
      peers_ready = v2WaitReady(&remote_header->epoch, args.epoch, &cache, &local_header->error,
                                args.timeout_cycles);
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

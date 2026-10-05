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
__device__ __forceinline__ void v2RunSendLsa(const V2KernelArgs& args, const V2Work& work,
                                             std::uint32_t channel, int tid, int nthreads,
                                             int main_barrier, int wait_barrier, int* ready,
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

__device__ __forceinline__ void v2RunRecvLsa(const V2KernelArgs& args, const V2Work& work,
                                             std::uint32_t channel, int tid, int nthreads,
                                             int barrier, int* ready,
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

__device__ __forceinline__ void v2RunSendGin(const V2KernelArgs& args, const V2Work& work,
                                             std::uint32_t channel, const ncclGin& gin,
                                             int tid, int nthreads, int main_barrier,
                                             int wait_barrier, int* ready,
                                             unsigned long long* step_cache) {
  int nworkers = 0;
  const std::uint32_t roles = v2Roles(V2Direction::kSend, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  const ncclTeam world = ncclTeamWorld(args.dev_comm);
  const ncclTeam rail = ncclTeamRail(args.dev_comm);
  const std::uint32_t lsa_size = static_cast<std::uint32_t>(args.dev_comm.lsaSize);
  const std::uint32_t local_lsa_rank = static_cast<std::uint32_t>(args.dev_comm.lsaRank);
  const std::uint32_t destination_node = work.peer / lsa_size;
  const std::uint32_t destination_proxy = destination_node * lsa_size + local_lsa_rank;
  const int gin_peer = ncclTeamRankToTeam(rail, world, static_cast<int>(destination_proxy));
  const std::uint32_t source_node = args.local_rank / lsa_size;
  const std::uint32_t ack_proxy = source_node * lsa_size + work.peer % lsa_size;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    const unsigned long long generation = v2GinGeneration(args, step);
    V2FifoSlot* ack_slot = v2FifoSlot(args, ack_proxy, work.peer, channel, step,
                                      ack_proxy == args.local_rank);
    if (roles & kRoleWaitSend) {
      *step_cache = 0;
      *ready = generation == 1 ||
               v2WaitReady(&ack_slot->consumed_step, generation - 1, step_cache,
                           error, args.timeout_cycles);
    }
    if (roles & kRoleWorker) {
      v2GroupBarrier(wait_barrier, nworkers);
      if (*ready) {
        std::uint8_t* payload = v2GinPayload(args, work.peer, channel, step);
        v2CopyFragmentsToContiguous(args, work, payload, cursor, slice_bytes, tid, nworkers);
      }
    }

    v2GroupBarrier(main_barrier, nthreads);
    if ((roles & kRolePostSend) && v2LoadError(error) == 0) {
      gin.put(rail, gin_peer,
              args.window, v2GinPayloadOffset(args, args.local_rank, channel, step),
              args.window, v2GinPayloadOffset(args, work.peer, channel, step),
              slice_bytes);
      gin.putValue(rail, gin_peer, args.window,
                   v2GinSlotOffset(args, args.local_rank, channel, step) +
                     offsetof(V2FifoSlot, ready_step),
                   generation);
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }

  if (work.final && work.nbytes != 0) {
    const unsigned long long final_step = step - 1;
    const unsigned long long generation = v2GinGeneration(args, final_step);
    V2FifoSlot* ack_slot = v2FifoSlot(args, ack_proxy, work.peer, channel, final_step,
                                      ack_proxy == args.local_rank);
    if (roles & kRoleWaitSend) {
      *step_cache = 0;
      *ready = v2WaitReady(&ack_slot->consumed_step, generation, step_cache,
                           error, args.timeout_cycles);
    }
    v2GroupBarrier(main_barrier, nthreads);
  }
}

__device__ __forceinline__ void v2RunRecvGin(const V2KernelArgs& args, const V2Work& work,
                                             std::uint32_t channel, const ncclGin& gin,
                                             int tid, int nthreads, int barrier, int* ready,
                                             unsigned long long* step_cache) {
  int nworkers = 0;
  const std::uint32_t roles = v2Roles(V2Direction::kRecv, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  const ncclTeam world = ncclTeamWorld(args.dev_comm);
  const ncclTeam rail = ncclTeamRail(args.dev_comm);
  const std::uint32_t lsa_size = static_cast<std::uint32_t>(args.dev_comm.lsaSize);
  const std::uint32_t receiver_node = args.local_rank / lsa_size;
  const std::uint32_t ready_proxy = receiver_node * lsa_size + work.peer % lsa_size;
  const std::uint32_t source_node = work.peer / lsa_size;
  const std::uint32_t ack_proxy = source_node * lsa_size + args.local_rank % lsa_size;
  const int gin_peer = ncclTeamRankToTeam(rail, world, static_cast<int>(ack_proxy));
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    const unsigned long long generation = v2GinGeneration(args, step);
    V2FifoSlot* ready_slot = v2FifoSlot(args, ready_proxy, work.peer, channel, step,
                                        ready_proxy == args.local_rank);
    if (roles & kRoleWaitRecv) {
      *step_cache = 0;
      *ready = v2WaitReady(&ready_slot->ready_step, generation, step_cache,
                           error, args.timeout_cycles);
    }
    v2GroupBarrier(barrier, nthreads);
    if (*ready && (roles & kRoleWorker)) {
      const std::uint8_t* payload =
        v2GinProxyPayload(args, ready_proxy, work.peer, channel, step);
      v2CopyContiguousToFragments(args, work, payload, cursor, slice_bytes, tid, nworkers);
    }

    v2GroupBarrier(barrier, nthreads);
    if ((roles & kRolePostRecv) && v2LoadError(error) == 0) {
      gin.putValue(rail, gin_peer, args.window,
                   v2GinSlotOffset(args, args.local_rank, channel, step) +
                     offsetof(V2FifoSlot, consumed_step),
                   generation);
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }
}

__device__ __forceinline__ void v2RunBatch(const V2KernelArgs& args, const V2WorkBatch& batch,
                                           std::uint32_t channel, const ncclGin* gin) {
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
    const V2Work& work = args.works[batch.work_begin + group];
    if (subtid == 0) {
      shared.ready[group] = 1;
      shared.step_cache[group] = 0;
    }
    v2GroupBarrier(main_barrier, subthreads);

    if (args.use_gin && args.direction == V2Direction::kSend) {
      v2RunSendGin(args, work, channel, *gin, subtid, subthreads,
                   main_barrier, wait_barrier, &shared.ready[group], &shared.step_cache[group]);
    } else if (args.use_gin) {
      v2RunRecvGin(args, work, channel, *gin, subtid, subthreads,
                   main_barrier, &shared.ready[group], &shared.step_cache[group]);
    } else if (args.direction == V2Direction::kSend) {
      v2RunSendLsa(args, work, channel, subtid, subthreads, main_barrier, wait_barrier,
                   &shared.ready[group], &shared.step_cache[group]);
    } else {
      v2RunRecvLsa(args, work, channel, subtid, subthreads, main_barrier,
                   &shared.ready[group], &shared.step_cache[group]);
    }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreadsPerBlock, 1) device_v2_kernel(V2KernelArgs args) {
  auto* local_header = reinterpret_cast<V2WindowHeader*>(args.local_window);
  const std::uint32_t channel = args.channel_ids[blockIdx.x];
  const V2ChannelQueue queue = args.channels[blockIdx.x];

  if (args.use_gin) {
    ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
    ncclBarrierSession<ncclCoopCta> barrier{
      ncclCoopCta(), ncclTeamTagWorld(), gin, channel};
    const ncclResult_t entry = barrier.sync(ncclCoopCta(), cuda::memory_order_acquire,
                                             ncclGinFenceLevel::Relaxed, args.timeout_cycles);
    if (entry != ncclSuccess) {
      if (threadIdx.x == 0) atomicExch_system(&local_header->error, 6U);
      __syncthreads();
      return;
    }

    for (std::uint32_t index = 0; index < queue.batch_count; ++index) {
      v2RunBatch(args, args.batches[queue.first_batch + index], channel, &gin);
    }
    gin.flush(ncclCoopCta());
    const ncclResult_t exit = barrier.sync(ncclCoopCta(), cuda::memory_order_release,
                                            ncclGinFenceLevel::Relaxed, args.timeout_cycles);
    if (exit != ncclSuccess && threadIdx.x == 0) atomicExch_system(&local_header->error, 7U);
    return;
  }

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

  for (std::uint32_t index = 0; index < queue.batch_count; ++index) {
    v2RunBatch(args, args.batches[queue.first_batch + index], channel, nullptr);
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

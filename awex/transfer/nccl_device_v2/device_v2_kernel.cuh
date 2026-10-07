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

#include "device_v2_gin.cuh"

namespace awex {
namespace nccl_device_v2 {

struct V2BatchShared {
  int ready[kMaxWorksPerBatch];
  unsigned long long step_cache[kMaxWorksPerBatch];
  unsigned long long forward_step_cache[kMaxWorksPerBatch];
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
                                          unsigned long long* step_cache, V2KernelProfile* profile) {
  int nworkers = 0;
  const std::uint32_t roles = v2Roles(V2Direction::kSend, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      work.step_bytes < work.nbytes - cursor ? work.step_bytes : work.nbytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, args.local_rank, work.peer, channel, step, work.fifo_depth, true);
    if (roles & kRoleWaitSend) {
      const unsigned long long wait_start = clock64();
      *ready = v2WaitFree(slot, step, work.fifo_depth, step_cache, error, args.timeout_cycles);
      profile->output_wait_cycles += clock64() - wait_start;
    }
    const unsigned long long copy_start = tid == 0 ? clock64() : 0;
    if (roles & kRoleWorker) {
      v2GroupBarrier(wait_barrier, nworkers);
      if (*ready) {
        std::uint8_t* payload =
          v2FifoPayload(args, args.local_rank, work.peer, channel, step, work.fifo_depth, true);
        v2CopyFragmentsToContiguous(args, work, payload, cursor, slice_bytes, tid, nworkers);
      }
    }

    // Like NCCL SIMPLE send, a wide group reserves its final warp for Post.
    // Workers can begin the next step while Post fences and publishes this one.
    v2GroupBarrier(main_barrier, nthreads);
    if (tid == 0) {
      profile->copy_cycles += clock64() - copy_start;
      ++profile->slice_count;
    }
    if ((roles & kRolePostSend) && v2LoadError(error) == 0) {
      const unsigned long long post_start = clock64();
      slot->bytes = static_cast<std::uint32_t>(slice_bytes);
      v2Publish(&slot->ready_step, step);
      profile->post_cycles += clock64() - post_start;
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }

  if (work.final && work.nbytes != 0) {
    V2FifoSlot* slot =
      v2FifoSlot(args, args.local_rank, work.peer, channel, step - 1, work.fifo_depth, true);
    if (roles & kRoleWaitSend) {
      const unsigned long long wait_start = clock64();
      *ready = v2WaitConsumed(slot, step - 1, step_cache, error, args.timeout_cycles);
      profile->final_wait_cycles += clock64() - wait_start;
    }
    v2GroupBarrier(main_barrier, nthreads);
  }
}

__device__ __forceinline__ void v2RunRecv(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel, int tid,
                                          int nthreads, int barrier, int* ready,
                                          unsigned long long* step_cache, V2KernelProfile* profile) {
  int nworkers = 0;
  const std::uint32_t roles = v2Roles(V2Direction::kRecv, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      work.step_bytes < work.nbytes - cursor ? work.step_bytes : work.nbytes - cursor;
    V2FifoSlot* slot = v2FifoSlot(args, work.peer, args.local_rank, channel, step, work.fifo_depth, false);
    if (roles & kRoleWaitRecv) {
      const unsigned long long wait_start = clock64();
      *ready = v2WaitReady(&slot->ready_step, step, step_cache, error, args.timeout_cycles);
      profile->input_wait_cycles += clock64() - wait_start;
    }
    const unsigned long long copy_start = tid == 0 ? clock64() : 0;
    v2GroupBarrier(barrier, nthreads);
    if (*ready && (roles & kRoleWorker)) {
      const std::uint8_t* payload =
        v2FifoPayload(args, work.peer, args.local_rank, channel, step, work.fifo_depth, false);
      v2CopyContiguousToFragments(args, work, payload, cursor, slice_bytes, tid, nworkers);
    }

    v2GroupBarrier(barrier, nthreads);
    if (tid == 0) {
      profile->copy_cycles += clock64() - copy_start;
      ++profile->slice_count;
    }
    if ((roles & kRolePostRecv) && v2LoadError(error) == 0) {
      const unsigned long long post_start = clock64();
      v2Publish(&slot->consumed_step, step);
      profile->post_cycles += clock64() - post_start;
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }
}

// A ring receiver consumes one FIFO step and republishes that same step to its
// successor before returning credit to its predecessor. Both LSA and GIN edges
// use the same route channel and byte partition, so mixed intra/inter-node rings
// retain NCCL broadcast's chunk-pipelined behavior.
__device__ __forceinline__ void v2RunRelay(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                           int tid, int nthreads, int barrier, int* ready,
                                           unsigned long long* input_cache, unsigned long long* output_cache,
                                           V2KernelProfile* profile) {
  int nworkers = 0;
  const std::uint32_t roles = v2Roles(V2Direction::kRecv, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  const bool input_gin = args.peer_transports[work.peer] == static_cast<std::uint8_t>(V2Transport::kGin);
  const bool output_gin =
    args.peer_transports[work.forward_peer] == static_cast<std::uint8_t>(V2Transport::kGin);
  std::uint64_t cursor = 0;
  std::uint64_t input_step = work.step_begin;
  std::uint64_t output_step = work.forward_step_begin;

  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      work.step_bytes < work.nbytes - cursor ? work.step_bytes : work.nbytes - cursor;
    V2FifoSlot* input_slot = input_gin
      ? nullptr
      : v2FifoSlot(args, work.peer, args.local_rank, channel, input_step, work.fifo_depth, false);
    V2FifoSlot* output_slot = output_gin
      ? nullptr
      : v2FifoSlot(args, args.local_rank, work.forward_peer, channel, output_step, work.fifo_depth, true);

    if (roles & kRoleWaitRecv) {
      bool input_ready = false;
      bool output_ready = false;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
      ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
      unsigned long long wait_start = clock64();
      input_ready = input_gin
        ? v2GinWaitSignal(args, gin, v2GinReadySignal(args, work.peer, channel), input_step, 2U)
        : v2WaitReady(&input_slot->ready_step, input_step, input_cache, error, args.timeout_cycles);
      profile->input_wait_cycles += clock64() - wait_start;
      wait_start = clock64();
      output_ready = output_gin
        ? (output_step <= work.fifo_depth ||
           v2GinWaitSignal(args, gin, v2GinCreditSignal(args, work.forward_peer, channel),
                           output_step - work.fifo_depth, 3U))
        : v2WaitFree(output_slot, output_step, work.fifo_depth, output_cache, error, args.timeout_cycles);
      profile->output_wait_cycles += clock64() - wait_start;
#else
      unsigned long long wait_start = clock64();
      input_ready = !input_gin &&
        v2WaitReady(&input_slot->ready_step, input_step, input_cache, error, args.timeout_cycles);
      profile->input_wait_cycles += clock64() - wait_start;
      wait_start = clock64();
      output_ready = !output_gin &&
        v2WaitFree(output_slot, output_step, work.fifo_depth, output_cache, error, args.timeout_cycles);
      profile->output_wait_cycles += clock64() - wait_start;
#endif
      *ready = input_ready && output_ready;
    }
    const unsigned long long copy_start = tid == 0 ? clock64() : 0;
    v2GroupBarrier(barrier, nthreads);

    const std::uint8_t* input_payload = nullptr;
    std::uint8_t* output_payload = nullptr;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
    if (input_gin) input_payload = v2GinLocalPayload(args, work.peer, channel, input_step, work.fifo_depth);
    if (output_gin) {
      output_payload = v2GinLocalPayload(args, work.forward_peer, channel, output_step, work.fifo_depth);
    }
#endif
    if (!input_gin) {
      input_payload = v2FifoPayload(args, work.peer, args.local_rank, channel, input_step, work.fifo_depth, false);
    }
    if (!output_gin) {
      output_payload =
        v2FifoPayload(args, args.local_rank, work.forward_peer, channel, output_step, work.fifo_depth, true);
    }
    if (*ready && (roles & kRoleWorker)) {
      v2CopyContiguousToFragmentsAndContiguous(args, work, output_payload, input_payload, cursor, slice_bytes, tid,
                                               nworkers);
    }

    v2GroupBarrier(barrier, nthreads);
    if (tid == 0) {
      profile->copy_cycles += clock64() - copy_start;
      ++profile->slice_count;
    }
    if ((roles & kRolePostRecv) && v2LoadError(error) == 0) {
      const unsigned long long post_start = clock64();
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
      ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
      const ncclTeam world = ncclTeamWorld(args.dev_comm);
      if (input_gin) {
        const std::uint64_t work_step = input_step - work.step_begin + 1;
        const bool work_complete = cursor + slice_bytes == work.nbytes;
        const std::uint32_t returned_credits = static_cast<std::uint32_t>(
          work_complete && work_step % args.gin_credit_batch != 0 ? work_step % args.gin_credit_batch
                                                                  : args.gin_credit_batch);
        if (work_complete || work_step % args.gin_credit_batch == 0) {
          gin.signal(world, work.peer,
                     V2GinCreditSignalAdd{v2GinCreditSignal(args, args.local_rank, channel), returned_credits});
        }
      } else {
        v2Publish(&input_slot->consumed_step, input_step);
      }

      // The input is no longer needed after the fused local copy. Release it
      // before a potentially backpressured forward put so congestion on the
      // successor does not unnecessarily stall the predecessor.
      if (output_gin) {
        const std::uint64_t work_step = output_step - work.forward_step_begin + 1;
        const bool work_complete = cursor + slice_bytes == work.nbytes;
        const bool publish_ready = work_complete || work_step % kV2GinReadyBatch == 0;
        gin.put(world, work.forward_peer, args.window,
                v2GinPayloadOffset(args, work.forward_peer, args.local_rank, channel, output_step, work.fifo_depth),
                args.window,
                v2GinPayloadOffset(args, args.local_rank, work.forward_peer, channel, output_step, work.fifo_depth),
                slice_bytes, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
                cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault);
        if (publish_ready) {
          const std::uint64_t ready_count =
            work_complete && work_step % kV2GinReadyBatch != 0 ? work_step % kV2GinReadyBatch : kV2GinReadyBatch;
          gin.signal(world, work.forward_peer,
                     V2GinReadySignalAdd{v2GinReadySignal(args, args.local_rank, channel), ready_count});
        }
      } else {
        output_slot->bytes = static_cast<std::uint32_t>(slice_bytes);
        v2Publish(&output_slot->ready_step, output_step);
      }
#else
      v2Publish(&input_slot->consumed_step, input_step);
      output_slot->bytes = static_cast<std::uint32_t>(slice_bytes);
      v2Publish(&output_slot->ready_step, output_step);
#endif
      profile->post_cycles += clock64() - post_start;
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++input_step;
    ++output_step;
  }

  if (work.final && work.nbytes != 0) {
    if (roles & kRoleWaitRecv) {
      const unsigned long long wait_start = clock64();
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
      if (output_gin) {
        ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
        *ready = v2GinWaitSignal(args, gin, v2GinCreditSignal(args, work.forward_peer, channel), output_step - 1, 4U);
      } else
#endif
      {
        V2FifoSlot* final_slot =
          v2FifoSlot(args, args.local_rank, work.forward_peer, channel, output_step - 1, work.fifo_depth, true);
        *ready = v2WaitConsumed(final_slot, output_step - 1, output_cache, error, args.timeout_cycles);
      }
      profile->final_wait_cycles += clock64() - wait_start;
    }
    v2GroupBarrier(barrier, nthreads);
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
      shared.forward_step_cache[group] = 0;
    }
    v2GroupBarrier(main_barrier, subthreads);

    const V2Work& work = args.works[batch.work_begin + group];
    V2KernelProfile* profile = args.profiles + static_cast<std::size_t>(blockIdx.x) * kMaxWorksPerBatch + group;
    const bool use_gin = args.peer_transports[work.peer] == static_cast<std::uint8_t>(V2Transport::kGin);
    if (work.forward_peer != kNoPeer) {
      v2RunRelay(args, work, channel, subtid, subthreads, main_barrier, &shared.ready[group],
                 &shared.step_cache[group], &shared.forward_step_cache[group], profile);
    } else if (use_gin) {
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
      if (args.direction == V2Direction::kSend) {
        v2GinRunSend(args, work, channel, subtid, subthreads, main_barrier, wait_barrier, &shared.ready[group],
                     profile);
      } else {
        v2GinRunRecv(args, work, channel, subtid, subthreads, main_barrier, &shared.ready[group], profile);
      }
#else
      if (subtid == 0) {
        auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
        atomicCAS(error, 0U, 5U);
      }
      v2GroupBarrier(main_barrier, subthreads);
#endif
    } else if (args.direction == V2Direction::kSend) {
      v2RunSend(args, work, channel, subtid, subthreads, main_barrier, wait_barrier, &shared.ready[group],
                &shared.step_cache[group], profile);
    } else {
      v2RunRecv(args, work, channel, subtid, subthreads, main_barrier, &shared.ready[group],
                &shared.step_cache[group], profile);
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
      if (args.peer_transports[peer] == static_cast<std::uint8_t>(V2Transport::kGin)) continue;
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
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  if (args.gin_enabled != 0) {
    ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
    const unsigned long long flush_start = clock64();
    gin.flush(ncclCoopCta());
    if (threadIdx.x == 0) {
      V2KernelProfile* profile = args.profiles + static_cast<std::size_t>(blockIdx.x) * kMaxWorksPerBatch;
      profile->flush_cycles += clock64() - flush_start;
    }
  }
#endif
}

}  // namespace nccl_device_v2
}  // namespace awex

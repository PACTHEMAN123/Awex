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

#if AWEX_NCCL_DEVICE_V2_HAS_GIN

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 7)
using V2GinReadySignalAdd = ncclGin_StrongSignalAdd;
using V2GinCreditSignalAdd = ncclGin_WeakSignalAdd;
#else
// NCCL 2.30.4 signalling puts have the strong ordering semantics later made
// explicit by StrongSignalAdd: the signal follows all earlier puts to the
// same peer on the same context.
using V2GinReadySignalAdd = ncclGin_SignalAdd;
using V2GinCreditSignalAdd = ncclGin_SignalAdd;
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 7)
constexpr ncclGinFenceLevel kV2GinNoFence = ncclGinFenceLevel::None;
#else
constexpr ncclGinFenceLevel kV2GinNoFence = ncclGinFenceLevel::Relaxed;
#endif

__device__ __forceinline__ ncclGinSignal_t v2GinReadySignal(const V2KernelArgs& args, std::uint32_t peer,
                                                            std::uint32_t channel) {
  return static_cast<ncclGinSignal_t>(static_cast<std::size_t>(peer) * args.layout.channel_count + channel);
}

__device__ __forceinline__ ncclGinSignal_t v2GinCreditSignal(const V2KernelArgs& args, std::uint32_t peer,
                                                             std::uint32_t channel) {
  const std::size_t ready_count = static_cast<std::size_t>(args.world_size) * args.layout.channel_count;
  return static_cast<ncclGinSignal_t>(ready_count + static_cast<std::size_t>(peer) * args.layout.channel_count +
                                      channel);
}

__device__ __forceinline__ std::size_t v2GinPayloadOffset(const V2KernelArgs& args, std::uint32_t window_rank,
                                                          std::uint32_t peer, std::uint32_t channel,
                                                          unsigned long long step, std::uint32_t fifo_depth) {
  const std::uint32_t payload_slot =
    args.payload_peer_slots[static_cast<std::size_t>(window_rank) * args.world_size + peer];
  const std::size_t connection = static_cast<std::size_t>(payload_slot) * args.layout.channel_count + channel;
  const std::size_t slot =
    connection * args.layout.fifo_depth + static_cast<std::size_t>(step % fifo_depth);
  return args.layout.payload_offset + slot * args.layout.slot_bytes;
}

__device__ __forceinline__ std::uint8_t* v2GinLocalPayload(const V2KernelArgs& args, std::uint32_t peer,
                                                           std::uint32_t channel, unsigned long long step,
                                                           std::uint32_t fifo_depth) {
  return args.local_window + v2GinPayloadOffset(args, args.local_rank, peer, channel, step, fifo_depth);
}

// NCCL's rail ring submits contiguous chunks from registered storage. Apply
// that progression to adjacent existing FIFO slots, not a new receive buffer.
// Keep at least half the window available for producer/consumer overlap and
// batched credits. Padded slots and small windows retain single-slice puts.
__device__ __forceinline__ std::uint32_t v2GinPutSteps(const V2KernelArgs& args, const V2Work& work) {
  return work.ring_id != kNoRing && work.step_bytes == args.layout.slot_bytes && work.fifo_depth >= 4
    ? work.fifo_depth / 2 : 1;
}

struct V2GinFifoPut {
  std::uint64_t source_step = 0;
  std::uint64_t destination_step = 0;
  std::uint64_t bytes = 0;
  std::uint32_t steps = 0;

  template <typename Coop>
  __device__ __forceinline__ void append(const V2KernelArgs& args, const V2Work& work,
      const ncclGin& gin, std::uint32_t peer, std::uint32_t channel, std::uint32_t source_peer,
      std::uint64_t source, std::uint64_t destination, std::uint64_t slice_bytes,
      std::uint64_t slice_index, bool complete, Coop coop) {
    if (steps == 0) {
      source_step = source;
      destination_step = destination;
    }
    ++steps;
    bytes += slice_bytes;
    // A put must not cross either physical modulo-FIFO boundary. The last
    // partial slice is submitted at the work boundary, never joined to padding.
    // Anchor groups to the work's logical slice index, not to the preceding
    // wrap flush. Otherwise a short wrap batch can shift the next publication
    // past the window boundary while the receiver is waiting to retire it.
    if ((slice_index + 1) % v2GinPutSteps(args, work) != 0 && !complete &&
        (source + 1) % work.fifo_depth != 0 && (destination + 1) % work.fifo_depth != 0) return;
    gin.put(ncclTeamWorld(args.dev_comm), peer, args.window,
            v2GinPayloadOffset(args, peer, args.local_rank, channel, destination_step, work.fifo_depth),
            args.window,
            v2GinPayloadOffset(args, args.local_rank, source_peer, channel, source_step, work.fifo_depth),
            bytes, V2GinReadySignalAdd{v2GinReadySignal(args, args.local_rank, channel), steps},
            ncclGin_None{}, coop, ncclGin_None{}, cuda::thread_scope_thread,
            cuda::thread_scope_device, ncclGinOptFlagsDefault);
    steps = 0;
    bytes = 0;
  }
};

__device__ __forceinline__ bool v2GinWaitSignal(const V2KernelArgs& args, const ncclGin& gin, ncclGinSignal_t signal,
                                                unsigned long long expected, unsigned int error_code) {
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  const unsigned long long start = clock64();
  while (gin.readSignal(signal) < expected) {
    if (v2LoadError(error) != 0) return false;
    if (clock64() - start > args.timeout_cycles) {
      atomicCAS(error, 0U, error_code);
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ std::uint32_t v2GinRoles(V2Direction direction, int tid, int nthreads, int* nworkers) {
  const bool send = direction == V2Direction::kSend;
  *nworkers = nthreads - (nthreads >= 3 * kWarpSize ? kWarpSize : 0);
  std::uint32_t roles = tid < *nworkers ? kRoleWorker : 0;
  if (tid == 0) roles |= send ? kRoleWaitSend : kRoleWaitRecv;
  if (tid == nthreads - 1) roles |= send ? kRolePostSend : kRolePostRecv;
  return roles;
}

__device__ __forceinline__ void v2GinRunSend(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                             int tid, int nthreads, int main_barrier, int wait_barrier, int* ready,
                                             V2KernelProfile* profile) {
  int nworkers = 0;
  const std::uint32_t roles = v2GinRoles(V2Direction::kSend, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
  const ncclGinSignal_t credit_signal = v2GinCreditSignal(args, work.peer, channel);
  V2GinFifoPut fifo_put;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      work.step_bytes < work.nbytes - cursor ? work.step_bytes : work.nbytes - cursor;
    if ((roles & kRoleWaitSend) && step > work.fifo_depth) {
      const unsigned long long wait_start = clock64();
      *ready = v2GinWaitSignal(args, gin, credit_signal, step - work.fifo_depth, 3U);
      profile->output_wait_cycles += clock64() - wait_start;
    }
    const unsigned long long copy_start = tid == 0 ? clock64() : 0;
    if (roles & kRoleWorker) {
      v2GroupBarrier(wait_barrier, nworkers);
      if (*ready) {
        v2CopyFragmentsToContiguous(
          args, work, v2GinLocalPayload(args, work.peer, channel, step, work.fifo_depth), cursor, slice_bytes, tid,
          nworkers);
      }
    }

    v2GroupBarrier(main_barrier, nthreads);
    if (tid == 0) {
      profile->copy_cycles += clock64() - copy_start;
      ++profile->slice_count;
    }
    if ((roles & kRolePostSend) && v2LoadError(error) == 0) {
      const unsigned long long post_start = clock64();
      // The cumulative ready counter requires ordered completion so a later
      // put cannot satisfy the wait for an earlier FIFO step.
      fifo_put.append(args, work, gin, work.peer, channel, work.peer, step, step, slice_bytes,
                      cursor / work.step_bytes, cursor + slice_bytes == work.nbytes, ncclCoopThread{});
      profile->post_cycles += clock64() - post_start;
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }

  if (work.final && work.nbytes != 0) {
    if (roles & kRoleWaitSend) {
      const unsigned long long wait_start = clock64();
      *ready = v2GinWaitSignal(args, gin, credit_signal, step - 1, 4U);
      profile->final_wait_cycles += clock64() - wait_start;
    }
    v2GroupBarrier(main_barrier, nthreads);
  }
}

__device__ __forceinline__ void v2GinRunRecv(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                             int tid, int nthreads, int barrier, int* ready,
                                             V2KernelProfile* profile) {
  int nworkers = 0;
  const std::uint32_t roles = v2GinRoles(V2Direction::kRecv, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
  const ncclTeam world = ncclTeamWorld(args.dev_comm);
  const ncclGinSignal_t ready_signal = v2GinReadySignal(args, work.peer, channel);
  const ncclGinSignal_t credit_signal = v2GinCreditSignal(args, args.local_rank, channel);
  const std::uint32_t credit_batch = args.gin_credit_batch;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      work.step_bytes < work.nbytes - cursor ? work.step_bytes : work.nbytes - cursor;
    if (roles & kRoleWaitRecv) {
      const unsigned long long wait_start = clock64();
      *ready = v2GinWaitSignal(args, gin, ready_signal, step, 2U);
      profile->input_wait_cycles += clock64() - wait_start;
    }
    const unsigned long long copy_start = tid == 0 ? clock64() : 0;
    v2GroupBarrier(barrier, nthreads);
    if (*ready && (roles & kRoleWorker)) {
      v2CopyContiguousToFragments(
        args, work, v2GinLocalPayload(args, work.peer, channel, step, work.fifo_depth), cursor, slice_bytes, tid,
        nworkers);
    }

    v2GroupBarrier(barrier, nthreads);
    if (tid == 0) {
      profile->copy_cycles += clock64() - copy_start;
      ++profile->slice_count;
    }
    const std::uint64_t work_step = step - work.step_begin + 1;
    const bool work_complete = cursor + slice_bytes == work.nbytes;
    const std::uint32_t returned_credits = static_cast<std::uint32_t>(
      work_complete && work_step % credit_batch != 0 ? work_step % credit_batch : credit_batch);
    if ((roles & kRolePostRecv) && v2LoadError(error) == 0 &&
        (work_complete || work_step % credit_batch == 0)) {
      const unsigned long long post_start = clock64();
      gin.signal(world, work.peer, V2GinCreditSignalAdd{credit_signal, returned_credits});
      profile->post_cycles += clock64() - post_start;
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }
}

__global__ void v2GinResetSignalsKernel(V2KernelArgs args) {
  ncclCoopCta coop;
  for (std::uint32_t context = 0; context < args.dev_comm.ginContextCount; ++context) {
    ncclGin gin{args.dev_comm, static_cast<int>(context)};
    for (std::uint32_t signal = threadIdx.x; signal < args.gin_signal_count; signal += blockDim.x) {
      gin.resetSignal(static_cast<ncclGinSignal_t>(signal));
    }
    coop.sync();
  }
  ncclGin barrier_gin{args.dev_comm, 0};
  ncclGinBarrierSession<ncclCoopCta> barrier{coop, barrier_gin, ncclTeamTagWorld(), 0};
  const ncclResult_t result =
    barrier.sync(coop, cuda::memory_order_acq_rel, kV2GinNoFence, args.timeout_cycles);
  if (threadIdx.x == 0 && result != ncclSuccess) {
    auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
    atomicCAS(error, 0U, 1U);
  }
}

#endif

}  // namespace nccl_device_v2
}  // namespace awex

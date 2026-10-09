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

#include "primitives.cuh"

namespace shardstream {
namespace transport {

#if SHARDSTREAM_HAS_GIN

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 7)
using GinReadySignalAdd = ncclGin_StrongSignalAdd;
using GinCreditSignalAdd = ncclGin_WeakSignalAdd;
#else
// NCCL 2.30.4 signalling puts have the strong ordering semantics later made
// explicit by StrongSignalAdd: the signal follows all earlier puts to the
// same peer on the same context.
using GinReadySignalAdd = ncclGin_SignalAdd;
using GinCreditSignalAdd = ncclGin_SignalAdd;
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 7)
constexpr ncclGinFenceLevel kGinNoFence = ncclGinFenceLevel::None;
#else
constexpr ncclGinFenceLevel kGinNoFence = ncclGinFenceLevel::Relaxed;
#endif

__device__ __forceinline__ ncclGinSignal_t ginReadySignal(const KernelArgs& args, std::uint32_t peer,
                                                            std::uint32_t channel) {
  return static_cast<ncclGinSignal_t>(static_cast<std::size_t>(peer) * args.layout.channel_count + channel);
}

__device__ __forceinline__ ncclGinSignal_t ginCreditSignal(const KernelArgs& args, std::uint32_t peer,
                                                             std::uint32_t channel) {
  const std::size_t ready_count = static_cast<std::size_t>(args.world_size) * args.layout.channel_count;
  return static_cast<ncclGinSignal_t>(ready_count + static_cast<std::size_t>(peer) * args.layout.channel_count +
                                      channel);
}

__device__ __forceinline__ std::size_t ginPayloadOffset(const KernelArgs& args, std::uint32_t window_rank,
                                                          std::uint32_t peer, std::uint32_t channel,
                                                          unsigned long long step, std::uint32_t fifo_depth) {
  const std::uint32_t payload_slot =
    args.payload_peer_slots[static_cast<std::size_t>(window_rank) * args.world_size + peer];
  const std::size_t connection = static_cast<std::size_t>(payload_slot) * args.layout.channel_count + channel;
  const std::size_t slot =
    connection * args.layout.fifo_depth + static_cast<std::size_t>(step % fifo_depth);
  return args.layout.payload_offset + slot * args.layout.slot_bytes;
}

__device__ __forceinline__ std::uint8_t* ginLocalPayload(const KernelArgs& args, std::uint32_t peer,
                                                           std::uint32_t channel, unsigned long long step,
                                                           std::uint32_t fifo_depth) {
  return args.local_window + ginPayloadOffset(args, args.local_rank, peer, channel, step, fifo_depth);
}

// NCCL's rail ring submits contiguous chunks from registered storage. Apply
// that progression to adjacent existing FIFO slots, not a new receive buffer.
// Keep at least half the window available for producer/consumer overlap and
// batched credits. Padded slots and small windows retain single-slice puts.
__device__ __forceinline__ std::uint32_t ginPutSteps(
    const KernelArgs& args, const Work& work, bool quantized_source = false) {
  // Large direct fanouts share each rail across many independently credited
  // receivers. Publish each completed slice promptly instead of reserving
  // half every FIFO before making any of those receivers runnable. Relay
  // rings retain their contiguous half-window puts.
  if (work.ring_id == kNoRing && quantized_source && args.active_peer_count >= 8) return 1;
  return (work.ring_id != kNoRing || quantized_source) &&
      work.step_bytes == args.layout.slot_bytes && work.fifo_depth >= 4
    ? work.fifo_depth / 2 : 1;
}

struct GinFifoPut {
  std::uint64_t source_step = 0;
  std::uint64_t destination_step = 0;
  std::uint64_t bytes = 0;
  std::uint32_t steps = 0;

  template <typename Coop>
  __device__ __forceinline__ void append(const KernelArgs& args, const Work& work,
      const ncclGin& gin, std::uint32_t peer, std::uint32_t channel, std::uint32_t source_peer,
      std::uint64_t source, std::uint64_t destination, std::uint64_t slice_bytes,
      std::uint64_t slice_index, bool complete, Coop coop, bool quantized_source = false) {
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
    if ((slice_index + 1) % ginPutSteps(args, work, quantized_source) != 0 && !complete &&
        (source + 1) % work.fifo_depth != 0 && (destination + 1) % work.fifo_depth != 0) return;
    gin.put(ncclTeamWorld(args.dev_comm), peer, args.window,
            ginPayloadOffset(args, peer, args.local_rank, channel, destination_step, work.fifo_depth),
            args.window,
            ginPayloadOffset(args, args.local_rank, source_peer, channel, source_step, work.fifo_depth),
            bytes, GinReadySignalAdd{ginReadySignal(args, args.local_rank, channel), steps},
            ncclGin_None{}, coop, ncclGin_None{}, cuda::thread_scope_thread,
            cuda::thread_scope_device, ncclGinOptFlagsDefault);
    steps = 0;
    bytes = 0;
  }
};

__device__ __forceinline__ bool ginWaitSignal(const KernelArgs& args, const ncclGin& gin, ncclGinSignal_t signal,
                                                unsigned long long expected, unsigned int error_code) {
  auto* error = &reinterpret_cast<WindowHeader*>(args.local_window)->error;
  const unsigned long long start = clock64();
  while (gin.readSignal(signal) < expected) {
    if (loadError(error) != 0) return false;
    if (clock64() - start > args.timeout_cycles) {
      atomicCAS(error, 0U, error_code);
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ std::uint32_t ginRoles(Direction direction, int tid, int nthreads, int* nworkers) {
  const bool send = direction == Direction::kSend;
  *nworkers = nthreads - (nthreads >= 3 * kWarpSize ? kWarpSize : 0);
  std::uint32_t roles = tid < *nworkers ? kRoleWorker : 0;
  if (tid == 0) roles |= send ? kRoleWaitSend : kRoleWaitRecv;
  if (tid == nthreads - 1) roles |= send ? kRolePostSend : kRolePostRecv;
  return roles;
}

__device__ __forceinline__ void ginRunSend(const KernelArgs& args, const Work& work, std::uint32_t channel,
                                             int tid, int nthreads, int main_barrier, int wait_barrier, int* ready,
                                             KernelProfile* profile) {
  int nworkers = 0;
  const std::uint32_t roles = ginRoles(Direction::kSend, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<WindowHeader*>(args.local_window)->error;
  ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
  const ncclGinSignal_t credit_signal = ginCreditSignal(args, work.peer, channel);
  // Direct FP8 fanout also benefits from submitting adjacent existing FIFO
  // slots as one put. Source stores complete before publication, and the
  // cumulative ready signal covers every copied slot in the group. Keep the
  // copy-only source's original single-step submission and all ring routes.
  bool quantized_source = false;
  // Only the publishing thread needs this classification, and only where
  // direct coalescing can actually be enabled. Large model works contain
  // thousands of descriptors; walking them in every copy thread wastes
  // bandwidth even for legacy padded slots and already-classified rings.
  if ((roles & kRolePostSend) && work.ring_id == kNoRing && args.active_peer_count < 8 &&
      work.step_bytes == args.layout.slot_bytes && work.fifo_depth >= 4) {
    for (std::uint32_t index = 0; index < work.fragment_count && !quantized_source; ++index)
      quantized_source = args.fragments[work.fragment_begin + index].block_rows != 0;
  }
  GinFifoPut fifo_put;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      work.step_bytes < work.nbytes - cursor ? work.step_bytes : work.nbytes - cursor;
    if ((roles & kRoleWaitSend) && step > work.fifo_depth) {
      const unsigned long long wait_start = clock64();
      *ready = ginWaitSignal(args, gin, credit_signal, step - work.fifo_depth, 3U);
      profile->output_wait_cycles += clock64() - wait_start;
    }
    const unsigned long long copy_start = tid == 0 ? clock64() : 0;
    if (roles & kRoleWorker) {
      groupBarrier(wait_barrier, nworkers);
      if (*ready) {
        copyFragmentsToContiguous(
          args, work, ginLocalPayload(args, work.peer, channel, step, work.fifo_depth), cursor, slice_bytes, tid,
          nworkers);
      }
    }

    groupBarrier(main_barrier, nthreads);
    if (tid == 0) {
      profile->copy_cycles += clock64() - copy_start;
      ++profile->slice_count;
    }
    if ((roles & kRolePostSend) && loadError(error) == 0) {
      const unsigned long long post_start = clock64();
      // The cumulative ready counter requires ordered completion so a later
      // put cannot satisfy the wait for an earlier FIFO step.
      fifo_put.append(args, work, gin, work.peer, channel, work.peer, step, step, slice_bytes,
                      cursor / work.step_bytes, cursor + slice_bytes == work.nbytes, ncclCoopThread{},
                      quantized_source);
      profile->post_cycles += clock64() - post_start;
    }
    if (loadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }

  if (work.final && work.nbytes != 0) {
    if (roles & kRoleWaitSend) {
      const unsigned long long wait_start = clock64();
      *ready = ginWaitSignal(args, gin, credit_signal, step - 1, 4U);
      profile->final_wait_cycles += clock64() - wait_start;
    }
    groupBarrier(main_barrier, nthreads);
  }
}

__device__ __forceinline__ void ginRunRecv(const KernelArgs& args, const Work& work, std::uint32_t channel,
                                             int tid, int nthreads, int barrier, int* ready,
                                             KernelProfile* profile) {
  int nworkers = 0;
  const std::uint32_t roles = ginRoles(Direction::kRecv, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<WindowHeader*>(args.local_window)->error;
  ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
  const ncclTeam world = ncclTeamWorld(args.dev_comm);
  const ncclGinSignal_t ready_signal = ginReadySignal(args, work.peer, channel);
  const ncclGinSignal_t credit_signal = ginCreditSignal(args, args.local_rank, channel);
  const std::uint32_t credit_batch = args.gin_credit_batch;
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      work.step_bytes < work.nbytes - cursor ? work.step_bytes : work.nbytes - cursor;
    if (roles & kRoleWaitRecv) {
      const unsigned long long wait_start = clock64();
      *ready = ginWaitSignal(args, gin, ready_signal, step, 2U);
      profile->input_wait_cycles += clock64() - wait_start;
    }
    const unsigned long long copy_start = tid == 0 ? clock64() : 0;
    groupBarrier(barrier, nthreads);
    if (*ready && (roles & kRoleWorker)) {
      copyContiguousToFragments(
        args, work, ginLocalPayload(args, work.peer, channel, step, work.fifo_depth), cursor, slice_bytes, tid,
        nworkers);
    }

    groupBarrier(barrier, nthreads);
    if (tid == 0) {
      profile->copy_cycles += clock64() - copy_start;
      ++profile->slice_count;
    }
    const std::uint64_t work_step = step - work.step_begin + 1;
    const bool work_complete = cursor + slice_bytes == work.nbytes;
    const std::uint32_t returned_credits = static_cast<std::uint32_t>(
      work_complete && work_step % credit_batch != 0 ? work_step % credit_batch : credit_batch);
    if ((roles & kRolePostRecv) && loadError(error) == 0 &&
        (work_complete || work_step % credit_batch == 0)) {
      const unsigned long long post_start = clock64();
      gin.signal(world, work.peer, GinCreditSignalAdd{credit_signal, returned_credits});
      profile->post_cycles += clock64() - post_start;
    }
    if (loadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }
}

__global__ void ginResetSignalsKernel(KernelArgs args) {
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
    barrier.sync(coop, cuda::memory_order_acq_rel, kGinNoFence, args.timeout_cycles);
  if (threadIdx.x == 0 && result != ncclSuccess) {
    auto* error = &reinterpret_cast<WindowHeader*>(args.local_window)->error;
    atomicCAS(error, 0U, 1U);
  }
}

#endif

}  // namespace transport
}  // namespace shardstream

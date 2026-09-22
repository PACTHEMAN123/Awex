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

#if AWEX_NCCL_DEVICE_V2_HAS_EXPLICIT_SIGNAL_STRENGTH
using V2GinReadySignalInc = ncclGin_StrongSignalInc;
using V2GinCreditSignalInc = ncclGin_WeakSignalInc;
#else
// NCCL 2.30.4 SignalInc has the strong ordering semantics later made
// explicit by StrongSignalInc: the signal follows all earlier puts to the
// same peer on the same context.
using V2GinReadySignalInc = ncclGin_SignalInc;
using V2GinCreditSignalInc = ncclGin_SignalInc;
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
                                                          unsigned long long step) {
  const std::uint32_t payload_slot =
    args.payload_peer_slots[static_cast<std::size_t>(window_rank) * args.world_size + peer];
  const std::size_t connection = static_cast<std::size_t>(payload_slot) * args.layout.channel_count + channel;
  const std::size_t slot =
    connection * args.layout.fifo_depth + static_cast<std::size_t>(step % args.layout.fifo_depth);
  return args.layout.payload_offset + slot * args.layout.slot_bytes;
}

__device__ __forceinline__ std::uint8_t* v2GinLocalPayload(const V2KernelArgs& args, std::uint32_t peer,
                                                           std::uint32_t channel, unsigned long long step) {
  return args.local_window + v2GinPayloadOffset(args, args.local_rank, peer, channel, step);
}

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
                                             int tid, int nthreads, int main_barrier, int wait_barrier, int* ready) {
  int nworkers = 0;
  const std::uint32_t roles = v2GinRoles(V2Direction::kSend, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
  const ncclTeam world = ncclTeamWorld(args.dev_comm);
  const ncclGinSignal_t ready_signal = v2GinReadySignal(args, args.local_rank, channel);
  const ncclGinSignal_t credit_signal = v2GinCreditSignal(args, work.peer, channel);
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    if ((roles & kRoleWaitSend) && step > args.layout.fifo_depth) {
      *ready = v2GinWaitSignal(args, gin, credit_signal, step - args.layout.fifo_depth, 3U);
    }
    if (roles & kRoleWorker) {
      v2GroupBarrier(wait_barrier, nworkers);
      if (*ready) {
        v2CopyFragmentsToContiguous(args, work, v2GinLocalPayload(args, work.peer, channel, step), cursor, slice_bytes,
                                    tid, nworkers);
      }
    }

    v2GroupBarrier(main_barrier, nthreads);
    if ((roles & kRolePostSend) && v2LoadError(error) == 0) {
      // The cumulative ready counter requires ordered completion so a later
      // put cannot satisfy the wait for an earlier FIFO step.
      gin.put(world, work.peer, args.window, v2GinPayloadOffset(args, work.peer, args.local_rank, channel, step),
              args.window, v2GinPayloadOffset(args, args.local_rank, work.peer, channel, step), slice_bytes,
              V2GinReadySignalInc{ready_signal});
    }
    if (v2LoadError(error) != 0) return;
    cursor += slice_bytes;
    ++step;
  }

  if (work.final && work.nbytes != 0) {
    if (roles & kRoleWaitSend) {
      *ready = v2GinWaitSignal(args, gin, credit_signal, step - 1, 4U);
    }
    v2GroupBarrier(main_barrier, nthreads);
  }
}

__device__ __forceinline__ void v2GinRunRecv(const V2KernelArgs& args, const V2Work& work, std::uint32_t channel,
                                             int tid, int nthreads, int barrier, int* ready) {
  int nworkers = 0;
  const std::uint32_t roles = v2GinRoles(V2Direction::kRecv, tid, nthreads, &nworkers);
  auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
  ncclGin gin{args.dev_comm, static_cast<int>(channel % args.dev_comm.ginContextCount)};
  const ncclTeam world = ncclTeamWorld(args.dev_comm);
  const ncclGinSignal_t ready_signal = v2GinReadySignal(args, work.peer, channel);
  const ncclGinSignal_t credit_signal = v2GinCreditSignal(args, args.local_rank, channel);
  std::uint64_t cursor = 0;
  std::uint64_t step = work.step_begin;
  while (cursor < work.nbytes) {
    const std::uint64_t slice_bytes =
      args.layout.slot_bytes < work.nbytes - cursor ? args.layout.slot_bytes : work.nbytes - cursor;
    if (roles & kRoleWaitRecv) {
      *ready = v2GinWaitSignal(args, gin, ready_signal, step, 2U);
    }
    v2GroupBarrier(barrier, nthreads);
    if (*ready && (roles & kRoleWorker)) {
      v2CopyContiguousToFragments(args, work, v2GinLocalPayload(args, work.peer, channel, step), cursor, slice_bytes,
                                  tid, nworkers);
    }

    v2GroupBarrier(barrier, nthreads);
    if ((roles & kRolePostRecv) && v2LoadError(error) == 0) {
      gin.signal(world, work.peer, V2GinCreditSignalInc{credit_signal});
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
    barrier.sync(coop, cuda::memory_order_acq_rel, ncclGinFenceLevel::None, args.timeout_cycles);
  if (threadIdx.x == 0 && result != ncclSuccess) {
    auto* error = &reinterpret_cast<V2WindowHeader*>(args.local_window)->error;
    atomicCAS(error, 0U, 1U);
  }
}

#endif

}  // namespace nccl_device_v2
}  // namespace awex

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

#include "copy_primitives.cuh"
#include "ring_slot.cuh"

namespace awex {
namespace device_transfer {

// A block owns one (peer, lane) pair for the whole launch.  Segment ordinals
// lane, lane + Q, ... reuse the same remote slot, so the number of resident
// blocks and the registered window stay bounded while a step can be arbitrary
// in size.
__global__ void __launch_bounds__(kThreadsPerBlock, 1) transfer_impl(
    DeviceTransferArgs args) {
  auto* local_control = reinterpret_cast<ControlBlock*>(args.local_base);
  const bool sender = args.role == TransferRole::kSender;

  // The epoch handshake prevents either side from touching a ring before its
  // peer has launched the same transfer generation.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (args.profile != nullptr) {
      args.profile->kernel_start_ns = global_timer_ns();
    }
    __threadfence_system();
    atomicExch_system(&local_control->epoch, args.sequence);
  }

  const std::uint32_t active_peer_index = blockIdx.x / args.slots_per_peer;
  const std::uint32_t lane = blockIdx.x % args.slots_per_peer;
  if (active_peer_index >= args.active_peer_count) {
    return;
  }

  const std::uint32_t peer = args.active_peers[active_peer_index];
  const std::uint32_t peer_task_begin = args.peer_offsets[peer];
  const std::uint32_t peer_task_count = args.expected_counts[peer];
  if (peer_task_count == 0 || peer_task_begin >= args.task_count) {
    return;
  }

  auto* remote_base = reinterpret_cast<std::uint8_t*>(
      args.tasks[peer_task_begin].remote_base);
  auto* remote_control = reinterpret_cast<ControlBlock*>(remote_base);

  __shared__ int ready;
  if (threadIdx.x == 0) {
    ready = wait_for_ticket(
        &local_control->epoch,
        args.sequence,
        &local_control->error,
        args.timeout_cycles);
    if (ready) {
      ready = wait_for_ticket(
          &remote_control->epoch,
          args.sequence,
          &local_control->error,
          args.timeout_cycles);
    }
  }
  __syncthreads();
  if (!ready) {
    return;
  }

  unsigned long long last_ticket = 0;
  RingSlotState* last_slot_state = nullptr;
  for (std::uint32_t ordinal = lane; ordinal < peer_task_count;
       ordinal += args.slots_per_peer) {
    const std::uint32_t task_index = peer_task_begin + ordinal;
    if (task_index >= args.task_count) {
      if (threadIdx.x == 0) {
        atomicExch_system(&local_control->error, 1U);
      }
      return;
    }

    const DeviceTask task = args.tasks[task_index];
    if (task.peer != peer || task.ordinal != ordinal ||
        task.nbytes > args.slot_bytes) {
      if (threadIdx.x == 0) {
        atomicExch_system(&local_control->error, 1U);
      }
      return;
    }

    const std::uint32_t channel =
        sender ? static_cast<std::uint32_t>(args.local_rank) : peer;
    RingSlot slot = ring_slot(
        sender ? remote_base : args.local_base,
        args.ring_data_offset,
        args.slot_bytes,
        channel,
        lane,
        args.slots_per_peer);
    auto* tensor_data = reinterpret_cast<std::uint8_t*>(task.tensor_ptr);
    const unsigned long long ticket = make_ticket(args.sequence, ordinal);

    if (threadIdx.x == 0) {
      if (sender) {
        ready = wait_until_free(
            slot.state, &local_control->error, args.timeout_cycles);
      } else {
        ready = wait_for_ticket(
            &slot.state->ready_ticket,
            ticket,
            &local_control->error,
            args.timeout_cycles);
      }
      if (ready && !sender && args.profile != nullptr) {
        const auto ready_ns = global_timer_ns();
        atomicMin(&args.profile->first_ready_ns, ready_ns);
        atomicMax(&args.profile->last_ready_ns, ready_ns);
      }
    }
    __syncthreads();
    if (!ready) {
      return;
    }
    if (!sender) {
      __threadfence_system();
    }

    if (threadIdx.x == 0 && args.profile != nullptr) {
      atomicMin(&args.profile->first_copy_ns, global_timer_ns());
    }
    __syncthreads();
    if (sender) {
      copy_contiguous_1d(slot.data, tensor_data, task.nbytes);
    } else {
      copy_contiguous_1d(tensor_data, slot.data, task.nbytes);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (args.profile != nullptr) {
        atomicMax(&args.profile->last_copy_ns, global_timer_ns());
      }
      if (sender) {
        if (args.profile != nullptr) {
          atomicMin(&args.profile->publish_start_ns, global_timer_ns());
        }
        mark_ready(slot.state, ticket);
        if (args.profile != nullptr) {
          atomicMax(&args.profile->publish_done_ns, global_timer_ns());
        }
      } else {
        mark_consumed(slot.state, ticket);
      }
    }
    __syncthreads();

    last_ticket = ticket;
    last_slot_state = slot.state;
  }

  // Reuse already acknowledges every earlier generation.  Waiting for the
  // lane's final ticket makes tensor lifetime safe when the kernel returns.
  if (sender && last_slot_state != nullptr && threadIdx.x == 0) {
    ready = wait_for_ticket(
        &last_slot_state->consumed_ticket,
        last_ticket,
        &local_control->error,
        args.timeout_cycles);
    if (ready && args.profile != nullptr) {
      atomicMax(&args.profile->peer_done_ns, global_timer_ns());
    }
  }
}

}  // namespace device_transfer
}  // namespace awex

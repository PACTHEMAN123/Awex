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

#include "device_transfer_types.cuh"

namespace awex {
namespace device_transfer {

struct RingSlot {
  // Both pointers address registered global memory.  Shared memory may cache
  // block-local control, but cannot carry producer/consumer state across GPUs.
  RingSlotState* state;
  std::uint8_t* data;
};

__device__ __forceinline__ unsigned long long load_system(
    volatile unsigned long long* address) {
  return atomicAdd_system(
      const_cast<unsigned long long*>(address), 0ULL);
}

__device__ __forceinline__ bool has_error(
    volatile unsigned int* error) {
  return atomicAdd(const_cast<unsigned int*>(error), 0U) != 0U;
}

__device__ __forceinline__ bool wait_for_ticket(
    volatile unsigned long long* address,
    unsigned long long ticket,
    volatile unsigned int* error,
    unsigned long long timeout_cycles) {
  const unsigned long long start = clock64();
  while (true) {
    if (load_system(address) == ticket) {
      return true;
    }
    if (has_error(error)) {
      return false;
    }
    if (clock64() - start > timeout_cycles) {
      atomicExch_system(const_cast<unsigned int*>(error), 1U);
      return false;
    }
  }
}

__device__ __forceinline__ bool wait_until_free(
    volatile RingSlotState* slot,
    volatile unsigned int* error,
    unsigned long long timeout_cycles) {
  const unsigned long long start = clock64();
  while (true) {
    if (load_system(&slot->ready_ticket) ==
        load_system(&slot->consumed_ticket)) {
      return true;
    }
    if (has_error(error)) {
      return false;
    }
    if (clock64() - start > timeout_cycles) {
      atomicExch_system(const_cast<unsigned int*>(error), 1U);
      return false;
    }
  }
}

__device__ __forceinline__ RingSlot ring_slot(
    std::uint8_t* window_base,
    std::size_t ring_data_offset,
    std::size_t slot_bytes,
    std::uint32_t channel,
    std::uint32_t lane,
    std::uint32_t slots_per_peer) {
  const std::size_t slot_index =
      static_cast<std::size_t>(channel) * slots_per_peer + lane;
  auto* states = reinterpret_cast<RingSlotState*>(
      window_base + sizeof(ControlBlock));
  return RingSlot{
      &states[slot_index],
      window_base + ring_data_offset + slot_index * slot_bytes,
  };
}

__device__ __forceinline__ unsigned long long make_ticket(
    unsigned long long sequence, std::uint32_t ordinal) {
  return (sequence << 32) |
      (static_cast<unsigned long long>(ordinal) + 1ULL);
}

__device__ __forceinline__ void mark_ready(
    RingSlotState* state, unsigned long long ticket) {
  __threadfence_system();
  atomicExch_system(&state->ready_ticket, ticket);
}

__device__ __forceinline__ void mark_consumed(
    RingSlotState* state, unsigned long long ticket) {
  __threadfence_system();
  atomicExch_system(&state->consumed_ticket, ticket);
}

}  // namespace device_transfer
}  // namespace awex

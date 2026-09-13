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

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace awex {
namespace device_transfer {

constexpr int kThreadsPerBlock = 256;
constexpr std::uint64_t kVectorBytes = sizeof(uint4);
constexpr std::size_t kDefaultRingSlotBytes = 512 * 1024;
constexpr std::uint32_t kDefaultRingSlotsPerPeer = 64;
// Static peer channels avoid producer contention while this global budget
// keeps each rank's registered payload bounded as world_size grows.
constexpr std::uint32_t kDefaultRingSlotBudget = 256;
constexpr std::size_t kRingAlignment = 256;
constexpr std::size_t kWindowAlignment = 4096;

static_assert(kVectorBytes == 16, "device transfer requires 16-byte uint4");

struct alignas(64) ControlBlock {
  unsigned long long epoch;
  unsigned int error;
};

struct DeviceTask {
  // nbytes is the logical payload size. tensor_offset addresses that payload
  // through fixed-width rows, allowing direct copies into pitched tensor views.
  std::uintptr_t tensor_ptr;
  std::uintptr_t remote_base;
  std::uint64_t nbytes;
  std::uint64_t tensor_offset;
  std::uint64_t tensor_row_bytes;
  std::uint64_t tensor_row_stride;
  std::uint32_t peer;
  // Dense per-peer segment index.  It selects a ring lane and forms the ticket.
  std::uint32_t ordinal;
  // Multicast tasks carry the LSA bases of every intended consumer in a
  // flattened launch array. Unicast tasks have one target.
  std::uint32_t target_begin;
  std::uint32_t target_count;
  std::uint32_t flags;
};

constexpr std::uint32_t kTaskMulticast = 1U;

struct alignas(16) RingSlotState {
  // The slot is reusable exactly when both tickets are equal.
  unsigned long long ready_ticket;
  unsigned long long consumed_ticket;
};

struct DeviceProfile {
  unsigned long long kernel_start_ns;
  unsigned long long first_ready_ns;
  unsigned long long last_ready_ns;
  unsigned long long first_copy_ns;
  unsigned long long last_copy_ns;
  unsigned long long publish_start_ns;
  unsigned long long publish_done_ns;
  unsigned long long peer_done_ns;
};

enum class TransferRole : std::uint32_t {
  kSender,
  kReceiver,
};

struct DeviceTransferArgs {
  const DeviceTask* tasks;
  std::uint32_t task_count;
  TransferRole role;
  unsigned long long sequence;
  int local_rank;
  std::uint8_t* local_base;
  const std::uint32_t* expected_counts;
  const std::uint32_t* peer_offsets;
  const std::uint32_t* active_peers;
  const std::uintptr_t* target_bases;
  std::uint32_t active_peer_count;
  std::uint32_t slots_per_peer;
  std::size_t slot_bytes;
  std::size_t ring_data_offset;
  DeviceProfile* profile;
  unsigned long long timeout_cycles;
};

}  // namespace device_transfer
}  // namespace awex

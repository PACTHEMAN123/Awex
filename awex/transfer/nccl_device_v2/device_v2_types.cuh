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
namespace nccl_device_v2 {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxWorksPerBatch = 8;
constexpr int kMaxChannelsPerPeer = 8;
constexpr std::uint32_t kDefaultChannelsPerPeer = 2;
constexpr std::uint32_t kDefaultFifoDepth = 8;
constexpr std::size_t kDefaultChunkBytes = 4 * 1024 * 1024;
constexpr std::size_t kDefaultStepBytes = 256 * 1024;
constexpr std::size_t kWindowAlignment = 4096;
constexpr std::size_t kFifoAlignment = 256;

static_assert(kThreadsPerBlock % kWarpSize == 0, "block must contain full warps");
static_assert(kMaxWorksPerBatch <= kThreadsPerBlock / kWarpSize, "draft requires one warp per work side");

enum class V2Direction : std::uint32_t {
  kSend,
  kRecv,
};

// This is the fixed-plan task shape consumed by the v2 lowering layer. The
// fields intentionally mirror the existing device task without adding a new
// logical operation or changing its ordering.
struct V2Task {
  std::uintptr_t tensor_ptr;
  std::uint64_t nbytes;
  std::uint64_t tensor_offset;
  std::uint64_t tensor_row_bytes;
  std::uint64_t tensor_row_stride;
  std::uint32_t peer;
  std::uint32_t ordinal;
};

// A side is optional so the device work format can represent a future
// send/recv pair in the same batch. The first draft normally enables only one
// side because the existing Awex calls have a sender/receiver role.
struct V2WorkSide {
  std::uint32_t enabled;
  std::uint32_t channel_base;
  std::uint32_t channel_count;
  // Bit p marks the final work for this peer/channel partition. Only those
  // partitions drain the final consumed step before the kernel returns.
  std::uint32_t final_parts;
  std::uintptr_t tensor_ptr;
  std::uint64_t nbytes;
  std::uint64_t tensor_offset;
  std::uint64_t tensor_row_bytes;
  std::uint64_t tensor_row_stride;
  // One absolute step base for each channel partition. Keeping this in the
  // work record makes channel-local connector state explicit in the draft.
  std::uint64_t step_begin[kMaxChannelsPerPeer];
};

struct alignas(16) V2Work {
  std::uint32_t peer;
  std::uint32_t task_ordinal;
  std::uint32_t chunk_ordinal;
  std::uint32_t chunk_count;
  V2WorkSide send;
  V2WorkSide recv;
};

struct V2WorkBatch {
  std::uint32_t work_begin;
  std::uint32_t work_count;
};

struct V2ChannelQueue {
  std::uint32_t first_batch;
  std::uint32_t batch_count;
};

struct alignas(16) V2FifoSlot {
  // A slot is reusable when consumed_step has reached step - fifo_depth.
  unsigned long long ready_step;
  unsigned long long consumed_step;
  std::uint32_t bytes;
  std::uint32_t reserved;
};

struct alignas(64) V2WindowHeader {
  unsigned long long epoch;
  unsigned int error;
  unsigned int reserved;
};

struct V2WindowLayout {
  std::size_t state_offset;
  std::size_t payload_offset;
  std::size_t slot_bytes;
  std::uint32_t fifo_depth;
  std::uint32_t channel_count;
  std::uint32_t world_size;
  std::size_t window_bytes;
};

struct V2KernelArgs {
  const V2Work* works;
  const V2WorkBatch* batches;
  const V2ChannelQueue* channels;
  std::uint32_t channel_count;
  std::uint32_t local_rank;
  std::uint32_t world_size;
  V2WindowLayout layout;
  std::uint8_t* local_window;
  const std::uintptr_t* peer_windows;
  unsigned long long epoch;
  unsigned long long timeout_cycles;
};

}  // namespace nccl_device_v2
}  // namespace awex

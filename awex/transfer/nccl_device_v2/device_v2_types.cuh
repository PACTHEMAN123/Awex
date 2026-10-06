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
#include <nccl.h>

#if __has_include(<nccl_device.h>)
#include <nccl_device.h>
#if defined(NCCL_OS_LINUX) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 4)
#define AWEX_NCCL_DEVICE_V2_HAS_GIN 1
#else
#define AWEX_NCCL_DEVICE_V2_HAS_GIN 0
#endif
#else
#include <nccl_device/core.h>
#define AWEX_NCCL_DEVICE_V2_HAS_GIN 0
#endif

#include <cstddef>
#include <cstdint>

namespace awex {
namespace nccl_device_v2 {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 640;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kMaxWorksPerBatch = 8;
constexpr int kMaxChannels = 64;
constexpr int kCopyPackBytes = 16;
constexpr int kCopyUnroll = 8;
constexpr std::uint32_t kDefaultFifoDepth = 8;
constexpr std::size_t kDefaultChunkBytes = 4 * 1024 * 1024;
constexpr std::size_t kDefaultStepBytes = 512 * 1024;
constexpr std::size_t kDefaultNetworkStepBytes = 128 * 1024;
constexpr std::size_t kWindowAlignment = 4096;
constexpr std::size_t kFifoAlignment = 256;

static_assert(kThreadsPerBlock % kWarpSize == 0, "block must contain full warps");
static_assert(kMaxWorksPerBatch <= 15, "work groups use CUDA named barriers 1-15");
static_assert(kMaxChannels <= 64, "channel masks use 64-bit values");

enum class V2Direction : std::uint32_t {
  kSend,
  kRecv,
};

enum class V2Transport : std::uint8_t {
  kLsa,
  kGin,
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

// One physical tensor span in a peer's virtual byte stream. Lowering may split
// a channel chunk across multiple fragments without exposing tensor boundaries
// to the FIFO protocol.
struct alignas(16) V2Fragment {
  std::uintptr_t tensor_ptr;
  std::uint64_t nbytes;
  std::uint64_t tensor_offset;
  std::uint64_t tensor_row_bytes;
  std::uint64_t tensor_row_stride;
  std::uint64_t work_offset;
};

struct alignas(16) V2Work {
  std::uint32_t peer;
  std::uint32_t fragment_begin;
  std::uint32_t fragment_count;
  std::uint32_t chunk_ordinal;
  std::uint32_t chunk_count;
  std::uint32_t final;
  std::uint32_t step_bytes;
  std::uint32_t fifo_depth;
  std::uint64_t stream_offset;
  std::uint64_t nbytes;
  std::uint64_t step_begin;
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
  std::uint32_t payload_peer_count;
  std::size_t window_bytes;
};

struct V2KernelArgs {
  const V2Work* works;
  const V2Fragment* fragments;
  const V2WorkBatch* batches;
  const V2ChannelQueue* channels;
  const std::uint32_t* channel_ids;
  const std::uint32_t* active_peers;
  const std::uint8_t* peer_transports;
  std::uint32_t channel_count;
  std::uint32_t active_peer_count;
  std::uint32_t local_rank;
  std::uint32_t world_size;
  V2Direction direction;
  V2WindowLayout layout;
  std::uint8_t* local_window;
  const std::uintptr_t* peer_windows;
  const std::uint32_t* payload_peer_slots;
  std::uint32_t gin_enabled;
  std::uint32_t gin_credit_batch;
  std::uint32_t gin_signal_count;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  ncclWindow_t window;
  ncclDevComm dev_comm;
#endif
  unsigned long long epoch;
  unsigned long long timeout_cycles;
};

}  // namespace nccl_device_v2
}  // namespace awex

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

#include <cuda_runtime.h>
#include <nccl.h>
#if __has_include(<nccl_device.h>)
#include <nccl_device.h>
#else
#include <nccl_device/core.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "device_v2_launch.cuh"
#include "device_v2_lowering.h"
#include "device_v2_topology.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace v2 = awex::nccl_device_v2;

namespace {

constexpr int kMaxRanks = 256;

struct LaunchBuffers {
  v2::V2Work* works = nullptr;
  v2::V2Fragment* fragments = nullptr;
  v2::V2WorkBatch* batches = nullptr;
  v2::V2ChannelQueue* channels = nullptr;
  std::uint32_t* channel_ids = nullptr;
  std::uint32_t* active_peers = nullptr;
  v2::V2QuantMatrix* quant_matrices = nullptr;
  v2::V2QuantBlock* quant_blocks = nullptr;
  CUtensorMap* tma_tensor_maps = nullptr;
  v2::V2TmaQuantTile* tma_tiles = nullptr;
  v2::V2TmaChannelQueue* tma_queues = nullptr;
};

struct TmaSchedule {
  std::vector<CUtensorMap> tensor_maps;
  std::vector<v2::V2TmaQuantTile> tiles;
  std::vector<v2::V2TmaChannelQueue> queues;
  std::vector<std::uint64_t> next_steps;
  std::uint32_t matrix_count = 0;
};

struct DeviceState {
  ncclComm_t comm = nullptr;
  ncclWindow_t window = nullptr;
  void* local_base = nullptr;
  std::vector<void*> remote_bases;
  uintptr_t* device_peer_windows = nullptr;
  std::uint32_t* device_payload_peer_slots = nullptr;
  v2::V2WindowLayout layout{};
  std::size_t window_bytes = 0;
  std::size_t dense_window_bytes = 0;
  std::uint64_t timeout_cycles = 0;
  std::uint64_t last_sequence = 0;
  std::uint32_t total_channels = 1;
  v2::V2Topology topology;
  std::uint32_t fifo_depth = v2::kDefaultFifoDepth;
  std::size_t chunk_bytes = v2::kDefaultChunkBytes;
  std::size_t step_bytes = v2::kDefaultStepBytes;
  bool tma_supported = false;
  bool plan_initialized = false;
  v2::V2Direction direction = v2::V2Direction::kSend;
  std::vector<v2::V2LoweringTask> tasks;
  std::vector<std::uint32_t> active_peers;
  std::vector<v2::V2QuantMatrix> quant_matrices;
  std::vector<v2::V2QuantBlock> quant_blocks;
  TmaSchedule tma_schedule;
  v2::V2Schedule schedule;
  LaunchBuffers buffers;
  int rank = 0;
  int world_size = 0;
  int device = 0;
};

void release_buffers(LaunchBuffers* buffers);

[[noreturn]] void throw_nccl(ncclResult_t result, const char* expression) {
  throw std::runtime_error(std::string(expression) + " failed: " + ncclGetErrorString(result));
}

void check_nccl(ncclResult_t result, const char* expression) {
  if (result != ncclSuccess) {
    throw_nccl(result, expression);
  }
}

void check_cuda(cudaError_t result, const char* expression) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed: " + cudaGetErrorString(result));
  }
}

#define AWEX_NCCL_V2_CHECK(expr) check_nccl((expr), #expr)
#define AWEX_CUDA_V2_CHECK(expr) check_cuda((expr), #expr)

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::runtime_error(std::string(description) + " size overflows");
  }
  return left * right;
}

std::uint32_t power_of_two_down(std::uint32_t value) {
  std::uint32_t result = 1;
  while (result <= value / 2) result *= 2;
  return result;
}

std::unique_ptr<DeviceState> make_state(const std::string& unique_id_bytes, int world_size, int rank, int device,
                                        int timeout_ms, std::uint32_t max_channels, std::uint32_t fifo_depth,
                                        std::size_t step_bytes, std::size_t chunk_bytes) {
  if (world_size < 2 || world_size > kMaxRanks) {
    throw std::runtime_error("nccl_device_v2 world_size must be in [2, 256]");
  }
  if (rank < 0 || rank >= world_size) {
    throw std::runtime_error("nccl_device_v2 rank is outside world_size");
  }
  if (unique_id_bytes.size() != sizeof(ncclUniqueId)) {
    throw std::runtime_error("Invalid NCCL unique id size");
  }
  if (max_channels == 0 || max_channels > v2::kMaxChannels) {
    throw std::runtime_error("invalid nccl_device_v2 max_channels");
  }
  if (fifo_depth == 0 || step_bytes == 0) {
    throw std::runtime_error("nccl_device_v2 FIFO depth and step size must be positive");
  }
  if (chunk_bytes != 0 && chunk_bytes < step_bytes) {
    throw std::runtime_error("nccl_device_v2 chunk_bytes must be zero or at least step_bytes");
  }
  auto state = std::make_unique<DeviceState>();
  state->rank = rank;
  state->world_size = world_size;
  state->device = device;
  state->fifo_depth = fifo_depth;
  state->step_bytes = step_bytes;
  state->chunk_bytes = chunk_bytes;
  AWEX_CUDA_V2_CHECK(cudaSetDevice(device));
  int compute_capability_major = 0;
  AWEX_CUDA_V2_CHECK(
    cudaDeviceGetAttribute(&compute_capability_major, cudaDevAttrComputeCapabilityMajor, device));
  state->tma_supported = compute_capability_major >= 9;
  int multiprocessor_count = 0;
  AWEX_CUDA_V2_CHECK(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  const std::uint32_t channel_limit = power_of_two_down(std::min<std::uint32_t>(max_channels, multiprocessor_count));

  int clock_rate_khz = 0;
  AWEX_CUDA_V2_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device));
  state->timeout_cycles =
    static_cast<std::uint64_t>(std::max(timeout_ms, 1)) * static_cast<std::uint64_t>(clock_rate_khz);

  ncclUniqueId unique_id;
  std::memcpy(&unique_id, unique_id_bytes.data(), sizeof(unique_id));
  try {
    AWEX_NCCL_V2_CHECK(ncclCommInitRank(&state->comm, world_size, unique_id, rank));

    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    AWEX_NCCL_V2_CHECK(ncclCommQueryProperties(state->comm, &properties));
    if (!properties.deviceApiSupport) {
      throw std::runtime_error("The loaded NCCL communicator does not support the Device API");
    }
    const ncclTeam_t lsa_team = ncclTeamLsa(state->comm);
    if (lsa_team.nRanks != world_size || lsa_team.rank != rank || lsa_team.stride != 1) {
      throw std::runtime_error("nccl_device_v2 requires a contiguous LSA domain");
    }

    state->topology = v2::discoverV2Topology(state->comm, world_size, rank, device, channel_limit);
    state->total_channels = state->topology.total_channels;
  } catch (...) {
    if (state->comm != nullptr) {
      ncclCommAbort(state->comm);
    }
    throw;
  }
  return state;
}

void destroy_state(DeviceState* state) {
  if (state == nullptr) {
    return;
  }
  AWEX_CUDA_V2_CHECK(cudaSetDevice(state->device));
  release_buffers(&state->buffers);
  if (state->device_payload_peer_slots != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(state->device_payload_peer_slots));
    state->device_payload_peer_slots = nullptr;
  }
  if (state->device_peer_windows != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(state->device_peer_windows));
    state->device_peer_windows = nullptr;
  }
  if (state->window != nullptr) {
    AWEX_NCCL_V2_CHECK(ncclCommWindowDeregister(state->comm, state->window));
    state->window = nullptr;
  }
  if (state->local_base != nullptr) {
    AWEX_NCCL_V2_CHECK(ncclMemFree(state->local_base));
    state->local_base = nullptr;
  }
  if (state->comm != nullptr) {
    AWEX_NCCL_V2_CHECK(ncclCommAbort(state->comm));
    state->comm = nullptr;
  }
}

void release_buffers(LaunchBuffers* buffers) {
  if (buffers->works != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->works));
    buffers->works = nullptr;
  }
  if (buffers->fragments != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->fragments));
    buffers->fragments = nullptr;
  }
  if (buffers->batches != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->batches));
    buffers->batches = nullptr;
  }
  if (buffers->channels != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->channels));
    buffers->channels = nullptr;
  }
  if (buffers->channel_ids != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->channel_ids));
    buffers->channel_ids = nullptr;
  }
  if (buffers->active_peers != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->active_peers));
    buffers->active_peers = nullptr;
  }
  if (buffers->quant_matrices != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->quant_matrices));
    buffers->quant_matrices = nullptr;
  }
  if (buffers->quant_blocks != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->quant_blocks));
    buffers->quant_blocks = nullptr;
  }
  if (buffers->tma_tensor_maps != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->tma_tensor_maps));
    buffers->tma_tensor_maps = nullptr;
  }
  if (buffers->tma_tiles != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->tma_tiles));
    buffers->tma_tiles = nullptr;
  }
  if (buffers->tma_queues != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->tma_queues));
    buffers->tma_queues = nullptr;
  }
}

void allocate_buffers(const v2::V2Schedule& schedule, std::size_t active_peer_count,
                      std::size_t quant_matrix_count, std::size_t quant_block_count,
                      const TmaSchedule& tma_schedule, LaunchBuffers* buffers) {
  try {
    if (!schedule.works.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->works),
                                    schedule.works.size() * sizeof(v2::V2Work)));
    }
    if (!schedule.fragments.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->fragments),
                                    schedule.fragments.size() * sizeof(v2::V2Fragment)));
    }
    if (!schedule.batches.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->batches),
                                    schedule.batches.size() * sizeof(v2::V2WorkBatch)));
    }
    if (!schedule.channels.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->channels),
                                    schedule.channels.size() * sizeof(v2::V2ChannelQueue)));
    }
    if (!schedule.channel_ids.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->channel_ids),
                                    schedule.channel_ids.size() * sizeof(std::uint32_t)));
    }
    if (active_peer_count != 0) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->active_peers),
                                    active_peer_count * sizeof(std::uint32_t)));
    }
    if (quant_matrix_count != 0) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->quant_matrices),
                                    quant_matrix_count * sizeof(v2::V2QuantMatrix)));
    }
    if (quant_block_count != 0) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->quant_blocks),
                                    quant_block_count * sizeof(v2::V2QuantBlock)));
    }
    if (!tma_schedule.tensor_maps.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->tma_tensor_maps),
                                    tma_schedule.tensor_maps.size() * sizeof(CUtensorMap)));
    }
    if (!tma_schedule.tiles.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->tma_tiles),
                                    tma_schedule.tiles.size() * sizeof(v2::V2TmaQuantTile)));
    }
    if (!tma_schedule.queues.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers->tma_queues),
                                    tma_schedule.queues.size() * sizeof(v2::V2TmaChannelQueue)));
    }
  } catch (...) {
    release_buffers(buffers);
    throw;
  }
}

void upload_buffers(const v2::V2Schedule& schedule, const std::vector<std::uint32_t>& active_peers,
                    const std::vector<v2::V2QuantMatrix>& quant_matrices,
                    const std::vector<v2::V2QuantBlock>& quant_blocks,
                    const TmaSchedule& tma_schedule, LaunchBuffers* buffers, cudaStream_t stream) {
  if (!schedule.works.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->works, schedule.works.data(),
                                       schedule.works.size() * sizeof(v2::V2Work), cudaMemcpyHostToDevice, stream));
  }
  if (!schedule.fragments.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->fragments, schedule.fragments.data(),
                                       schedule.fragments.size() * sizeof(v2::V2Fragment), cudaMemcpyHostToDevice,
                                       stream));
  }
  if (!schedule.batches.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->batches, schedule.batches.data(),
                                       schedule.batches.size() * sizeof(v2::V2WorkBatch), cudaMemcpyHostToDevice,
                                       stream));
  }
  if (!schedule.channels.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->channels, schedule.channels.data(),
                                       schedule.channels.size() * sizeof(v2::V2ChannelQueue), cudaMemcpyHostToDevice,
                                       stream));
  }
  if (!schedule.channel_ids.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->channel_ids, schedule.channel_ids.data(),
                                       schedule.channel_ids.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                       stream));
  }
  if (!active_peers.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->active_peers, active_peers.data(),
                                       active_peers.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
  }
  if (!quant_matrices.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->quant_matrices, quant_matrices.data(),
                                       quant_matrices.size() * sizeof(v2::V2QuantMatrix), cudaMemcpyHostToDevice,
                                       stream));
  }
  if (!quant_blocks.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->quant_blocks, quant_blocks.data(),
                                       quant_blocks.size() * sizeof(v2::V2QuantBlock), cudaMemcpyHostToDevice,
                                       stream));
  }
  if (!tma_schedule.tensor_maps.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->tma_tensor_maps, tma_schedule.tensor_maps.data(),
                                       tma_schedule.tensor_maps.size() * sizeof(CUtensorMap),
                                       cudaMemcpyHostToDevice, stream));
  }
  if (!tma_schedule.tiles.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->tma_tiles, tma_schedule.tiles.data(),
                                       tma_schedule.tiles.size() * sizeof(v2::V2TmaQuantTile),
                                       cudaMemcpyHostToDevice, stream));
  }
  if (!tma_schedule.queues.empty()) {
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(buffers->tma_queues, tma_schedule.queues.data(),
                                       tma_schedule.queues.size() * sizeof(v2::V2TmaChannelQueue),
                                       cudaMemcpyHostToDevice, stream));
  }
}

bool same_tasks(const std::vector<v2::V2LoweringTask>& left, const std::vector<v2::V2LoweringTask>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto& a = left[index];
    const auto& b = right[index];
    if (a.tensor_ptr != b.tensor_ptr || a.nbytes != b.nbytes || a.tensor_offset != b.tensor_offset ||
        a.tensor_row_bytes != b.tensor_row_bytes || a.tensor_row_stride != b.tensor_row_stride ||
        a.tensor_dtype != b.tensor_dtype || a.wire_dtype != b.wire_dtype ||
        a.tensor_element_bytes != b.tensor_element_bytes || a.wire_element_bytes != b.wire_element_bytes ||
        a.quant_scale_ptr != b.quant_scale_ptr || a.quant_rows != b.quant_rows || a.quant_cols != b.quant_cols ||
        a.quant_row_offset != b.quant_row_offset || a.quant_col_offset != b.quant_col_offset ||
        a.quant_scale_row_stride != b.quant_scale_row_stride || a.quant_mode != b.quant_mode ||
        a.quant_block_rows != b.quant_block_rows || a.quant_block_cols != b.quant_block_cols ||
        a.quant_group != b.quant_group || a.peer != b.peer || a.ordinal != b.ordinal) {
      return false;
    }
  }
  return true;
}

v2::V2DataType tensor_dtype(const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Half:
      return v2::V2DataType::kFloat16;
    case at::ScalarType::BFloat16:
      return v2::V2DataType::kBFloat16;
    case at::ScalarType::Float:
      return v2::V2DataType::kFloat32;
    case at::ScalarType::Float8_e4m3fn:
      return v2::V2DataType::kFloat8E4M3;
    case at::ScalarType::Float8_e5m2:
      return v2::V2DataType::kFloat8E5M2;
    default:
      return v2::V2DataType::kOpaque;
  }
}

v2::V2DataType parse_wire_dtype(int64_t value) {
  if (value < static_cast<int64_t>(v2::V2DataType::kOpaque) ||
      value > static_cast<int64_t>(v2::V2DataType::kFloat8E5M2)) {
    throw std::runtime_error("nccl_device_v2 wire dtype code is invalid");
  }
  return static_cast<v2::V2DataType>(value);
}

std::uint32_t dtype_bytes(v2::V2DataType dtype) {
  switch (dtype) {
    case v2::V2DataType::kFloat16:
    case v2::V2DataType::kBFloat16:
      return 2;
    case v2::V2DataType::kFloat32:
      return 4;
    case v2::V2DataType::kFloat8E4M3:
    case v2::V2DataType::kFloat8E5M2:
      return 1;
    case v2::V2DataType::kOpaque:
      return 0;
  }
  return 0;
}

struct QuantPlan {
  std::vector<v2::V2QuantMatrix> matrices;
  std::vector<v2::V2QuantBlock> blocks;
};

QuantPlan build_quant_plan(const std::vector<v2::V2LoweringTask>& tasks) {
  using Key = std::tuple<std::uintptr_t, std::uintptr_t, std::uint64_t, std::uint64_t, std::uint64_t,
                         std::uint64_t, std::uint64_t, std::uint64_t, std::uint32_t, std::uint32_t,
                         std::uint32_t, std::uint32_t>;
  std::set<Key> seen;
  QuantPlan plan;
  for (const auto& task : tasks) {
    if (task.quant_mode == v2::V2QuantMode::kNone) continue;
    const Key key{
      task.tensor_ptr,
      task.quant_scale_ptr,
      task.tensor_row_stride,
      task.quant_rows,
      task.quant_cols,
      task.quant_row_offset,
      task.quant_col_offset,
      task.quant_scale_row_stride,
      static_cast<std::uint32_t>(task.tensor_dtype),
      task.tensor_element_bytes,
      task.quant_block_rows,
      task.quant_block_cols,
    };
    if (!seen.insert(key).second) continue;
    const std::size_t matrix_index = plan.matrices.size();
    plan.matrices.push_back(v2::V2QuantMatrix{
      task.tensor_ptr,
      task.quant_scale_ptr,
      task.tensor_row_stride,
      task.quant_rows,
      task.quant_cols,
      task.quant_row_offset,
      task.quant_col_offset,
      task.quant_scale_row_stride,
      task.tensor_dtype,
      task.tensor_element_bytes,
      task.quant_block_rows,
      task.quant_block_cols,
    });
    const std::uint64_t row_blocks =
      (task.quant_rows + task.quant_block_rows - 1) / task.quant_block_rows;
    const std::uint64_t col_blocks =
      (task.quant_cols + task.quant_block_cols - 1) / task.quant_block_cols;
    const std::uint64_t blocks = row_blocks * col_blocks;
    if (blocks > std::numeric_limits<std::uint32_t>::max() ||
        matrix_index > std::numeric_limits<std::uint32_t>::max() ||
        plan.blocks.size() > std::numeric_limits<std::uint32_t>::max() - blocks) {
      throw std::runtime_error("nccl_device_v2 block-wise FP8 matrix is too large");
    }
    for (std::uint32_t block = 0; block < blocks; ++block) {
      plan.blocks.push_back(v2::V2QuantBlock{
        static_cast<std::uint32_t>(matrix_index),
        block,
      });
    }
  }
  return plan;
}

std::uintptr_t task_base_address(const v2::V2LoweringTask& task) {
  const std::uint64_t row = task.tensor_offset / task.tensor_row_bytes;
  const std::uint64_t col = task.tensor_offset - row * task.tensor_row_bytes;
  return task.tensor_ptr + row * task.tensor_row_stride + col;
}

bool tma_scale_task(const v2::V2LoweringTask& scale, std::uint32_t peer) {
  return scale.peer == peer && scale.tensor_dtype == v2::V2DataType::kFloat32 &&
         scale.wire_dtype == v2::V2DataType::kFloat32 && scale.tensor_element_bytes == sizeof(float) &&
         scale.wire_element_bytes == sizeof(float);
}

bool tma_send_weight(const v2::V2LoweringTask& weight) {
  if (weight.quant_mode != v2::V2QuantMode::kBlockwiseFloat8E4M3 ||
      weight.tensor_dtype != v2::V2DataType::kBFloat16 ||
      weight.wire_dtype != v2::V2DataType::kFloat8E4M3 || weight.tensor_element_bytes != 2 ||
      weight.wire_element_bytes != 1 || weight.quant_block_rows != v2::kTmaQuantBlockRows ||
      weight.quant_block_cols != v2::kTmaQuantBlockCols) {
    return false;
  }
  if (weight.quant_rows == 0 || weight.quant_cols == 0 ||
      weight.quant_rows % v2::kTmaQuantBlockRows != 0 ||
      weight.quant_cols % v2::kTmaQuantBlockCols != 0 ||
      weight.nbytes != weight.quant_rows * weight.quant_cols ||
      task_base_address(weight) % 16 != 0 || weight.tensor_row_stride % 16 != 0) {
    return false;
  }
  return true;
}

bool tma_recv_weight(const v2::V2LoweringTask& weight, std::uint64_t* rows, std::uint64_t* cols) {
  if (weight.quant_mode != v2::V2QuantMode::kNone ||
      weight.tensor_dtype != v2::V2DataType::kFloat8E4M3 ||
      weight.wire_dtype != v2::V2DataType::kFloat8E4M3 || weight.tensor_element_bytes != 1 ||
      weight.wire_element_bytes != 1 || weight.tensor_row_bytes == 0 ||
      weight.tensor_offset % weight.tensor_row_bytes != 0 || weight.nbytes % weight.tensor_row_bytes != 0) {
    return false;
  }
  *rows = weight.nbytes / weight.tensor_row_bytes;
  *cols = weight.tensor_row_bytes;
  if (*rows == 0 || *cols == 0 || *rows % v2::kTmaQuantBlockRows != 0 ||
      *cols % v2::kTmaQuantBlockCols != 0 || weight.nbytes != *rows * *cols ||
      task_base_address(weight) % 16 != 0 || weight.tensor_row_stride % 16 != 0) {
    return false;
  }
  return true;
}

CUtensorMap make_tma_tensor_map(const v2::V2LoweringTask& task, std::uint64_t rows, std::uint64_t cols) {
  CUtensorMap tensor_map{};
  const cuuint64_t global_dims[2] = {cols, rows};
  const cuuint64_t global_strides[1] = {task.tensor_row_stride};
  const cuuint32_t box_dims[2] = {v2::kTmaQuantBlockCols, v2::kTmaQuantBlockRows};
  const cuuint32_t element_strides[2] = {1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
    &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, reinterpret_cast<void*>(task_base_address(task)),
    global_dims, global_strides, box_dims, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") +
                             (message == nullptr ? "unknown CUDA driver error" : message));
  }
  return tensor_map;
}

TmaSchedule build_tma_schedule(const std::vector<v2::V2LoweringTask>& tasks, v2::V2Direction direction,
                               const v2::V2LoweringConfig& config, std::vector<bool>* handled) {
  TmaSchedule schedule;
  schedule.next_steps = config.initial_steps.empty()
                          ? std::vector<std::uint64_t>(static_cast<std::size_t>(config.world_size) *
                                                        config.total_channels, 1)
                          : config.initial_steps;
  handled->assign(tasks.size(), false);
  std::vector<std::vector<v2::V2TmaQuantTile>> peer_tiles(config.world_size);
  constexpr std::uint32_t kNoQuantGroup = std::numeric_limits<std::uint32_t>::max();
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> grouped_tasks;
  std::vector<std::uint64_t> group_order;
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    const auto& task = tasks[index];
    if (task.quant_group == kNoQuantGroup) continue;
    const std::uint64_t key = (static_cast<std::uint64_t>(task.peer) << 32) | task.quant_group;
    auto [entry, inserted] = grouped_tasks.emplace(key, std::vector<std::size_t>{});
    if (inserted) group_order.push_back(key);
    entry->second.push_back(index);
  }

  for (const std::uint64_t key : group_order) {
    const auto& group = grouped_tasks.at(key);
    const std::uint32_t peer = static_cast<std::uint32_t>(key >> 32);
    if (direction == v2::V2Direction::kSend) {
      std::vector<std::size_t> weight_indices;
      std::size_t scale_index = tasks.size();
      bool eligible = true;
      for (const std::size_t index : group) {
        if (tma_send_weight(tasks[index])) {
          weight_indices.push_back(index);
        } else if (tma_scale_task(tasks[index], peer) && scale_index == tasks.size()) {
          scale_index = index;
        } else {
          eligible = false;
          break;
        }
      }
      if (!eligible || weight_indices.empty() || scale_index == tasks.size()) continue;
      std::uint64_t tile_count = 0;
      for (const std::size_t weight_index : weight_indices) {
        const auto& weight = tasks[weight_index];
        tile_count += weight.quant_rows / v2::kTmaQuantBlockRows *
                      (weight.quant_cols / v2::kTmaQuantBlockCols);
      }
      if (tasks[scale_index].nbytes != tile_count * sizeof(float)) continue;
      const auto& first_weight = tasks[weight_indices.front()];
      const std::uintptr_t first_scale_ptr =
        first_weight.quant_scale_ptr +
        (first_weight.quant_row_offset / v2::kTmaQuantBlockRows * first_weight.quant_scale_row_stride +
         first_weight.quant_col_offset / v2::kTmaQuantBlockCols) * sizeof(float);
      if (task_base_address(tasks[scale_index]) != first_scale_ptr) continue;
      for (const std::size_t weight_index : weight_indices) {
        const auto& weight = tasks[weight_index];
        const std::uint64_t rows = weight.quant_rows;
        const std::uint64_t cols = weight.quant_cols;
        const std::uint32_t map_index = static_cast<std::uint32_t>(schedule.tensor_maps.size());
        schedule.tensor_maps.push_back(make_tma_tensor_map(weight, rows, cols));
        const std::uintptr_t scale_ptr =
          weight.quant_scale_ptr +
          (weight.quant_row_offset / v2::kTmaQuantBlockRows * weight.quant_scale_row_stride +
           weight.quant_col_offset / v2::kTmaQuantBlockCols) * sizeof(float);
        for (std::uint32_t tile_row = 0; tile_row < rows / v2::kTmaQuantBlockRows; ++tile_row) {
          for (std::uint32_t tile_col = 0; tile_col < cols / v2::kTmaQuantBlockCols; ++tile_col) {
            peer_tiles[peer].push_back(v2::V2TmaQuantTile{
              task_base_address(weight),
              scale_ptr,
              weight.tensor_row_stride,
              weight.quant_scale_row_stride * sizeof(float),
              0,
              map_index,
              tile_row,
              tile_col,
              0,
            });
          }
        }
        (*handled)[weight_index] = true;
      }
      (*handled)[scale_index] = true;
      ++schedule.matrix_count;
      continue;
    }

    std::vector<std::size_t> weight_indices;
    std::size_t scale_index = tasks.size();
    bool eligible = true;
    std::uint64_t tile_count = 0;
    std::uint64_t matrix_cols = 0;
    for (const std::size_t index : group) {
      std::uint64_t rows = 0;
      std::uint64_t cols = 0;
      if (tma_recv_weight(tasks[index], &rows, &cols)) {
        if (matrix_cols != 0 && matrix_cols != cols) {
          eligible = false;
          break;
        }
        matrix_cols = cols;
        tile_count += rows / v2::kTmaQuantBlockRows * (cols / v2::kTmaQuantBlockCols);
        weight_indices.push_back(index);
      } else if (tma_scale_task(tasks[index], peer) && scale_index == tasks.size()) {
        scale_index = index;
      } else {
        eligible = false;
        break;
      }
    }
    if (!eligible || weight_indices.empty() || scale_index == tasks.size()) continue;
    const auto& scale = tasks[scale_index];
    const std::uint64_t scale_cols = matrix_cols / v2::kTmaQuantBlockCols;
    if (scale.nbytes != tile_count * sizeof(float) || scale.tensor_row_bytes != scale_cols * sizeof(float) ||
        scale.tensor_row_stride % 16 != 0) {
      continue;
    }
    for (const std::size_t weight_index : weight_indices) {
      const auto& weight = tasks[weight_index];
      const std::uint64_t rows = weight.nbytes / weight.tensor_row_bytes;
      const std::uint64_t cols = weight.tensor_row_bytes;
      const std::uint64_t destination_row = weight.tensor_offset / weight.tensor_row_bytes;
      const std::uintptr_t scale_ptr =
        task_base_address(scale) + destination_row / v2::kTmaQuantBlockRows * scale.tensor_row_stride;
      for (std::uint32_t tile_row = 0; tile_row < rows / v2::kTmaQuantBlockRows; ++tile_row) {
        for (std::uint32_t tile_col = 0; tile_col < cols / v2::kTmaQuantBlockCols; ++tile_col) {
          peer_tiles[peer].push_back(v2::V2TmaQuantTile{
            task_base_address(weight),
            scale_ptr,
            weight.tensor_row_stride,
            scale.tensor_row_stride,
            0,
            0,
            tile_row,
            tile_col,
            0,
          });
        }
      }
      (*handled)[weight_index] = true;
    }
    (*handled)[scale_index] = true;
    ++schedule.matrix_count;
  }

  for (std::uint32_t peer = 0; peer < config.world_size; ++peer) {
    if (peer_tiles[peer].empty()) continue;
    const std::uint32_t max_channels =
      std::max<std::uint32_t>(1, std::min(config.peer_channels[peer], config.total_channels));
    std::uint32_t min_channels = max_channels;
    while (static_cast<std::uint64_t>(min_channels) * config.world_size > config.total_channels && min_channels > 1) {
      min_channels /= 2;
    }
    const std::uint64_t stream_bytes = peer_tiles[peer].size() * v2::kTmaQuantPacketBytes;
    const std::uint32_t channel_count =
      v2::v2ChannelsForBytes(stream_bytes, min_channels, max_channels, config.step_bytes);
    const std::uint32_t channel_base =
      v2::v2ChannelBase(config.local_rank, peer, config.world_size, config.total_channels);
    std::vector<std::vector<v2::V2TmaQuantTile>> channel_tiles(channel_count);
    for (std::size_t tile = 0; tile < peer_tiles[peer].size(); ++tile) {
      channel_tiles[tile % channel_count].push_back(peer_tiles[peer][tile]);
    }
    for (std::uint32_t part = 0; part < channel_count; ++part) {
      if (channel_tiles[part].empty()) continue;
      const std::uint32_t channel = (channel_base + part) & (config.total_channels - 1);
      v2::V2TmaChannelQueue queue{
        peer,
        channel,
        static_cast<std::uint32_t>(schedule.tiles.size()),
        static_cast<std::uint32_t>(channel_tiles[part].size()),
      };
      const std::size_t connection = static_cast<std::size_t>(peer) * config.total_channels + channel;
      for (auto tile : channel_tiles[part]) {
        tile.step = schedule.next_steps[connection]++;
        schedule.tiles.push_back(tile);
      }
      schedule.queues.push_back(queue);
    }
  }
  return schedule;
}

std::vector<v2::V2LoweringTask> remove_tma_tasks(const std::vector<v2::V2LoweringTask>& tasks,
                                                  const std::vector<bool>& handled,
                                                  std::uint32_t world_size) {
  std::vector<std::uint32_t> ordinals(world_size, 0);
  std::vector<v2::V2LoweringTask> generic;
  generic.reserve(tasks.size());
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    if (handled[index]) continue;
    auto task = tasks[index];
    task.ordinal = ordinals[task.peer]++;
    generic.push_back(task);
  }
  return generic;
}

void initialize_sparse_window(DeviceState* state, const std::vector<std::uint32_t>& active_peers,
                              v2::V2Direction direction, cudaStream_t stream) {
  const std::uint32_t inactive = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> local_payload_slots(state->world_size, inactive);
  if (direction == v2::V2Direction::kSend) {
    for (std::size_t index = 0; index < active_peers.size(); ++index) {
      local_payload_slots[active_peers[index]] = static_cast<std::uint32_t>(index);
    }
  }
  const auto payload_peer_slots = v2::topology_detail::allGather(
    state->comm, local_payload_slots.data(), local_payload_slots.size(), state->world_size, stream);
  for (const std::uint32_t peer : active_peers) {
    const std::size_t owner = direction == v2::V2Direction::kSend ? state->rank : peer;
    const std::size_t connection = direction == v2::V2Direction::kSend ? peer : state->rank;
    if (payload_peer_slots[owner * state->world_size + connection] == inactive) {
      throw std::runtime_error("nccl_device_v2 peer directions do not define a matching payload window");
    }
  }

  const std::uint32_t payload_peer_count =
    direction == v2::V2Direction::kSend ? static_cast<std::uint32_t>(active_peers.size()) : 0;
  state->layout = v2::makeV2WindowLayout(state->world_size, state->total_channels, state->fifo_depth,
                                         state->step_bytes, payload_peer_count);
  state->window_bytes = state->layout.window_bytes;
  state->dense_window_bytes = v2::makeV2WindowLayout(state->world_size, state->total_channels, state->fifo_depth,
                                                      state->step_bytes, state->world_size)
                                .window_bytes;
  try {
    AWEX_NCCL_V2_CHECK(ncclMemAlloc(&state->local_base, state->window_bytes));
    AWEX_NCCL_V2_CHECK(ncclCommWindowRegister(state->comm, state->local_base, state->window_bytes, &state->window,
                                              NCCL_WIN_COLL_SYMMETRIC));

    state->remote_bases.resize(state->world_size, nullptr);
    state->remote_bases[state->rank] = state->local_base;
    for (int peer = 0; peer < state->world_size; ++peer) {
      if (peer == state->rank) continue;
      AWEX_NCCL_V2_CHECK(ncclGetLsaDevicePointer(state->window, 0, peer, &state->remote_bases[peer]));
    }

    AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->device_peer_windows),
                                  checked_multiply(static_cast<std::size_t>(state->world_size), sizeof(uintptr_t),
                                                   "nccl_device_v2 peer window table")));
    std::vector<uintptr_t> peer_windows(state->world_size, 0);
    for (int peer = 0; peer < state->world_size; ++peer) {
      peer_windows[peer] = reinterpret_cast<uintptr_t>(state->remote_bases[peer]);
    }
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(state->device_peer_windows, peer_windows.data(),
                                       peer_windows.size() * sizeof(uintptr_t), cudaMemcpyHostToDevice, stream));

    AWEX_CUDA_V2_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&state->device_payload_peer_slots),
      checked_multiply(payload_peer_slots.size(), sizeof(std::uint32_t), "nccl_device_v2 payload peer table")));
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(state->device_payload_peer_slots, payload_peer_slots.data(),
                                       payload_peer_slots.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                       stream));
  } catch (...) {
    if (state->device_payload_peer_slots != nullptr) {
      cudaFree(state->device_payload_peer_slots);
      state->device_payload_peer_slots = nullptr;
    }
    if (state->device_peer_windows != nullptr) {
      cudaFree(state->device_peer_windows);
      state->device_peer_windows = nullptr;
    }
    if (state->window != nullptr) {
      ncclCommWindowDeregister(state->comm, state->window);
      state->window = nullptr;
    }
    if (state->local_base != nullptr) {
      ncclMemFree(state->local_base);
      state->local_base = nullptr;
    }
    throw;
  }
}

std::vector<v2::V2LoweringTask> build_tasks(
  const DeviceState& state, const py::list& tensors, const std::vector<int64_t>& lengths,
  const std::vector<int64_t>& tensor_offsets, const std::vector<int64_t>& tensor_row_bytes,
  const std::vector<int64_t>& tensor_row_strides, const std::vector<int64_t>& wire_dtypes,
  const std::vector<int64_t>& wire_element_bytes, const py::list& quant_scale_tensors,
  const std::vector<int64_t>& quant_modes, const std::vector<int64_t>& quant_rows,
  const std::vector<int64_t>& quant_cols, const std::vector<int64_t>& quant_row_offsets,
  const std::vector<int64_t>& quant_col_offsets, const std::vector<int64_t>& quant_scale_row_strides,
  const std::vector<int64_t>& quant_block_rows, const std::vector<int64_t>& quant_block_cols,
  const std::vector<int64_t>& quant_group_ids, const std::vector<int64_t>& peers, const std::vector<int64_t>& ordinals,
  const std::vector<int64_t>& expected_counts, bool sender, std::vector<std::uint32_t>* active_peers) {
  if (tensors.size() != lengths.size() || tensors.size() != tensor_offsets.size() ||
      tensors.size() != tensor_row_bytes.size() || tensors.size() != tensor_row_strides.size() ||
      tensors.size() != wire_dtypes.size() || tensors.size() != wire_element_bytes.size() ||
      tensors.size() != quant_scale_tensors.size() || tensors.size() != quant_modes.size() ||
      tensors.size() != quant_rows.size() || tensors.size() != quant_cols.size() ||
      tensors.size() != quant_row_offsets.size() || tensors.size() != quant_col_offsets.size() ||
      tensors.size() != quant_scale_row_strides.size() || tensors.size() != quant_block_rows.size() ||
      tensors.size() != quant_block_cols.size() || tensors.size() != quant_group_ids.size() ||
      tensors.size() != peers.size() || tensors.size() != ordinals.size()) {
    throw std::runtime_error("nccl_device_v2 tensor/task descriptor lengths do not match");
  }
  if (expected_counts.size() != static_cast<std::size_t>(state.world_size)) {
    throw std::runtime_error("nccl_device_v2 expected-count vector does not match world_size");
  }

  std::vector<std::uint32_t> expected(state.world_size, 0);
  for (int peer = 0; peer < state.world_size; ++peer) {
    if (expected_counts[peer] < 0 || static_cast<std::uint64_t>(expected_counts[peer]) > UINT32_MAX) {
      throw std::runtime_error("nccl_device_v2 expected count is invalid");
    }
    expected[peer] = static_cast<std::uint32_t>(expected_counts[peer]);
  }

  std::vector<std::uint8_t> peer_seen(state.world_size, 0);
  std::vector<std::uint32_t> actual(state.world_size, 0);
  std::vector<v2::V2LoweringTask> tasks;
  tasks.reserve(tensors.size());
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto tensor = tensors[index].cast<torch::Tensor>();
    if (!tensor.is_cuda() || tensor.get_device() != state.device) {
      throw std::runtime_error("nccl_device_v2 tensors must be on the transport CUDA device");
    }
    const auto local_dtype = tensor_dtype(tensor);
    const auto wire_dtype = parse_wire_dtype(wire_dtypes[index]);
    const auto local_element_bytes = static_cast<std::uint32_t>(tensor.element_size());
    if (wire_element_bytes[index] <= 0 || static_cast<std::uint64_t>(wire_element_bytes[index]) > UINT32_MAX) {
      throw std::runtime_error("nccl_device_v2 wire element size is invalid");
    }
    const auto wire_item_bytes = static_cast<std::uint32_t>(wire_element_bytes[index]);
    const std::uint32_t expected_wire_bytes = dtype_bytes(wire_dtype);
    if (expected_wire_bytes != 0 && expected_wire_bytes != wire_item_bytes) {
      throw std::runtime_error("nccl_device_v2 wire dtype and element size disagree");
    }
    const bool requires_cast = local_dtype != wire_dtype || local_element_bytes != wire_item_bytes;
    if (requires_cast && (local_dtype == v2::V2DataType::kOpaque || wire_dtype == v2::V2DataType::kOpaque)) {
      throw std::runtime_error("nccl_device_v2 streaming cast dtype is unsupported");
    }
    if (!sender && requires_cast) {
      throw std::runtime_error("nccl_device_v2 receiver tensor must use the wire dtype");
    }
    if (quant_modes[index] < static_cast<int64_t>(v2::V2QuantMode::kNone) ||
        quant_modes[index] > static_cast<int64_t>(v2::V2QuantMode::kBlockwiseFloat8E4M3)) {
      throw std::runtime_error("nccl_device_v2 quantization mode is invalid");
    }
    const auto quant_mode = static_cast<v2::V2QuantMode>(quant_modes[index]);
    std::uintptr_t quant_scale_ptr = 0;
    if (quant_mode != v2::V2QuantMode::kNone) {
      if (!sender || wire_dtype != v2::V2DataType::kFloat8E4M3 ||
          local_dtype == v2::V2DataType::kOpaque || quant_rows[index] <= 0 || quant_cols[index] <= 0 ||
          quant_row_offsets[index] < 0 || quant_col_offsets[index] < 0 ||
          quant_scale_row_strides[index] <= 0 || quant_block_rows[index] != 128 || quant_block_cols[index] != 128 ||
          tensor_offsets[index] != 0 || tensor_row_bytes[index] != quant_cols[index] * local_element_bytes ||
          lengths[index] != quant_rows[index] * quant_cols[index] * wire_item_bytes ||
          quant_row_offsets[index] % quant_block_rows[index] != 0 ||
          quant_col_offsets[index] % quant_block_cols[index] != 0) {
        throw std::runtime_error("nccl_device_v2 block-wise FP8 descriptor is invalid");
      }
      const auto scale = quant_scale_tensors[index].cast<torch::Tensor>();
      if (!scale.is_cuda() || scale.get_device() != state.device || scale.scalar_type() != at::ScalarType::Float ||
          !scale.is_contiguous()) {
        throw std::runtime_error("nccl_device_v2 block-wise FP8 scale tensor must be contiguous CUDA FP32");
      }
      const std::uint64_t scale_rows =
        (static_cast<std::uint64_t>(quant_row_offsets[index] + quant_rows[index]) + quant_block_rows[index] - 1) /
        quant_block_rows[index];
      const std::uint64_t scale_cols =
        (static_cast<std::uint64_t>(quant_col_offsets[index] + quant_cols[index]) + quant_block_cols[index] - 1) /
        quant_block_cols[index];
      const std::uint64_t required_scale_numel =
        (scale_rows - 1) * quant_scale_row_strides[index] + scale_cols;
      if (required_scale_numel > static_cast<std::uint64_t>(scale.numel())) {
        throw std::runtime_error("nccl_device_v2 block-wise FP8 scale tensor is too small");
      }
      quant_scale_ptr = reinterpret_cast<std::uintptr_t>(scale.data_ptr());
    }
    const int64_t peer = peers[index];
    const int64_t ordinal = ordinals[index];
    const int64_t quant_group_id = quant_group_ids[index];
    if (peer < 0 || peer >= state.world_size || peer == state.rank) {
      throw std::runtime_error("nccl_device_v2 task peer is invalid");
    }
    if (ordinal < 0 || static_cast<std::uint64_t>(ordinal) >= expected[peer]) {
      throw std::runtime_error("nccl_device_v2 task ordinal is invalid");
    }
    if (quant_group_id < -1 ||
        (quant_group_id >= 0 && static_cast<std::uint64_t>(quant_group_id) >= UINT32_MAX)) {
      throw std::runtime_error("nccl_device_v2 quantization group is invalid");
    }
    if (lengths[index] < 0 || tensor_offsets[index] < 0 || tensor_row_bytes[index] <= 0 ||
        tensor_row_strides[index] < tensor_row_bytes[index] ||
        lengths[index] % wire_item_bytes != 0 || tensor_offsets[index] % local_element_bytes != 0 ||
        tensor_row_bytes[index] % local_element_bytes != 0 ||
        tensor_row_strides[index] % local_element_bytes != 0) {
      throw std::runtime_error("nccl_device_v2 task byte range is invalid");
    }
    if (!peer_seen[peer]) {
      peer_seen[peer] = 1;
      active_peers->push_back(static_cast<std::uint32_t>(peer));
    }
    tasks.push_back(v2::V2LoweringTask{
      reinterpret_cast<uintptr_t>(tensor.data_ptr()),
      static_cast<std::uint64_t>(lengths[index]),
      static_cast<std::uint64_t>(tensor_offsets[index]),
      static_cast<std::uint64_t>(tensor_row_bytes[index]),
      static_cast<std::uint64_t>(tensor_row_strides[index]),
      local_dtype,
      wire_dtype,
      local_element_bytes,
      wire_item_bytes,
      quant_scale_ptr,
      static_cast<std::uint64_t>(quant_rows[index]),
      static_cast<std::uint64_t>(quant_cols[index]),
      static_cast<std::uint64_t>(quant_row_offsets[index]),
      static_cast<std::uint64_t>(quant_col_offsets[index]),
      static_cast<std::uint64_t>(quant_scale_row_strides[index]),
      quant_mode,
      static_cast<std::uint32_t>(quant_block_rows[index]),
      static_cast<std::uint32_t>(quant_block_cols[index]),
      quant_group_id < 0 ? UINT32_MAX : static_cast<std::uint32_t>(quant_group_id),
      static_cast<std::uint32_t>(peer),
      static_cast<std::uint32_t>(ordinal),
    });
    ++actual[peer];
  }
  if (actual != expected) {
    throw std::runtime_error("nccl_device_v2 task count does not match expected peer counts");
  }
  return tasks;
}

py::dict launch(int64_t handle, const py::list& tensors, const std::vector<int64_t>& lengths,
                const std::vector<int64_t>& tensor_offsets, const std::vector<int64_t>& tensor_row_bytes,
                const std::vector<int64_t>& tensor_row_strides, const std::vector<int64_t>& wire_dtypes,
                const std::vector<int64_t>& wire_element_bytes, const py::list& quant_scale_tensors,
                const std::vector<int64_t>& quant_modes, const std::vector<int64_t>& quant_rows,
                const std::vector<int64_t>& quant_cols, const std::vector<int64_t>& quant_row_offsets,
                const std::vector<int64_t>& quant_col_offsets, const std::vector<int64_t>& quant_scale_row_strides,
                const std::vector<int64_t>& quant_block_rows, const std::vector<int64_t>& quant_block_cols,
                const std::vector<int64_t>& quant_group_ids, const std::vector<int64_t>& peers,
                const std::vector<int64_t>& ordinals,
                const std::vector<int64_t>& expected_counts, bool sender, int64_t sequence) {
  using Clock = std::chrono::steady_clock;
  const auto launch_start = Clock::now();
  auto* state = reinterpret_cast<DeviceState*>(handle);
  if (state == nullptr) {
    throw std::runtime_error("Invalid nccl_device_v2 state handle");
  }
  if (sequence <= 0 || static_cast<std::uint64_t>(sequence) <= state->last_sequence) {
    throw std::runtime_error("nccl_device_v2 sequence must increase and be positive");
  }

  std::vector<std::uint32_t> active_peers;
  const auto tasks = build_tasks(*state, tensors, lengths, tensor_offsets, tensor_row_bytes, tensor_row_strides,
                                 wire_dtypes, wire_element_bytes, quant_scale_tensors, quant_modes, quant_rows,
                                 quant_cols, quant_row_offsets, quant_col_offsets, quant_scale_row_strides,
                                 quant_block_rows, quant_block_cols, quant_group_ids, peers, ordinals,
                                 expected_counts, sender, &active_peers);
  const v2::V2Direction direction = sender ? v2::V2Direction::kSend : v2::V2Direction::kRecv;
  AWEX_CUDA_V2_CHECK(cudaSetDevice(state->device));
  auto stream = at::cuda::getCurrentCUDAStream(state->device).stream();

  double host_lowering_time_ms = 0.0;
  double metadata_upload_time_ms = 0.0;
  double plan_initialization_time_ms = 0.0;
  const bool plan_cache_hit = state->plan_initialized;
  if (!state->plan_initialized) {
    const auto initialization_start = Clock::now();
    v2::V2LoweringConfig config;
    config.local_rank = static_cast<std::uint32_t>(state->rank);
    config.world_size = static_cast<std::uint32_t>(state->world_size);
    config.total_channels = state->total_channels;
    config.fifo_depth = state->fifo_depth;
    config.chunk_bytes = state->chunk_bytes;
    config.step_bytes = state->step_bytes;
    config.peer_channels = state->topology.peer_channels;
    const auto lowering_start = Clock::now();
    TmaSchedule tma_schedule;
    std::vector<v2::V2LoweringTask> generic_tasks = tasks;
    if (state->tma_supported && state->step_bytes >= v2::kTmaQuantPacketBytes) {
      std::vector<bool> tma_handled;
      tma_schedule = build_tma_schedule(tasks, direction, config, &tma_handled);
      generic_tasks = remove_tma_tasks(tasks, tma_handled, static_cast<std::uint32_t>(state->world_size));
      config.initial_steps = tma_schedule.next_steps;
    }
    auto schedule = v2::lowerFixedTasks(generic_tasks, active_peers, direction, config);
    host_lowering_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - lowering_start).count();
    auto quant_plan = build_quant_plan(generic_tasks);

    initialize_sparse_window(state, active_peers, direction, stream);
    LaunchBuffers buffers;
    try {
      const auto metadata_start = Clock::now();
      allocate_buffers(schedule, active_peers.size(), quant_plan.matrices.size(), quant_plan.blocks.size(),
                       tma_schedule, &buffers);
      upload_buffers(schedule, active_peers, quant_plan.matrices, quant_plan.blocks, tma_schedule, &buffers, stream);
      metadata_upload_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - metadata_start).count();
      state->tasks = tasks;
      state->active_peers = active_peers;
      state->quant_matrices = std::move(quant_plan.matrices);
      state->quant_blocks = std::move(quant_plan.blocks);
      state->tma_schedule = std::move(tma_schedule);
      state->schedule = std::move(schedule);
      state->buffers = buffers;
      state->direction = direction;
      state->plan_initialized = true;
    } catch (...) {
      release_buffers(&buffers);
      throw;
    }
    plan_initialization_time_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - initialization_start).count();
  } else if (direction != state->direction || active_peers != state->active_peers || !same_tasks(tasks, state->tasks)) {
    throw std::runtime_error("nccl_device_v2 launch descriptors changed after the fixed plan was cached");
  }

  const auto& schedule = state->schedule;
  const auto& cached_peers = state->active_peers;

  py::dict metrics;
  metrics["work_count"] = py::int_(schedule.works.size());
  metrics["fragment_count"] = py::int_(schedule.fragments.size());
  metrics["chunk_count"] = py::int_(schedule.chunk_count);
  metrics["batch_count"] = py::int_(schedule.batches.size());
  metrics["channel_count"] = py::int_(schedule.channel_count);
  metrics["active_peer_count"] = py::int_(cached_peers.size());
  metrics["fifo_depth"] = py::int_(state->fifo_depth);
  metrics["threads_per_channel"] = py::int_(v2::kThreadsPerBlock);
  metrics["warps_per_channel"] = py::int_(v2::kWarpsPerBlock);
  metrics["vector_bytes"] = py::int_(v2::kCopyPackBytes);
  metrics["copy_unroll"] = py::int_(v2::kCopyUnroll);
  std::uint64_t tensor_bytes = 0;
  std::uint64_t wire_bytes = 0;
  std::uint64_t streaming_cast_tasks = 0;
  std::uint64_t blockwise_fp8_tasks = 0;
  for (const auto& task : tasks) {
    wire_bytes += task.nbytes;
    tensor_bytes += task.nbytes / task.wire_element_bytes * task.tensor_element_bytes;
    if (task.tensor_dtype != task.wire_dtype || task.tensor_element_bytes != task.wire_element_bytes) {
      ++streaming_cast_tasks;
    }
    if (task.quant_mode == v2::V2QuantMode::kBlockwiseFloat8E4M3) {
      ++blockwise_fp8_tasks;
    }
  }
  metrics["tensor_bytes"] = py::int_(tensor_bytes);
  metrics["wire_bytes"] = py::int_(wire_bytes);
  metrics["streaming_cast_tasks"] = py::int_(streaming_cast_tasks);
  metrics["blockwise_fp8_tasks"] = py::int_(blockwise_fp8_tasks);
  metrics["blockwise_fp8_matrix_count"] = py::int_(state->quant_matrices.size());
  metrics["blockwise_fp8_block_count"] = py::int_(state->quant_blocks.size());
  metrics["fused_tma_supported"] = py::bool_(state->tma_supported);
  metrics["fused_tma_threads"] = py::int_(v2::kTmaQuantThreads);
  metrics["fused_tma_control_warps"] = py::int_(1);
  metrics["fused_tma_worker_warps"] = py::int_(v2::kTmaQuantThreads / v2::kWarpSize - 1);
  metrics["fused_tma_matrix_count"] = py::int_(state->tma_schedule.matrix_count);
  metrics["fused_tma_tile_count"] = py::int_(state->tma_schedule.tiles.size());
  metrics["fused_tma_queue_count"] = py::int_(state->tma_schedule.queues.size());
  metrics["fused_tma_packet_bytes"] = py::int_(v2::kTmaQuantPacketBytes);
  metrics["fused_tma_wire_bytes"] =
    py::int_(state->tma_schedule.tiles.size() * v2::kTmaQuantPacketBytes);
  metrics["channel_limit"] = py::int_(state->total_channels);
  metrics["topology_requested_channels_per_peer"] = py::int_(state->topology.requested_channels_per_peer);
  metrics["topology_channels_per_peer"] = py::int_(state->topology.channels_per_peer);
  metrics["topology_nvml_available"] = py::bool_(state->topology.nvml_available);
  std::uint32_t active_nvlink_count = 0;
  std::uint32_t active_raw_channels = 0;
  float active_path_bandwidth_gbps = 0.0F;
  for (const std::uint32_t peer : cached_peers) {
    active_nvlink_count = std::max(active_nvlink_count, state->topology.peer_paths[peer].nvlink_count);
    active_raw_channels = std::max(active_raw_channels, state->topology.peer_paths[peer].raw_channels);
    active_path_bandwidth_gbps = std::max(active_path_bandwidth_gbps, state->topology.peer_paths[peer].bandwidth_gbps);
  }
  metrics["topology_nvlink_count"] = py::int_(active_nvlink_count);
  metrics["topology_raw_channels"] = py::int_(active_raw_channels);
  metrics["topology_path_bandwidth_gbps"] = py::float_(active_path_bandwidth_gbps);
  metrics["slot_bytes"] = py::int_(state->step_bytes);
  metrics["payload_peer_count"] = py::int_(state->layout.payload_peer_count);
  metrics["control_window_bytes"] = py::int_(state->layout.payload_offset);
  metrics["payload_buffer_bytes"] = py::int_(
    static_cast<std::size_t>(state->layout.payload_peer_count) * state->total_channels * state->fifo_depth *
    state->step_bytes);
  metrics["registered_window_bytes"] = py::int_(state->window_bytes);
  metrics["dense_window_bytes"] = py::int_(state->dense_window_bytes);
  metrics["registered_window_savings_bytes"] = py::int_(state->dense_window_bytes - state->window_bytes);
  metrics["plan_cache_hit"] = py::bool_(plan_cache_hit);
  metrics["host_lowering_time_ms"] = host_lowering_time_ms;
  metrics["metadata_upload_time_ms"] = metadata_upload_time_ms;
  metrics["plan_initialization_time_ms"] = plan_initialization_time_ms;

  AWEX_CUDA_V2_CHECK(cudaMemsetAsync(state->local_base, 0, state->layout.payload_offset, stream));
  const v2::V2KernelArgs args{
      state->buffers.works,
      state->buffers.fragments,
      state->buffers.batches,
      state->buffers.channels,
      state->buffers.channel_ids,
      state->buffers.active_peers,
      schedule.channel_count,
      static_cast<std::uint32_t>(cached_peers.size()),
      static_cast<std::uint32_t>(state->rank),
      static_cast<std::uint32_t>(state->world_size),
      direction,
      state->layout,
      reinterpret_cast<std::uint8_t*>(state->local_base),
      state->device_peer_windows,
      state->device_payload_peer_slots,
      static_cast<unsigned long long>(sequence),
      state->timeout_cycles,
  };
  const v2::V2TmaKernelArgs tma_args{
      args,
      state->buffers.tma_tensor_maps,
      state->buffers.tma_tiles,
      state->buffers.tma_queues,
      static_cast<std::uint32_t>(state->tma_schedule.queues.size()),
  };
  const auto kernel_start = Clock::now();
  cudaEvent_t quant_start = nullptr;
  cudaEvent_t quant_end = nullptr;
  cudaEvent_t tma_start = nullptr;
  cudaEvent_t tma_end = nullptr;
  if (!state->quant_matrices.empty()) {
    AWEX_CUDA_V2_CHECK(cudaEventCreate(&quant_start));
    AWEX_CUDA_V2_CHECK(cudaEventCreate(&quant_end));
    AWEX_CUDA_V2_CHECK(cudaEventRecord(quant_start, stream));
    AWEX_CUDA_V2_CHECK(v2::launchBlockwiseFp8Scales(state->buffers.quant_matrices,
                                                    state->buffers.quant_blocks,
                                                    static_cast<std::uint32_t>(state->quant_blocks.size()), stream));
    AWEX_CUDA_V2_CHECK(cudaEventRecord(quant_end, stream));
  }
  if (!state->tma_schedule.queues.empty()) {
    AWEX_CUDA_V2_CHECK(cudaEventCreate(&tma_start));
    AWEX_CUDA_V2_CHECK(cudaEventCreate(&tma_end));
    AWEX_CUDA_V2_CHECK(cudaEventRecord(tma_start, stream));
    AWEX_CUDA_V2_CHECK(v2::launchDeviceV2Tma(tma_args, direction, stream));
    AWEX_CUDA_V2_CHECK(cudaEventRecord(tma_end, stream));
  }
  AWEX_CUDA_V2_CHECK(v2::launchDeviceV2(args, stream));
  AWEX_CUDA_V2_CHECK(cudaStreamSynchronize(stream));
  metrics["kernel_transfer_time_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - kernel_start).count();
  float quant_scale_kernel_time_ms = 0.0F;
  if (quant_start != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaEventElapsedTime(&quant_scale_kernel_time_ms, quant_start, quant_end));
    AWEX_CUDA_V2_CHECK(cudaEventDestroy(quant_start));
    AWEX_CUDA_V2_CHECK(cudaEventDestroy(quant_end));
  }
  metrics["quant_scale_kernel_time_ms"] = py::float_(quant_scale_kernel_time_ms);
  float fused_tma_kernel_time_ms = 0.0F;
  if (tma_start != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaEventElapsedTime(&fused_tma_kernel_time_ms, tma_start, tma_end));
    AWEX_CUDA_V2_CHECK(cudaEventDestroy(tma_start));
    AWEX_CUDA_V2_CHECK(cudaEventDestroy(tma_end));
  }
  metrics["fused_tma_kernel_time_ms"] = py::float_(fused_tma_kernel_time_ms);

  v2::V2WindowHeader header{};
  AWEX_CUDA_V2_CHECK(cudaMemcpy(&header, state->local_base, sizeof(header), cudaMemcpyDeviceToHost));
  if (header.error != 0) {
    throw std::runtime_error(
      "nccl_device_v2 kernel aborted while waiting for the peer (error=" + std::to_string(header.error) + ")");
  }
  state->last_sequence = static_cast<std::uint64_t>(sequence);
  metrics["next_step"] = py::int_(schedule.next_step);
  metrics["extension_total_time_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - launch_start).count();
  return metrics;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("unique_id_size", []() { return sizeof(ncclUniqueId); });
  module.def("get_unique_id", []() {
    ncclUniqueId unique_id;
    AWEX_NCCL_V2_CHECK(ncclGetUniqueId(&unique_id));
    return py::bytes(reinterpret_cast<const char*>(&unique_id), sizeof(unique_id));
  });
  module.def("create", [](const py::bytes& id, int world_size, int rank, int device, int timeout_ms, int max_channels,
                          int fifo_depth, int64_t step_bytes, int64_t chunk_bytes) {
    if (step_bytes <= 0 || chunk_bytes < 0) {
      throw std::runtime_error("invalid nccl_device_v2 step/chunk bytes");
    }
    const std::string unique_id = id;
    auto state = make_state(unique_id, world_size, rank, device, timeout_ms, static_cast<std::uint32_t>(max_channels),
                            static_cast<std::uint32_t>(fifo_depth), static_cast<std::size_t>(step_bytes),
                            static_cast<std::size_t>(chunk_bytes));
    return reinterpret_cast<int64_t>(state.release());
  });
  module.def("launch", &launch);
  module.def("destroy", [](int64_t handle) {
    auto* state = reinterpret_cast<DeviceState*>(handle);
    if (state == nullptr) {
      return;
    }
    try {
      destroy_state(state);
    } catch (...) {
      delete state;
      throw;
    }
    delete state;
  });
}

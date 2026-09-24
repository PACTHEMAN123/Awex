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
#include <stdexcept>
#include <string>
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
  bool plan_initialized = false;
  v2::V2Direction direction = v2::V2Direction::kSend;
  std::vector<v2::V2LoweringTask> tasks;
  std::vector<std::uint32_t> active_peers;
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
}

void allocate_buffers(const v2::V2Schedule& schedule, std::size_t active_peer_count, LaunchBuffers* buffers) {
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
  } catch (...) {
    release_buffers(buffers);
    throw;
  }
}

void upload_buffers(const v2::V2Schedule& schedule, const std::vector<std::uint32_t>& active_peers,
                    LaunchBuffers* buffers, cudaStream_t stream) {
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
        a.peer != b.peer || a.ordinal != b.ordinal) {
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
  const std::vector<int64_t>& wire_element_bytes, const std::vector<int64_t>& peers,
  const std::vector<int64_t>& ordinals, const std::vector<int64_t>& expected_counts, bool sender,
  std::vector<std::uint32_t>* active_peers) {
  if (tensors.size() != lengths.size() || tensors.size() != tensor_offsets.size() ||
      tensors.size() != tensor_row_bytes.size() || tensors.size() != tensor_row_strides.size() ||
      tensors.size() != wire_dtypes.size() || tensors.size() != wire_element_bytes.size() ||
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
    const int64_t peer = peers[index];
    const int64_t ordinal = ordinals[index];
    if (peer < 0 || peer >= state.world_size || peer == state.rank) {
      throw std::runtime_error("nccl_device_v2 task peer is invalid");
    }
    if (ordinal < 0 || static_cast<std::uint64_t>(ordinal) >= expected[peer]) {
      throw std::runtime_error("nccl_device_v2 task ordinal is invalid");
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
                const std::vector<int64_t>& wire_element_bytes, const std::vector<int64_t>& peers,
                const std::vector<int64_t>& ordinals, const std::vector<int64_t>& expected_counts, bool sender,
                int64_t sequence) {
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
                                 wire_dtypes, wire_element_bytes, peers, ordinals, expected_counts, sender,
                                 &active_peers);
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
    auto schedule = v2::lowerFixedTasks(tasks, active_peers, direction, config);
    host_lowering_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - lowering_start).count();

    initialize_sparse_window(state, active_peers, direction, stream);
    LaunchBuffers buffers;
    try {
      const auto metadata_start = Clock::now();
      allocate_buffers(schedule, active_peers.size(), &buffers);
      upload_buffers(schedule, active_peers, &buffers, stream);
      metadata_upload_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - metadata_start).count();
      state->tasks = tasks;
      state->active_peers = active_peers;
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
  for (const auto& task : tasks) {
    wire_bytes += task.nbytes;
    tensor_bytes += task.nbytes / task.wire_element_bytes * task.tensor_element_bytes;
    if (task.tensor_dtype != task.wire_dtype || task.tensor_element_bytes != task.wire_element_bytes) {
      ++streaming_cast_tasks;
    }
  }
  metrics["tensor_bytes"] = py::int_(tensor_bytes);
  metrics["wire_bytes"] = py::int_(wire_bytes);
  metrics["streaming_cast_tasks"] = py::int_(streaming_cast_tasks);
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
  const auto kernel_start = Clock::now();
  AWEX_CUDA_V2_CHECK(v2::launchDeviceV2(args, stream));
  AWEX_CUDA_V2_CHECK(cudaStreamSynchronize(stream));
  metrics["kernel_transfer_time_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - kernel_start).count();

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

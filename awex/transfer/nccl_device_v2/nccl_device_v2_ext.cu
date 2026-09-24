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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
namespace v2 = awex::nccl_device_v2;

namespace {

constexpr int kMaxRanks = 256;
constexpr int kMaxGinConnections = 4;

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
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  ncclDevComm dev_comm{};
  ncclGinType_t gin_type = NCCL_GIN_TYPE_NONE;
#endif
  void* local_base = nullptr;
  std::vector<void*> remote_bases;
  uintptr_t* device_peer_windows = nullptr;
  std::uint32_t* device_payload_peer_slots = nullptr;
  std::uint8_t* device_peer_transports = nullptr;
  std::uint8_t* device_gin_peer_contexts = nullptr;
  std::uint64_t* device_gin_channel_context_masks = nullptr;
  v2::V2WindowLayout layout{};
  std::size_t window_bytes = 0;
  std::size_t dense_window_bytes = 0;
  std::uint64_t timeout_cycles = 0;
  std::uint64_t last_sequence = 0;
  std::uint32_t total_channels = 1;
  v2::V2Topology topology;
  std::vector<std::uint32_t> peer_channels;
  std::uint32_t fifo_depth = v2::kDefaultFifoDepth;
  std::size_t chunk_bytes = v2::kDefaultChunkBytes;
  std::size_t step_bytes = v2::kDefaultStepBytes;
  std::size_t network_step_bytes = v2::kDefaultNetworkStepBytes;
  std::uint32_t requested_network_channels_per_peer = 0;
  std::uint32_t network_channels_per_peer = 1;
  std::uint32_t network_channel_budget = 0;
  std::uint32_t requested_gin_context_count = 0;
  std::uint32_t gin_doorbell_batch = 1;
  std::uint32_t gin_connection_count = 0;
  std::vector<std::uint64_t> peer_payload_bytes;
  std::vector<float> hca_effective_bandwidths_gbps;
  std::vector<std::uint8_t> hca_pci_distances;
  std::uint64_t numa_group_id = 0;
  std::vector<std::uint8_t> gin_peer_contexts;
  std::vector<std::uint64_t> gin_channel_context_masks;
  std::vector<std::uint64_t> gin_rail_scheduled_bytes;
  bool plan_initialized = false;
  bool window_initialized = false;
  v2::V2Direction direction = v2::V2Direction::kSend;
  std::vector<v2::V2LoweringTask> tasks;
  std::vector<std::uint32_t> active_peers;
  std::vector<std::uint32_t> window_active_peers;
  std::vector<std::uint8_t> peer_transports;
  v2::V2Schedule schedule;
  LaunchBuffers buffers;
  ncclTeam_t world_team{};
  ncclTeam_t lsa_team{};
  std::uint32_t gin_signal_count = 0;
  int nccl_version = 0;
  int rank = 0;
  int world_size = 0;
  int device = 0;
  bool gin_enabled = false;
  bool dev_comm_created = false;
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

std::vector<float> parse_float_list_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return {};
  std::vector<float> parsed;
  std::stringstream stream(value);
  std::string field;
  while (std::getline(stream, field, ',')) {
    try {
      const float number = std::stof(field);
      if (!std::isfinite(number) || number <= 0.0F) throw std::invalid_argument("non-positive");
      parsed.push_back(number);
    } catch (const std::exception&) {
      throw std::runtime_error(std::string(name) + " must contain positive comma-separated numbers");
    }
  }
  if (parsed.empty() || parsed.size() > kMaxGinConnections) {
    throw std::runtime_error(std::string(name) + " must contain between one and four values");
  }
  return parsed;
}

std::vector<std::uint8_t> parse_distance_list_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return {};
  std::vector<std::uint8_t> parsed;
  std::stringstream stream(value);
  std::string field;
  while (std::getline(stream, field, ',')) {
    try {
      const int number = std::stoi(field);
      if (number < 0 || number > 255) throw std::out_of_range("distance");
      parsed.push_back(static_cast<std::uint8_t>(number));
    } catch (const std::exception&) {
      throw std::runtime_error(std::string(name) + " must contain comma-separated byte values");
    }
  }
  return parsed;
}

std::uint64_t parse_uint64_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return 0;
  try {
    std::size_t consumed = 0;
    const std::uint64_t parsed = std::stoull(value, &consumed);
    if (value[consumed] != '\0') throw std::invalid_argument("trailing characters");
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(name) + " must be an unsigned integer");
  }
}

std::uint32_t power_of_two_down(std::uint32_t value) {
  std::uint32_t result = 1;
  while (result <= value / 2) result *= 2;
  return result;
}

std::uint32_t power_of_two_up(std::uint32_t value) {
  std::uint32_t result = 1;
  while (result < value && result < v2::kMaxChannels) result *= 2;
  return result;
}

std::unique_ptr<DeviceState> make_state(const std::string& unique_id_bytes, int world_size, int rank, int device,
                                        int timeout_ms, std::uint32_t max_channels, std::uint32_t fifo_depth,
                                        std::size_t step_bytes, std::size_t network_step_bytes,
                                        std::size_t chunk_bytes, std::uint32_t network_channels_per_peer,
                                        std::uint32_t gin_context_count, std::uint32_t gin_doorbell_batch) {
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
  if (network_channels_per_peer > v2::kMaxChannels) {
    throw std::runtime_error("invalid nccl_device_v2 network_channels_per_peer");
  }
  if (gin_context_count > v2::kMaxChannels) {
    throw std::runtime_error("invalid nccl_device_v2 GIN context count");
  }
  if (gin_doorbell_batch == 0 || gin_doorbell_batch > fifo_depth) {
    throw std::runtime_error("invalid nccl_device_v2 GIN doorbell batch");
  }
  if (fifo_depth == 0 || step_bytes == 0 || network_step_bytes == 0 ||
      step_bytes > std::numeric_limits<std::uint32_t>::max() ||
      network_step_bytes > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("nccl_device_v2 FIFO depth and step sizes must be positive and fit in V2Work");
  }
  if (chunk_bytes != 0 && chunk_bytes < std::max(step_bytes, network_step_bytes)) {
    throw std::runtime_error("nccl_device_v2 chunk_bytes must be zero or at least step_bytes");
  }
  auto state = std::make_unique<DeviceState>();
  state->rank = rank;
  state->world_size = world_size;
  state->device = device;
  state->fifo_depth = fifo_depth;
  state->step_bytes = step_bytes;
  state->network_step_bytes = network_step_bytes;
  state->requested_network_channels_per_peer = network_channels_per_peer;
  state->requested_gin_context_count = gin_context_count;
  state->gin_doorbell_batch = gin_doorbell_batch;
  state->chunk_bytes = chunk_bytes;
  state->hca_effective_bandwidths_gbps =
    parse_float_list_environment("AWEX_NCCL_DEVICE_V2_HCA_EFFECTIVE_BANDWIDTHS_GBPS");
  state->hca_pci_distances = parse_distance_list_environment("AWEX_NCCL_DEVICE_V2_HCA_PCI_DISTANCES");
  state->numa_group_id = parse_uint64_environment("AWEX_NCCL_DEVICE_V2_NUMA_GROUP_ID");
  if (!state->hca_effective_bandwidths_gbps.empty() && state->hca_pci_distances.empty()) {
    state->hca_pci_distances.assign(state->hca_effective_bandwidths_gbps.size(), 3);
  }
  if (state->hca_effective_bandwidths_gbps.size() != state->hca_pci_distances.size()) {
    throw std::runtime_error("nccl_device_v2 HCA bandwidth and PCI-distance tables differ in size");
  }
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
    AWEX_NCCL_V2_CHECK(ncclGetVersion(&state->nccl_version));
    state->world_team = ncclTeamWorld(state->comm);
    state->lsa_team = ncclTeamLsa(state->comm);
    state->peer_transports.assign(world_size, static_cast<std::uint8_t>(v2::V2Transport::kGin));
    for (int peer = 0; peer < world_size; ++peer) {
      if (ncclTeamRankIsMember(state->lsa_team, state->world_team, peer)) {
        state->peer_transports[peer] = static_cast<std::uint8_t>(v2::V2Transport::kLsa);
      }
    }
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
    state->gin_type = properties.ginType;
#endif

    state->topology = v2::discoverV2Topology(state->comm, world_size, rank, device, channel_limit);
    state->total_channels = state->topology.total_channels;
    state->peer_channels = state->topology.peer_channels;
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
  if (state->device_peer_transports != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(state->device_peer_transports));
    state->device_peer_transports = nullptr;
  }
  if (state->device_gin_peer_contexts != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(state->device_gin_peer_contexts));
    state->device_gin_peer_contexts = nullptr;
  }
  if (state->device_gin_channel_context_masks != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(state->device_gin_channel_context_masks));
    state->device_gin_channel_context_masks = nullptr;
  }
  if (state->device_payload_peer_slots != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(state->device_payload_peer_slots));
    state->device_payload_peer_slots = nullptr;
  }
  if (state->device_peer_windows != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(state->device_peer_windows));
    state->device_peer_windows = nullptr;
  }
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  if (state->dev_comm_created) {
    AWEX_NCCL_V2_CHECK(ncclDevCommDestroy(state->comm, &state->dev_comm));
    state->dev_comm_created = false;
  }
#endif
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
        a.tensor_row_bytes != b.tensor_row_bytes || a.tensor_row_stride != b.tensor_row_stride || a.peer != b.peer ||
        a.ordinal != b.ordinal) {
      return false;
    }
  }
  return true;
}

#if AWEX_NCCL_DEVICE_V2_HAS_GIN
struct GinRankLoadRecord {
  std::uint64_t numa_group_id = 0;
  std::array<std::uint64_t, kMaxRanks> peer_payload_bytes{};
  std::array<float, kMaxGinConnections> rail_bandwidths_gbps{};
  std::array<std::uint8_t, kMaxRanks> peer_channels{};
  std::array<std::uint8_t, kMaxRanks> peer_is_gin{};
  std::array<std::uint8_t, kMaxGinConnections> rail_pci_distances{};
  std::uint8_t rail_count = 0;
  std::uint8_t sender = 0;
};

struct GinFlowPart {
  std::uint32_t source = 0;
  std::uint32_t destination = 0;
  std::uint32_t part = 0;
  std::uint32_t part_count = 0;
  std::uint64_t bytes = 0;
};

using GinPartKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
using GinRailKey = std::pair<std::uint64_t, std::uint32_t>;

std::uint32_t gin_channels_for_record(const DeviceState& state, const GinRankLoadRecord& record,
                                      std::uint32_t peer) {
  const std::uint32_t max_channels =
    std::max<std::uint32_t>(1, std::min<std::uint32_t>(record.peer_channels[peer], state.total_channels));
  std::uint32_t min_channels = max_channels;
  while (static_cast<std::uint64_t>(min_channels) * state.world_size > state.total_channels && min_channels > 1) {
    min_channels /= 2;
  }
  return v2::v2ChannelsForBytes(record.peer_payload_bytes[peer], min_channels, max_channels,
                                state.network_step_bytes, true);
}

void configure_gin_peer_contexts(DeviceState* state, const std::vector<std::uint32_t>& gin_peers,
                                 v2::V2Direction direction, cudaStream_t stream) {
  const std::uint32_t context_count = static_cast<std::uint32_t>(state->dev_comm.ginContextCount);
  const std::uint32_t connection_count = state->gin_connection_count;
  const std::size_t table_size = static_cast<std::size_t>(state->world_size) * state->total_channels;
  state->gin_peer_contexts.resize(table_size);
  for (int peer = 0; peer < state->world_size; ++peer) {
    for (std::uint32_t channel = 0; channel < state->total_channels; ++channel) {
      state->gin_peer_contexts[static_cast<std::size_t>(peer) * state->total_channels + channel] =
        static_cast<std::uint8_t>(channel % context_count);
    }
  }
  state->gin_channel_context_masks.assign(state->total_channels, 0);
  state->gin_rail_scheduled_bytes.assign(state->hca_effective_bandwidths_gbps.size(), 0);

  GinRankLoadRecord local{};
  local.numa_group_id = state->numa_group_id;
  local.sender = direction == v2::V2Direction::kSend ? 1 : 0;
  local.rail_count = static_cast<std::uint8_t>(state->hca_effective_bandwidths_gbps.size());
  for (int peer = 0; peer < state->world_size; ++peer) {
    local.peer_payload_bytes[peer] = state->peer_payload_bytes[peer];
    local.peer_channels[peer] = static_cast<std::uint8_t>(state->peer_channels[peer]);
    local.peer_is_gin[peer] =
      state->peer_transports[peer] == static_cast<std::uint8_t>(v2::V2Transport::kGin) ? 1 : 0;
  }
  for (std::size_t rail = 0; rail < state->hca_effective_bandwidths_gbps.size(); ++rail) {
    local.rail_bandwidths_gbps[rail] = state->hca_effective_bandwidths_gbps[rail];
    local.rail_pci_distances[rail] = state->hca_pci_distances[rail];
  }
  auto records = v2::topology_detail::allGather(state->comm, &local, 1, state->world_size, stream);

  for (int source = 0; source < state->world_size; ++source) {
    if (records[source].sender == 0) continue;
    for (int destination = 0; destination < state->world_size; ++destination) {
      if (records[source].peer_payload_bytes[destination] == 0 ||
          records[source].peer_is_gin[destination] == 0) {
        continue;
      }
      const std::uint8_t shared_channels =
        std::max<std::uint8_t>(1, std::min(records[source].peer_channels[destination],
                                           records[destination].peer_channels[source]));
      records[source].peer_channels[destination] = shared_channels;
      records[destination].peer_channels[source] = shared_channels;
    }
  }
  state->network_channels_per_peer = 1;
  for (const std::uint32_t peer : gin_peers) {
    const std::uint32_t source = direction == v2::V2Direction::kSend ? state->rank : peer;
    const std::uint32_t destination = direction == v2::V2Direction::kSend ? peer : state->rank;
    state->peer_channels[peer] = records[source].peer_channels[destination];
    state->network_channels_per_peer =
      std::max(state->network_channels_per_peer, state->peer_channels[peer]);
  }

  std::vector<GinFlowPart> parts;
  for (int source = 0; source < state->world_size; ++source) {
    const auto& source_record = records[source];
    if (source_record.sender == 0 || source_record.rail_count == 0 || source_record.numa_group_id == 0) continue;
    for (int destination = 0; destination < state->world_size; ++destination) {
      const std::uint64_t bytes = source_record.peer_payload_bytes[destination];
      if (bytes == 0 || source_record.peer_is_gin[destination] == 0) continue;
      const auto& destination_record = records[destination];
      if (destination_record.rail_count == 0 || destination_record.numa_group_id == 0) continue;
      if (destination_record.sender == 0 && destination_record.peer_payload_bytes[source] != bytes) {
        throw std::runtime_error("nccl_device_v2 sender/receiver byte totals differ during GIN rail scheduling");
      }
      const std::uint32_t part_count = gin_channels_for_record(*state, source_record, destination);
      for (std::uint32_t part = 0; part < part_count; ++part) {
        const auto bounds = v2::v2PartBounds(part_count, part, bytes);
        if (bounds.first != bounds.second) {
          parts.push_back(GinFlowPart{static_cast<std::uint32_t>(source),
                                      static_cast<std::uint32_t>(destination), part, part_count,
                                      bounds.second - bounds.first});
        }
      }
    }
  }
  std::sort(parts.begin(), parts.end(), [](const GinFlowPart& left, const GinFlowPart& right) {
    if (left.bytes != right.bytes) return left.bytes > right.bytes;
    return std::tie(left.source, left.destination, left.part) <
           std::tie(right.source, right.destination, right.part);
  });

  std::map<GinRailKey, long double> rail_load;
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::array<std::uint32_t, kMaxGinConnections>>
    connection_uses;
  std::map<GinPartKey, std::uint8_t> part_contexts;
  const std::uint32_t contexts_per_connection = std::max<std::uint32_t>(1, context_count / connection_count);
  for (const GinFlowPart& part : parts) {
    const auto& source = records[part.source];
    const auto& destination = records[part.destination];
    const auto edge = std::make_pair(part.source, part.destination);
    auto& uses = connection_uses[edge];
    std::uint32_t selected_connection = 0;
    auto selected_key = std::tuple<long double, unsigned int, std::uint32_t, std::uint32_t>{
      std::numeric_limits<long double>::infinity(), std::numeric_limits<unsigned int>::max(),
      std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max()};
    for (std::uint32_t connection = 0; connection < connection_count; ++connection) {
      const std::uint32_t source_rail = connection % source.rail_count;
      const std::uint32_t destination_rail = connection % destination.rail_count;
      const long double bandwidth = std::max<long double>(
        1.0L, std::min(source.rail_bandwidths_gbps[source_rail],
                       destination.rail_bandwidths_gbps[destination_rail]));
      const long double work = static_cast<long double>(part.bytes) / bandwidth;
      const GinRailKey source_key{source.numa_group_id, source_rail};
      const GinRailKey destination_key{destination.numa_group_id, destination_rail};
      const long double projected = std::max(rail_load[source_key] + work, rail_load[destination_key] + work);
      const unsigned int distance = static_cast<unsigned int>(source.rail_pci_distances[source_rail]) +
                                    destination.rail_pci_distances[destination_rail];
      const std::uint32_t rotated = (connection + connection_count -
                                     ((part.source + part.destination) % connection_count)) %
                                    connection_count;
      const auto key = std::make_tuple(projected, distance, uses[connection], rotated);
      if (key < selected_key) {
        selected_key = key;
        selected_connection = connection;
      }
    }
    const std::uint32_t source_rail = selected_connection % source.rail_count;
    const std::uint32_t destination_rail = selected_connection % destination.rail_count;
    const long double bandwidth = std::max<long double>(
      1.0L, std::min(source.rail_bandwidths_gbps[source_rail],
                     destination.rail_bandwidths_gbps[destination_rail]));
    const long double work = static_cast<long double>(part.bytes) / bandwidth;
    rail_load[{source.numa_group_id, source_rail}] += work;
    rail_load[{destination.numa_group_id, destination_rail}] += work;
    const std::uint32_t use = uses[selected_connection]++;
    const std::uint32_t context =
      selected_connection + (use % contexts_per_connection) * connection_count;
    part_contexts[{part.source, part.destination, part.part}] = static_cast<std::uint8_t>(context);
  }

  for (const std::uint32_t peer : gin_peers) {
    const std::uint32_t source = direction == v2::V2Direction::kSend ? state->rank : peer;
    const std::uint32_t destination = direction == v2::V2Direction::kSend ? peer : state->rank;
    const auto& source_record = records[source];
    const std::uint32_t part_count = gin_channels_for_record(*state, source_record, destination);
    const std::uint32_t channel_base =
      v2::v2ChannelBase(source, destination, state->total_channels, part_count);
    for (std::uint32_t part = 0; part < part_count; ++part) {
      const std::uint32_t channel = (channel_base + part) & (state->total_channels - 1);
      const auto found = part_contexts.find({source, destination, part});
      const std::uint8_t context = found == part_contexts.end()
        ? static_cast<std::uint8_t>(channel % context_count)
        : found->second;
      state->gin_peer_contexts[static_cast<std::size_t>(peer) * state->total_channels + channel] = context;
      state->gin_channel_context_masks[channel] |= std::uint64_t{1} << context;
      if (!state->gin_rail_scheduled_bytes.empty()) {
        const auto bounds = v2::v2PartBounds(part_count, part, source_record.peer_payload_bytes[destination]);
        const std::uint32_t connection = context % connection_count;
        const std::uint32_t local_rail = connection % state->gin_rail_scheduled_bytes.size();
        state->gin_rail_scheduled_bytes[local_rail] += bounds.second - bounds.first;
      }
    }
  }
  for (std::uint32_t channel = 0; channel < state->total_channels; ++channel) {
    if (state->gin_channel_context_masks[channel] == 0) {
      state->gin_channel_context_masks[channel] = std::uint64_t{1} << (channel % context_count);
    }
  }

  AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->device_gin_peer_contexts),
                                state->gin_peer_contexts.size() * sizeof(std::uint8_t)));
  AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(state->device_gin_peer_contexts, state->gin_peer_contexts.data(),
                                     state->gin_peer_contexts.size() * sizeof(std::uint8_t),
                                     cudaMemcpyHostToDevice, stream));
  AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->device_gin_channel_context_masks),
                                state->gin_channel_context_masks.size() * sizeof(std::uint64_t)));
  AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(state->device_gin_channel_context_masks,
                                     state->gin_channel_context_masks.data(),
                                     state->gin_channel_context_masks.size() * sizeof(std::uint64_t),
                                     cudaMemcpyHostToDevice, stream));
}
#endif

void initialize_sparse_window(DeviceState* state, const std::vector<std::uint32_t>& active_peers,
                              const std::vector<v2::V2LoweringTask>& tasks, v2::V2Direction direction,
                              cudaStream_t stream) {
  if (state->window_initialized) {
    if (active_peers != state->window_active_peers) {
      throw std::runtime_error("nccl_device_v2 active peers changed after window initialization");
    }
    return;
  }
  const std::uint32_t inactive = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> local_payload_slots(state->world_size, inactive);
  for (std::size_t index = 0; index < active_peers.size(); ++index) {
    local_payload_slots[active_peers[index]] = static_cast<std::uint32_t>(index);
  }
  const auto payload_peer_slots = v2::topology_detail::allGather(
    state->comm, local_payload_slots.data(), local_payload_slots.size(), state->world_size, stream);
  std::uint32_t payload_peer_count = 0;
  for (int source = 0; source < state->world_size; ++source) {
    std::uint32_t source_peer_count = 0;
    for (int peer = 0; peer < state->world_size; ++peer) {
      const bool source_active = payload_peer_slots[static_cast<std::size_t>(source) * state->world_size + peer] !=
                                 inactive;
      const bool peer_active =
        payload_peer_slots[static_cast<std::size_t>(peer) * state->world_size + source] != inactive;
      if (source_active != peer_active) {
        throw std::runtime_error("nccl_device_v2 peer plans are not symmetric across ranks");
      }
      source_peer_count += source_active ? 1U : 0U;
    }
    payload_peer_count = std::max(payload_peer_count, source_peer_count);
  }

  const bool local_gin = std::any_of(active_peers.begin(), active_peers.end(), [&](std::uint32_t peer) {
    return state->peer_transports[peer] == static_cast<std::uint8_t>(v2::V2Transport::kGin);
  });
  const std::uint32_t local_gin_flag = local_gin ? 1U : 0U;
  const auto gin_flags =
    v2::topology_detail::allGather(state->comm, &local_gin_flag, 1, state->world_size, stream);
  state->gin_enabled = std::any_of(gin_flags.begin(), gin_flags.end(), [](std::uint32_t value) { return value != 0; });
  if (state->gin_enabled) {
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
    if (state->nccl_version < NCCL_VERSION(2, 30, 4)) {
      throw std::runtime_error("nccl_device_v2 GIN transport requires NCCL 2.30.4 or newer at runtime");
    }
    if (state->gin_type == NCCL_GIN_TYPE_NONE) {
      throw std::runtime_error("nccl_device_v2 found non-LSA peers, but the NCCL communicator has no GIN support");
    }
#else
    throw std::runtime_error(
      "nccl_device_v2 found non-LSA peers, but this extension was not built with NCCL 2.30.4+ GIN headers");
#endif
  }

  const std::size_t slot_bytes = std::max(state->step_bytes, state->network_step_bytes);
  state->layout = v2::makeV2WindowLayout(state->world_size, state->total_channels, state->fifo_depth, slot_bytes,
                                         payload_peer_count);
  state->window_bytes = state->layout.window_bytes;
  state->dense_window_bytes = v2::makeV2WindowLayout(state->world_size, state->total_channels, state->fifo_depth,
                                                      slot_bytes, state->world_size)
                                .window_bytes;
  try {
    AWEX_NCCL_V2_CHECK(ncclMemAlloc(&state->local_base, state->window_bytes));
    AWEX_NCCL_V2_CHECK(ncclCommWindowRegister(state->comm, state->local_base, state->window_bytes, &state->window,
                                              NCCL_WIN_COLL_SYMMETRIC));

    state->remote_bases.resize(state->world_size, nullptr);
    state->remote_bases[state->rank] = state->local_base;
    for (int peer = 0; peer < state->world_size; ++peer) {
      if (peer == state->rank ||
          state->peer_transports[peer] == static_cast<std::uint8_t>(v2::V2Transport::kGin)) {
        continue;
      }
      const int lsa_rank = ncclTeamRankToTeam(state->lsa_team, state->world_team, peer);
      if (lsa_rank < 0) throw std::runtime_error("nccl_device_v2 failed to map an LSA peer rank");
      AWEX_NCCL_V2_CHECK(ncclGetLsaDevicePointer(state->window, 0, lsa_rank, &state->remote_bases[peer]));
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

    AWEX_CUDA_V2_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->device_peer_transports),
                                  state->peer_transports.size() * sizeof(std::uint8_t)));
    AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(state->device_peer_transports, state->peer_transports.data(),
                                       state->peer_transports.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice,
                                       stream));

#if AWEX_NCCL_DEVICE_V2_HAS_GIN
    if (state->gin_enabled) {
      state->peer_payload_bytes.assign(state->world_size, 0);
      for (const auto& task : tasks) {
        auto& peer_bytes = state->peer_payload_bytes[task.peer];
        if (task.nbytes > std::numeric_limits<std::uint64_t>::max() - peer_bytes) {
          throw std::runtime_error("nccl_device_v2 peer payload size overflows");
        }
        peer_bytes += task.nbytes;
      }
      state->gin_signal_count = 2U * static_cast<std::uint32_t>(state->world_size) * state->total_channels;
      ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      // A request of one is rounded up by NCCL to one context per negotiated
      // connection. Explicit values remain available for diagnostics.
      requirements.ginContextCount = static_cast<int>(state->requested_gin_context_count == 0
          ? 1
          : std::min(state->requested_gin_context_count, state->total_channels));
      requirements.ginSignalCount = static_cast<int>(state->gin_signal_count);
      requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      requirements.worldGinBarrierCount = 1;
#if AWEX_NCCL_DEVICE_V2_HAS_EXPLICIT_SIGNAL_STRENGTH
      requirements.ginStrongSignalsRequired = true;
      requirements.ginVaSignalsRequired = false;
#endif
      AWEX_NCCL_V2_CHECK(ncclDevCommCreate(state->comm, &requirements, &state->dev_comm));
      state->dev_comm_created = true;
      if (state->dev_comm.ginContextCount == 0) {
        throw std::runtime_error("nccl_device_v2 GIN initialization returned no device contexts");
      }
      state->gin_connection_count = static_cast<std::uint32_t>(state->dev_comm.ginConnectionCount);
      if (state->gin_connection_count == 0) {
        throw std::runtime_error("nccl_device_v2 GIN initialization returned no network connections");
      }
      const std::uint32_t base_channels =
        std::min(state->total_channels, power_of_two_up(2 * state->gin_connection_count));
      state->network_channel_budget =
        std::min(state->total_channels, 6 * state->gin_connection_count);
      std::uint64_t total_gin_bytes = 0;
      std::vector<std::uint32_t> gin_peers;
      for (const std::uint32_t peer : active_peers) {
        if (state->peer_transports[peer] == static_cast<std::uint8_t>(v2::V2Transport::kGin)) {
          total_gin_bytes += state->peer_payload_bytes[peer];
          gin_peers.push_back(peer);
        }
      }
      state->network_channels_per_peer = 1;
      if (state->requested_network_channels_per_peer != 0) {
        const std::uint32_t channels = std::min(
          state->total_channels, power_of_two_up(state->requested_network_channels_per_peer));
        for (const std::uint32_t peer : gin_peers) state->peer_channels[peer] = channels;
      } else if (gin_peers.size() == 1) {
        // A single stream can use the whole issue budget without contending
        // with another peer. Round up so all negotiated connections remain
        // evenly striped.
        state->peer_channels[gin_peers.front()] =
          std::min(state->total_channels, power_of_two_up(state->network_channel_budget));
      } else {
        // First reserve only enough channels to keep one FIFO window in flight
        // for each peer. Small control peers should not consume the same eight
        // channels as multi-gigabyte weight streams.
        const std::uint64_t fifo_window_bytes =
          static_cast<std::uint64_t>(state->network_step_bytes) * state->fifo_depth;
        std::uint32_t assigned_channels = 0;
        for (const std::uint32_t peer : gin_peers) {
          const std::uint64_t window_demand =
            std::max<std::uint64_t>(1, v2::v2DivUp(state->peer_payload_bytes[peer], fifo_window_bytes));
          const std::uint32_t capped_demand =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(base_channels, window_demand));
          const std::uint32_t channels = power_of_two_up(capped_demand);
          state->peer_channels[peer] = channels;
          assigned_channels += channels;
        }

        // Promote only peers whose byte share justifies the next power of two,
        // and charge every promotion against the shared per-rank budget.
        std::sort(gin_peers.begin(), gin_peers.end(), [&](std::uint32_t left, std::uint32_t right) {
          return state->peer_payload_bytes[left] > state->peer_payload_bytes[right];
        });
        std::uint32_t remaining_channels = assigned_channels < state->network_channel_budget
          ? state->network_channel_budget - assigned_channels
          : 0;
        for (const std::uint32_t peer : gin_peers) {
          const std::uint64_t weighted_demand = total_gin_bytes == 0
            ? 1
            : v2::v2DivUp(static_cast<std::uint64_t>(state->network_channel_budget) *
                            state->peer_payload_bytes[peer],
                          total_gin_bytes);
          while (state->peer_channels[peer] <= remaining_channels &&
                 2 * state->peer_channels[peer] <= weighted_demand &&
                 state->peer_channels[peer] <= state->total_channels / 2) {
            remaining_channels -= state->peer_channels[peer];
            state->peer_channels[peer] *= 2;
          }
        }
      }
      for (const std::uint32_t peer : gin_peers) {
        state->network_channels_per_peer =
          std::max(state->network_channels_per_peer, state->peer_channels[peer]);
      }
      configure_gin_peer_contexts(state, gin_peers, direction, stream);
    }
#endif
    state->window_active_peers = active_peers;
    state->window_initialized = true;
  } catch (...) {
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
    if (state->dev_comm_created) {
      ncclDevCommDestroy(state->comm, &state->dev_comm);
      state->dev_comm_created = false;
    }
#endif
    if (state->device_peer_transports != nullptr) {
      cudaFree(state->device_peer_transports);
      state->device_peer_transports = nullptr;
    }
    if (state->device_gin_peer_contexts != nullptr) {
      cudaFree(state->device_gin_peer_contexts);
      state->device_gin_peer_contexts = nullptr;
    }
    if (state->device_gin_channel_context_masks != nullptr) {
      cudaFree(state->device_gin_channel_context_masks);
      state->device_gin_channel_context_masks = nullptr;
    }
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
    state->gin_connection_count = 0;
    state->network_channels_per_peer = 1;
    state->window_active_peers.clear();
    state->window_initialized = false;
    throw;
  }
}

std::vector<v2::V2LoweringTask> build_tasks(
  const DeviceState& state, const py::list& tensors, const std::vector<int64_t>& lengths,
  const std::vector<int64_t>& tensor_offsets, const std::vector<int64_t>& tensor_row_bytes,
  const std::vector<int64_t>& tensor_row_strides, const std::vector<int64_t>& peers,
  const std::vector<int64_t>& ordinals, const std::vector<int64_t>& expected_counts,
  std::vector<std::uint32_t>* active_peers) {
  if (tensors.size() != lengths.size() || tensors.size() != tensor_offsets.size() ||
      tensors.size() != tensor_row_bytes.size() || tensors.size() != tensor_row_strides.size() ||
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
    const int64_t peer = peers[index];
    const int64_t ordinal = ordinals[index];
    if (peer < 0 || peer >= state.world_size || peer == state.rank) {
      throw std::runtime_error("nccl_device_v2 task peer is invalid");
    }
    if (ordinal < 0 || static_cast<std::uint64_t>(ordinal) >= expected[peer]) {
      throw std::runtime_error("nccl_device_v2 task ordinal is invalid");
    }
    if (lengths[index] < 0 || tensor_offsets[index] < 0 || tensor_row_bytes[index] <= 0 ||
        tensor_row_strides[index] < tensor_row_bytes[index]) {
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
                const std::vector<int64_t>& tensor_row_strides, const std::vector<int64_t>& peers,
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
  const auto tasks = build_tasks(*state, tensors, lengths, tensor_offsets, tensor_row_bytes, tensor_row_strides, peers,
                                 ordinals, expected_counts, &active_peers);
  const v2::V2Direction direction = sender ? v2::V2Direction::kSend : v2::V2Direction::kRecv;
  AWEX_CUDA_V2_CHECK(cudaSetDevice(state->device));
  auto stream = at::cuda::getCurrentCUDAStream(state->device).stream();

  double host_lowering_time_ms = 0.0;
  double metadata_upload_time_ms = 0.0;
  double plan_initialization_time_ms = 0.0;
  const bool plan_cache_hit = state->plan_initialized;
  if (!state->plan_initialized) {
    const auto initialization_start = Clock::now();
    initialize_sparse_window(state, active_peers, tasks, direction, stream);
    v2::V2LoweringConfig config;
    config.local_rank = static_cast<std::uint32_t>(state->rank);
    config.world_size = static_cast<std::uint32_t>(state->world_size);
    config.total_channels = state->total_channels;
    config.fifo_depth = state->fifo_depth;
    config.chunk_bytes = state->chunk_bytes;
    config.step_bytes = state->step_bytes;
    config.network_step_bytes = state->network_step_bytes;
    config.peer_channels = state->peer_channels;
    config.peer_transports = state->peer_transports;
    const auto lowering_start = Clock::now();
    auto schedule = v2::lowerFixedTasks(tasks, active_peers, direction, config);
    host_lowering_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - lowering_start).count();

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
  std::uint32_t min_work_step_bytes = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t max_work_step_bytes = 0;
  for (const auto& work : schedule.works) {
    min_work_step_bytes = std::min(min_work_step_bytes, work.step_bytes);
    max_work_step_bytes = std::max(max_work_step_bytes, work.step_bytes);
  }
  if (schedule.works.empty()) min_work_step_bytes = 0;
  metrics["min_work_step_bytes"] = py::int_(min_work_step_bytes);
  metrics["max_work_step_bytes"] = py::int_(max_work_step_bytes);
  metrics["active_peer_count"] = py::int_(cached_peers.size());
  metrics["fifo_depth"] = py::int_(state->fifo_depth);
  metrics["threads_per_channel"] = py::int_(v2::kThreadsPerBlock);
  metrics["warps_per_channel"] = py::int_(v2::kWarpsPerBlock);
  metrics["vector_bytes"] = py::int_(v2::kCopyPackBytes);
  metrics["copy_unroll"] = py::int_(v2::kCopyUnroll);
  metrics["channel_limit"] = py::int_(state->total_channels);
  metrics["topology_requested_channels_per_peer"] = py::int_(state->topology.requested_channels_per_peer);
  metrics["topology_channels_per_peer"] = py::int_(state->topology.channels_per_peer);
  metrics["topology_nvml_available"] = py::bool_(state->topology.nvml_available);
  std::uint32_t active_nvlink_count = 0;
  std::uint32_t active_raw_channels = 0;
  std::uint32_t active_lsa_peers = 0;
  std::uint32_t active_gin_peers = 0;
  float active_path_bandwidth_gbps = 0.0F;
  for (const std::uint32_t peer : cached_peers) {
    if (state->peer_transports[peer] == static_cast<std::uint8_t>(v2::V2Transport::kGin)) {
      ++active_gin_peers;
    } else {
      ++active_lsa_peers;
    }
    active_nvlink_count = std::max(active_nvlink_count, state->topology.peer_paths[peer].nvlink_count);
    active_raw_channels = std::max(active_raw_channels, state->topology.peer_paths[peer].raw_channels);
    active_path_bandwidth_gbps = std::max(active_path_bandwidth_gbps, state->topology.peer_paths[peer].bandwidth_gbps);
  }
  metrics["topology_nvlink_count"] = py::int_(active_nvlink_count);
  metrics["topology_raw_channels"] = py::int_(active_raw_channels);
  metrics["topology_path_bandwidth_gbps"] = py::float_(active_path_bandwidth_gbps);
  metrics["lsa_peer_count"] = py::int_(active_lsa_peers);
  metrics["gin_peer_count"] = py::int_(active_gin_peers);
  metrics["gin_enabled"] = py::bool_(state->gin_enabled);
  metrics["gin_signal_count"] = py::int_(state->gin_signal_count);
  metrics["gin_connection_count"] = py::int_(state->gin_connection_count);
  metrics["requested_gin_context_count"] = py::int_(state->requested_gin_context_count);
  metrics["gin_doorbell_batch"] = py::int_(state->gin_doorbell_batch);
  const std::uint32_t gin_credit_batch = active_gin_peers > 1
    ? 1
    : std::min<std::uint32_t>(4, std::max<std::uint32_t>(1, state->fifo_depth / 2));
  metrics["gin_credit_batch"] = py::int_(gin_credit_batch);
  metrics["requested_network_channels_per_peer"] = py::int_(state->requested_network_channels_per_peer);
  metrics["network_channels_per_peer"] = py::int_(state->network_channels_per_peer);
  metrics["network_channel_budget"] = py::int_(state->network_channel_budget);
  py::list peer_channel_counts;
  py::list peer_payload_bytes;
  for (const std::uint32_t peer : cached_peers) {
    peer_channel_counts.append(py::int_(state->peer_channels[peer]));
    peer_payload_bytes.append(py::int_(state->peer_payload_bytes.empty() ? 0 : state->peer_payload_bytes[peer]));
  }
  metrics["active_peer_channel_limits"] = std::move(peer_channel_counts);
  metrics["active_peer_payload_bytes"] = std::move(peer_payload_bytes);
  py::list hca_bandwidths;
  for (const float bandwidth : state->hca_effective_bandwidths_gbps) {
    hca_bandwidths.append(py::float_(bandwidth));
  }
  metrics["gin_hca_effective_bandwidths_gbps"] = std::move(hca_bandwidths);
  py::list hca_distances;
  for (const std::uint8_t distance : state->hca_pci_distances) {
    hca_distances.append(py::int_(distance));
  }
  metrics["gin_hca_pci_distances"] = std::move(hca_distances);
  py::list rail_scheduled_bytes;
  for (const std::uint64_t bytes : state->gin_rail_scheduled_bytes) {
    rail_scheduled_bytes.append(py::int_(bytes));
  }
  metrics["gin_rail_scheduled_bytes"] = std::move(rail_scheduled_bytes);
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  metrics["gin_type"] = py::int_(static_cast<int>(state->gin_type));
  metrics["gin_context_count"] =
    py::int_(state->dev_comm_created ? static_cast<int>(state->dev_comm.ginContextCount) : 0);
#else
  metrics["gin_type"] = py::int_(0);
  metrics["gin_context_count"] = py::int_(0);
#endif
  metrics["local_step_bytes"] = py::int_(state->step_bytes);
  metrics["network_step_bytes"] = py::int_(state->network_step_bytes);
  metrics["slot_bytes"] = py::int_(state->layout.slot_bytes);
  metrics["payload_peer_count"] = py::int_(state->layout.payload_peer_count);
  metrics["control_window_bytes"] = py::int_(state->layout.payload_offset);
  metrics["payload_buffer_bytes"] = py::int_(
    static_cast<std::size_t>(state->layout.payload_peer_count) * state->total_channels * state->fifo_depth *
    state->layout.slot_bytes);
  metrics["registered_window_bytes"] = py::int_(state->window_bytes);
  metrics["dense_window_bytes"] = py::int_(state->dense_window_bytes);
  metrics["registered_window_savings_bytes"] = py::int_(state->dense_window_bytes - state->window_bytes);
  metrics["plan_cache_hit"] = py::bool_(plan_cache_hit);
  metrics["host_lowering_time_ms"] = host_lowering_time_ms;
  metrics["metadata_upload_time_ms"] = metadata_upload_time_ms;
  metrics["plan_initialization_time_ms"] = plan_initialization_time_ms;

  AWEX_CUDA_V2_CHECK(cudaMemsetAsync(state->local_base, 0, state->layout.payload_offset, stream));
  v2::V2KernelArgs args{};
  args.works = state->buffers.works;
  args.fragments = state->buffers.fragments;
  args.batches = state->buffers.batches;
  args.channels = state->buffers.channels;
  args.channel_ids = state->buffers.channel_ids;
  args.active_peers = state->buffers.active_peers;
  args.peer_transports = state->device_peer_transports;
  args.channel_count = schedule.channel_count;
  args.active_peer_count = static_cast<std::uint32_t>(cached_peers.size());
  args.local_rank = static_cast<std::uint32_t>(state->rank);
  args.world_size = static_cast<std::uint32_t>(state->world_size);
  args.direction = direction;
  args.layout = state->layout;
  args.local_window = reinterpret_cast<std::uint8_t*>(state->local_base);
  args.peer_windows = state->device_peer_windows;
  args.payload_peer_slots = state->device_payload_peer_slots;
  args.gin_peer_contexts = state->device_gin_peer_contexts;
  args.gin_channel_context_masks = state->device_gin_channel_context_masks;
  args.gin_enabled = state->gin_enabled ? 1U : 0U;
  args.gin_credit_batch = gin_credit_batch;
  args.gin_signal_count = state->gin_signal_count;
  args.gin_doorbell_batch = state->gin_doorbell_batch;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  args.window = state->window;
  if (state->dev_comm_created) args.dev_comm = state->dev_comm;
#endif
  args.epoch = static_cast<unsigned long long>(sequence);
  args.timeout_cycles = state->timeout_cycles;
  const auto kernel_start = Clock::now();
  AWEX_CUDA_V2_CHECK(v2::launchDeviceV2Reset(args, stream));
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
                          int fifo_depth, int64_t step_bytes, int64_t network_step_bytes, int64_t chunk_bytes,
                          int network_channels_per_peer, int gin_context_count, int gin_doorbell_batch) {
    if (step_bytes <= 0 || network_step_bytes <= 0 || chunk_bytes < 0) {
      throw std::runtime_error("invalid nccl_device_v2 step/chunk bytes");
    }
    if (network_channels_per_peer < 0 || gin_context_count < 0 || gin_doorbell_batch <= 0) {
      throw std::runtime_error("invalid nccl_device_v2 network parallelism");
    }
    const std::string unique_id = id;
    auto state = make_state(unique_id, world_size, rank, device, timeout_ms, static_cast<std::uint32_t>(max_channels),
                            static_cast<std::uint32_t>(fifo_depth), static_cast<std::size_t>(step_bytes),
                            static_cast<std::size_t>(network_step_bytes), static_cast<std::size_t>(chunk_bytes),
                            static_cast<std::uint32_t>(network_channels_per_peer),
                            static_cast<std::uint32_t>(gin_context_count),
                            static_cast<std::uint32_t>(gin_doorbell_batch));
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

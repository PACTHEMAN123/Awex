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

struct DeviceState {
  ncclComm_t comm = nullptr;
  ncclWindow_t window = nullptr;
  void* local_base = nullptr;
  std::vector<void*> remote_bases;
  uintptr_t* device_peer_windows = nullptr;
  v2::V2WindowLayout layout{};
  std::size_t window_bytes = 0;
  std::uint64_t timeout_cycles = 0;
  std::uint64_t next_step = 1;
  std::uint64_t last_sequence = 0;
  std::uint32_t channels_per_peer = v2::kDefaultChannelsPerPeer;
  std::uint32_t max_channels = 32;
  std::uint32_t fifo_depth = v2::kDefaultFifoDepth;
  std::size_t chunk_bytes = v2::kDefaultChunkBytes;
  std::size_t step_bytes = v2::kDefaultStepBytes;
  int rank = 0;
  int world_size = 0;
  int device = 0;
};

struct LaunchBuffers {
  v2::V2Work* works = nullptr;
  v2::V2WorkBatch* batches = nullptr;
  v2::V2ChannelQueue* channels = nullptr;
};

[[noreturn]] void throw_nccl(ncclResult_t result, const char* expression) {
  throw std::runtime_error(
      std::string(expression) + " failed: " + ncclGetErrorString(result));
}

void check_nccl(ncclResult_t result, const char* expression) {
  if (result != ncclSuccess) {
    throw_nccl(result, expression);
  }
}

void check_cuda(cudaError_t result, const char* expression) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(expression) + " failed: " + cudaGetErrorString(result));
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

std::unique_ptr<DeviceState> make_state(
    const std::string& unique_id_bytes,
    int world_size,
    int rank,
    int device,
    int timeout_ms,
    std::uint32_t channels_per_peer,
    std::uint32_t max_channels,
    std::uint32_t fifo_depth,
    std::size_t step_bytes,
    std::size_t chunk_bytes) {
  if (world_size < 2 || world_size > kMaxRanks) {
    throw std::runtime_error("nccl_device_v2 world_size must be in [2, 256]");
  }
  if (rank < 0 || rank >= world_size) {
    throw std::runtime_error("nccl_device_v2 rank is outside world_size");
  }
  if (unique_id_bytes.size() != sizeof(ncclUniqueId)) {
    throw std::runtime_error("Invalid NCCL unique id size");
  }
  if (channels_per_peer == 0 || channels_per_peer > v2::kMaxChannelsPerPeer) {
    throw std::runtime_error("invalid nccl_device_v2 channels_per_peer");
  }
  if (max_channels == 0 || max_channels < channels_per_peer) {
    throw std::runtime_error("invalid nccl_device_v2 max_channels");
  }
  if (fifo_depth == 0 || step_bytes == 0) {
    throw std::runtime_error("nccl_device_v2 FIFO depth and step size must be positive");
  }
  if (chunk_bytes != 0 && chunk_bytes < step_bytes) {
    throw std::runtime_error("nccl_device_v2 chunk_bytes must be zero or at least step_bytes");
  }
  const std::size_t layout_channels = std::min<std::size_t>(
      max_channels,
      checked_multiply(
          static_cast<std::size_t>(world_size),
          channels_per_peer,
          "nccl_device_v2 layout channels"));
  if (layout_channels == 0) {
    throw std::runtime_error("nccl_device_v2 layout has no channels");
  }

  auto state = std::make_unique<DeviceState>();
  state->rank = rank;
  state->world_size = world_size;
  state->device = device;
  state->channels_per_peer = channels_per_peer;
  state->max_channels = max_channels;
  state->fifo_depth = fifo_depth;
  state->step_bytes = step_bytes;
  state->chunk_bytes = chunk_bytes;
  state->layout = v2::makeV2WindowLayout(
      world_size,
      static_cast<std::uint32_t>(layout_channels),
      fifo_depth,
      step_bytes);
  state->window_bytes = state->layout.window_bytes;

  AWEX_CUDA_V2_CHECK(cudaSetDevice(device));
  int clock_rate_khz = 0;
  AWEX_CUDA_V2_CHECK(cudaDeviceGetAttribute(
      &clock_rate_khz,
      cudaDevAttrClockRate,
      device));
  state->timeout_cycles = static_cast<std::uint64_t>(std::max(timeout_ms, 1)) *
      static_cast<std::uint64_t>(clock_rate_khz);

  ncclUniqueId unique_id;
  std::memcpy(&unique_id, unique_id_bytes.data(), sizeof(unique_id));
  try {
    AWEX_NCCL_V2_CHECK(ncclCommInitRank(
        &state->comm,
        world_size,
        unique_id,
        rank));

    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    AWEX_NCCL_V2_CHECK(ncclCommQueryProperties(state->comm, &properties));
    if (!properties.deviceApiSupport) {
      throw std::runtime_error(
          "The loaded NCCL communicator does not support the Device API");
    }
    const ncclTeam_t lsa_team = ncclTeamLsa(state->comm);
    if (lsa_team.nRanks != world_size || lsa_team.rank != rank ||
        lsa_team.stride != 1) {
      throw std::runtime_error(
          "nccl_device_v2 requires a contiguous LSA domain");
    }

    AWEX_NCCL_V2_CHECK(ncclMemAlloc(
        &state->local_base,
        state->window_bytes));
    AWEX_NCCL_V2_CHECK(ncclCommWindowRegister(
        state->comm,
        state->local_base,
        state->window_bytes,
        &state->window,
        NCCL_WIN_COLL_SYMMETRIC));
    AWEX_CUDA_V2_CHECK(cudaMemset(
        state->local_base,
        0,
        state->layout.payload_offset));

    state->remote_bases.resize(world_size, nullptr);
    state->remote_bases[rank] = state->local_base;
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) {
        continue;
      }
      AWEX_NCCL_V2_CHECK(ncclGetLsaDevicePointer(
          state->window,
          0,
          peer,
          &state->remote_bases[peer]));
    }
    AWEX_CUDA_V2_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state->device_peer_windows),
        checked_multiply(
            static_cast<std::size_t>(world_size),
            sizeof(uintptr_t),
            "nccl_device_v2 peer window table")));
    std::vector<uintptr_t> peer_windows(world_size, 0);
    for (int peer = 0; peer < world_size; ++peer) {
      peer_windows[peer] = reinterpret_cast<uintptr_t>(state->remote_bases[peer]);
    }
    AWEX_CUDA_V2_CHECK(cudaMemcpy(
        state->device_peer_windows,
        peer_windows.data(),
        peer_windows.size() * sizeof(uintptr_t),
        cudaMemcpyHostToDevice));
  } catch (...) {
    if (state->device_peer_windows != nullptr) {
      cudaFree(state->device_peer_windows);
    }
    if (state->window != nullptr) {
      ncclCommWindowDeregister(state->comm, state->window);
    }
    if (state->local_base != nullptr) {
      ncclMemFree(state->local_base);
    }
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
  if (buffers->batches != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->batches));
    buffers->batches = nullptr;
  }
  if (buffers->channels != nullptr) {
    AWEX_CUDA_V2_CHECK(cudaFree(buffers->channels));
    buffers->channels = nullptr;
  }
}

void allocate_buffers(
    const v2::V2Schedule& schedule,
    LaunchBuffers* buffers) {
  try {
    if (!schedule.works.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&buffers->works),
          schedule.works.size() * sizeof(v2::V2Work)));
    }
    if (!schedule.batches.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&buffers->batches),
          schedule.batches.size() * sizeof(v2::V2WorkBatch)));
    }
    if (!schedule.channels.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&buffers->channels),
          schedule.channels.size() * sizeof(v2::V2ChannelQueue)));
    }
  } catch (...) {
    release_buffers(buffers);
    throw;
  }
}

std::vector<v2::V2LoweringTask> build_tasks(
    const DeviceState& state,
    const py::list& tensors,
    const std::vector<int64_t>& lengths,
    const std::vector<int64_t>& tensor_offsets,
    const std::vector<int64_t>& tensor_row_bytes,
    const std::vector<int64_t>& tensor_row_strides,
    const std::vector<int64_t>& peers,
    const std::vector<int64_t>& ordinals,
    const std::vector<int64_t>& expected_counts,
    std::vector<std::uint32_t>* active_peers) {
  if (tensors.size() != lengths.size() ||
      tensors.size() != tensor_offsets.size() ||
      tensors.size() != tensor_row_bytes.size() ||
      tensors.size() != tensor_row_strides.size() ||
      tensors.size() != peers.size() ||
      tensors.size() != ordinals.size()) {
    throw std::runtime_error("nccl_device_v2 tensor/task descriptor lengths do not match");
  }
  if (expected_counts.size() != static_cast<std::size_t>(state.world_size)) {
    throw std::runtime_error("nccl_device_v2 expected-count vector does not match world_size");
  }

  std::vector<std::uint32_t> expected(state.world_size, 0);
  for (int peer = 0; peer < state.world_size; ++peer) {
    if (expected_counts[peer] < 0 ||
        static_cast<std::uint64_t>(expected_counts[peer]) > UINT32_MAX) {
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
    if (lengths[index] < 0 || tensor_offsets[index] < 0 ||
        tensor_row_bytes[index] <= 0 ||
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

py::dict launch(
    int64_t handle,
    const py::list& tensors,
    const std::vector<int64_t>& lengths,
    const std::vector<int64_t>& tensor_offsets,
    const std::vector<int64_t>& tensor_row_bytes,
    const std::vector<int64_t>& tensor_row_strides,
    const std::vector<int64_t>& peers,
    const std::vector<int64_t>& ordinals,
    const std::vector<int64_t>& expected_counts,
    bool sender,
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
  const auto tasks = build_tasks(
      *state,
      tensors,
      lengths,
      tensor_offsets,
      tensor_row_bytes,
      tensor_row_strides,
      peers,
      ordinals,
      expected_counts,
      &active_peers);
  v2::V2LoweringConfig config;
  config.world_size = static_cast<std::uint32_t>(state->world_size);
  config.channels_per_peer = state->channels_per_peer;
  config.max_channels = state->max_channels;
  config.fifo_depth = state->fifo_depth;
  config.chunk_bytes = state->chunk_bytes;
  config.step_bytes = state->step_bytes;
  config.initial_step = state->next_step;
  const auto schedule = v2::lowerFixedTasks(
      tasks,
      active_peers,
      sender ? v2::V2Direction::kSend : v2::V2Direction::kRecv,
      config);

  py::dict metrics;
  metrics["work_count"] = py::int_(schedule.works.size());
  metrics["chunk_count"] = py::int_(schedule.chunk_count);
  metrics["batch_count"] = py::int_(schedule.batches.size());
  metrics["channel_count"] = py::int_(schedule.channel_count);
  metrics["active_peer_count"] = py::int_(active_peers.size());
  metrics["fifo_depth"] = py::int_(state->fifo_depth);
  metrics["slot_bytes"] = py::int_(state->step_bytes);
  metrics["registered_window_bytes"] = py::int_(state->window_bytes);
  metrics["host_lowering_time_ms"] =
      std::chrono::duration<double, std::milli>(Clock::now() - launch_start).count();

  AWEX_CUDA_V2_CHECK(cudaSetDevice(state->device));
  auto stream = at::cuda::getCurrentCUDAStream(state->device).stream();
  LaunchBuffers buffers;
  allocate_buffers(schedule, &buffers);
  try {
    if (!schedule.works.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(
          buffers.works,
          schedule.works.data(),
          schedule.works.size() * sizeof(v2::V2Work),
          cudaMemcpyHostToDevice,
          stream));
    }
    if (!schedule.batches.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(
          buffers.batches,
          schedule.batches.data(),
          schedule.batches.size() * sizeof(v2::V2WorkBatch),
          cudaMemcpyHostToDevice,
          stream));
    }
    if (!schedule.channels.empty()) {
      AWEX_CUDA_V2_CHECK(cudaMemcpyAsync(
          buffers.channels,
          schedule.channels.data(),
          schedule.channels.size() * sizeof(v2::V2ChannelQueue),
          cudaMemcpyHostToDevice,
          stream));
    }
    const auto metadata_done = Clock::now();
    metrics["metadata_upload_time_ms"] =
        std::chrono::duration<double, std::milli>(metadata_done - launch_start).count();

    const v2::V2KernelArgs args{
        buffers.works,
        buffers.batches,
        buffers.channels,
        schedule.channel_count,
        static_cast<std::uint32_t>(state->rank),
        static_cast<std::uint32_t>(state->world_size),
        state->layout,
        reinterpret_cast<std::uint8_t*>(state->local_base),
        state->device_peer_windows,
        static_cast<unsigned long long>(sequence),
        state->timeout_cycles,
    };
    const auto kernel_start = Clock::now();
    AWEX_CUDA_V2_CHECK(v2::launchDeviceV2(args, stream));
    AWEX_CUDA_V2_CHECK(cudaStreamSynchronize(stream));
    metrics["kernel_transfer_time_ms"] =
        std::chrono::duration<double, std::milli>(Clock::now() - kernel_start).count();

    v2::V2WindowHeader header{};
    AWEX_CUDA_V2_CHECK(cudaMemcpy(
        &header,
        state->local_base,
        sizeof(header),
        cudaMemcpyDeviceToHost));
    if (header.error != 0) {
      throw std::runtime_error("nccl_device_v2 kernel aborted while waiting for the peer");
    }
    state->next_step = schedule.next_step;
    state->last_sequence = static_cast<std::uint64_t>(sequence);
    metrics["next_step"] = py::int_(state->next_step);
    metrics["extension_total_time_ms"] =
        std::chrono::duration<double, std::milli>(Clock::now() - launch_start).count();
    release_buffers(&buffers);
  } catch (...) {
    release_buffers(&buffers);
    throw;
  }
  return metrics;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("unique_id_size", []() { return sizeof(ncclUniqueId); });
  module.def("get_unique_id", []() {
    ncclUniqueId unique_id;
    AWEX_NCCL_V2_CHECK(ncclGetUniqueId(&unique_id));
    return py::bytes(
        reinterpret_cast<const char*>(&unique_id),
        sizeof(unique_id));
  });
  module.def(
      "create",
      [](const py::bytes& id,
         int world_size,
         int rank,
         int device,
         int timeout_ms,
         int channels_per_peer,
         int max_channels,
         int fifo_depth,
         int64_t step_bytes,
         int64_t chunk_bytes) {
        if (step_bytes <= 0 || chunk_bytes < 0) {
          throw std::runtime_error("invalid nccl_device_v2 step/chunk bytes");
        }
        const std::string unique_id = id;
        auto state = make_state(
            unique_id,
            world_size,
            rank,
            device,
            timeout_ms,
            static_cast<std::uint32_t>(channels_per_peer),
            static_cast<std::uint32_t>(max_channels),
            static_cast<std::uint32_t>(fifo_depth),
            static_cast<std::size_t>(step_bytes),
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

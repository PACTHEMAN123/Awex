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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "impls/device_transfer.cuh"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace dt = awex::device_transfer;

namespace {

constexpr int kMaxRanks = 256;

struct DeviceState {
  ncclComm_t comm = nullptr;
  ncclDevComm_t* dev_comm = nullptr;
  ncclWindow_t window = nullptr;
  void* local_base = nullptr;
  void* multimem_base = nullptr;
  // Cached NCCL LSA device pointers; the transfer kernel uses direct
  // load/store operations instead of issuing a host-side NCCL P2P operation.
  std::vector<void*> remote_bases;
  size_t logical_data_bytes = 0;
  size_t window_bytes = 0;
  size_t ring_data_offset = 0;
  size_t ring_slot_bytes = 0;
  uint32_t ring_slots_per_peer = 0;
  uint64_t timeout_cycles = 0;
  uint64_t last_sequence = 0;
  int rank = 0;
  int world_size = 0;
  int device = 0;
  dt::DeviceTask* device_tasks = nullptr;
  uint32_t* device_expected_counts = nullptr;
  uint32_t* device_peer_offsets = nullptr;
  uint32_t* device_active_peers = nullptr;
  uintptr_t* device_target_bases = nullptr;
  dt::DeviceProfile* device_profile = nullptr;
  size_t task_capacity = 0;
  size_t expected_count_capacity = 0;
  size_t target_base_capacity = 0;
  cudaEvent_t metadata_start = nullptr;
  cudaEvent_t metadata_end = nullptr;
  cudaEvent_t kernel_end = nullptr;
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

#define AWEX_NCCL_CHECK(expr) check_nccl((expr), #expr)
#define AWEX_CUDA_CHECK(expr) check_cuda((expr), #expr)

size_t checked_add(size_t left, size_t right, const char* description) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::runtime_error(std::string(description) + " size overflows");
  }
  return left + right;
}

size_t checked_multiply(size_t left, size_t right, const char* description) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error(std::string(description) + " size overflows");
  }
  return left * right;
}

size_t align_up(size_t value, size_t alignment) {
  return checked_multiply(
      checked_add(value, alignment - 1, "aligned allocation") / alignment,
      alignment,
      "aligned allocation");
}

struct RingLayout {
  size_t slot_bytes;
  uint32_t slots_per_peer;
  size_t data_offset;
  size_t window_bytes;
};

RingLayout make_ring_layout(size_t logical_data_bytes, int world_size) {
  const size_t slot_bytes = dt::kDefaultRingSlotBytes;
  const size_t payload_slots = std::max<size_t>(
      1,
      checked_add(logical_data_bytes, slot_bytes - 1, "ring payload") /
          slot_bytes);
  const uint32_t slots_from_budget = std::max<uint32_t>(
      1, dt::kDefaultRingSlotBudget / static_cast<uint32_t>(world_size));
  const uint32_t slots_per_peer = static_cast<uint32_t>(std::min<size_t>(
      payload_slots,
      std::min<uint32_t>(
          dt::kDefaultRingSlotsPerPeer, slots_from_budget)));
  const size_t slot_count = checked_multiply(
      static_cast<size_t>(world_size), slots_per_peer, "ring slots");
  const size_t metadata_bytes = checked_add(
      sizeof(dt::ControlBlock),
      checked_multiply(
          slot_count, sizeof(dt::RingSlotState), "ring metadata"),
      "ring metadata");
  const size_t data_offset = align_up(metadata_bytes, dt::kRingAlignment);
  const size_t window_bytes = align_up(
      checked_add(
          data_offset,
          checked_multiply(slot_count, slot_bytes, "ring payload"),
          "ring window"),
      dt::kWindowAlignment);
  return RingLayout{slot_bytes, slots_per_peer, data_offset, window_bytes};
}

void ensure_launch_resources(
    DeviceState* state,
    size_t task_count,
    size_t expected_count_count,
    size_t target_base_count,
    bool collect_profile) {
  if (task_count > state->task_capacity) {
    dt::DeviceTask* new_tasks = nullptr;
    AWEX_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&new_tasks),
        task_count * sizeof(dt::DeviceTask)));
    if (state->device_tasks != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_tasks));
    }
    state->device_tasks = new_tasks;
    state->task_capacity = task_count;
  }

  if (target_base_count > state->target_base_capacity) {
    uintptr_t* new_target_bases = nullptr;
    AWEX_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&new_target_bases),
        target_base_count * sizeof(uintptr_t)));
    if (state->device_target_bases != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_target_bases));
    }
    state->device_target_bases = new_target_bases;
    state->target_base_capacity = target_base_count;
  }

  if (expected_count_count > state->expected_count_capacity) {
    uint32_t* new_expected_counts = nullptr;
    uint32_t* new_peer_offsets = nullptr;
    uint32_t* new_active_peers = nullptr;
    try {
      AWEX_CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&new_expected_counts),
          expected_count_count * sizeof(uint32_t)));
      AWEX_CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&new_peer_offsets),
          (expected_count_count + 1) * sizeof(uint32_t)));
      AWEX_CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&new_active_peers),
          expected_count_count * sizeof(uint32_t)));
    } catch (...) {
      if (new_expected_counts != nullptr) {
        cudaFree(new_expected_counts);
      }
      if (new_peer_offsets != nullptr) {
        cudaFree(new_peer_offsets);
      }
      if (new_active_peers != nullptr) {
        cudaFree(new_active_peers);
      }
      throw;
    }
    if (state->device_expected_counts != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_expected_counts));
    }
    if (state->device_peer_offsets != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_peer_offsets));
    }
    if (state->device_active_peers != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_active_peers));
    }
    state->device_expected_counts = new_expected_counts;
    state->device_peer_offsets = new_peer_offsets;
    state->device_active_peers = new_active_peers;
    state->expected_count_capacity = expected_count_count;
  }

  if (collect_profile && state->device_profile == nullptr) {
    dt::DeviceProfile* new_profile = nullptr;
    cudaEvent_t new_metadata_start = nullptr;
    cudaEvent_t new_metadata_end = nullptr;
    cudaEvent_t new_kernel_end = nullptr;
    try {
      AWEX_CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&new_profile), sizeof(dt::DeviceProfile)));
      AWEX_CUDA_CHECK(cudaEventCreate(&new_metadata_start));
      AWEX_CUDA_CHECK(cudaEventCreate(&new_metadata_end));
      AWEX_CUDA_CHECK(cudaEventCreate(&new_kernel_end));
    } catch (...) {
      if (new_profile != nullptr) {
        cudaFree(new_profile);
      }
      if (new_metadata_start != nullptr) {
        cudaEventDestroy(new_metadata_start);
      }
      if (new_metadata_end != nullptr) {
        cudaEventDestroy(new_metadata_end);
      }
      if (new_kernel_end != nullptr) {
        cudaEventDestroy(new_kernel_end);
      }
      throw;
    }
    state->device_profile = new_profile;
    state->metadata_start = new_metadata_start;
    state->metadata_end = new_metadata_end;
    state->kernel_end = new_kernel_end;
  }
}

void release_launch_resources(DeviceState* state) {
  if (state->device_tasks != nullptr) {
    AWEX_CUDA_CHECK(cudaFree(state->device_tasks));
    state->device_tasks = nullptr;
  }
  if (state->device_expected_counts != nullptr) {
    AWEX_CUDA_CHECK(cudaFree(state->device_expected_counts));
    state->device_expected_counts = nullptr;
  }
  if (state->device_peer_offsets != nullptr) {
    AWEX_CUDA_CHECK(cudaFree(state->device_peer_offsets));
    state->device_peer_offsets = nullptr;
  }
  if (state->device_active_peers != nullptr) {
    AWEX_CUDA_CHECK(cudaFree(state->device_active_peers));
    state->device_active_peers = nullptr;
  }
  if (state->device_target_bases != nullptr) {
    AWEX_CUDA_CHECK(cudaFree(state->device_target_bases));
    state->device_target_bases = nullptr;
  }
  if (state->device_profile != nullptr) {
    AWEX_CUDA_CHECK(cudaFree(state->device_profile));
    state->device_profile = nullptr;
  }
  if (state->metadata_start != nullptr) {
    AWEX_CUDA_CHECK(cudaEventDestroy(state->metadata_start));
    state->metadata_start = nullptr;
  }
  if (state->metadata_end != nullptr) {
    AWEX_CUDA_CHECK(cudaEventDestroy(state->metadata_end));
    state->metadata_end = nullptr;
  }
  if (state->kernel_end != nullptr) {
    AWEX_CUDA_CHECK(cudaEventDestroy(state->kernel_end));
    state->kernel_end = nullptr;
  }
  state->task_capacity = 0;
  state->expected_count_capacity = 0;
  state->target_base_capacity = 0;
}

std::unique_ptr<DeviceState> make_state(
    const std::string& unique_id_bytes,
    int world_size,
    int rank,
    size_t data_bytes,
    int device,
    int timeout_ms) {
  if (world_size < 2 || world_size > kMaxRanks) {
    throw std::runtime_error("nccl_device world_size must be in [2, 256]");
  }
  if (rank < 0 || rank >= world_size) {
    throw std::runtime_error("nccl_device rank is outside world_size");
  }
  if (unique_id_bytes.size() != sizeof(ncclUniqueId)) {
    throw std::runtime_error("Invalid NCCL unique id size");
  }

  auto state = std::make_unique<DeviceState>();
  state->rank = rank;
  state->world_size = world_size;
  state->device = device;
  AWEX_CUDA_CHECK(cudaSetDevice(device));
  int clock_rate_khz = 0;
  AWEX_CUDA_CHECK(cudaDeviceGetAttribute(
      &clock_rate_khz, cudaDevAttrClockRate, device));
  state->timeout_cycles = static_cast<uint64_t>(std::max(timeout_ms, 1)) *
      static_cast<uint64_t>(clock_rate_khz);

  ncclUniqueId unique_id;
  std::memcpy(&unique_id, unique_id_bytes.data(), sizeof(unique_id));
  try {
    AWEX_NCCL_CHECK(ncclCommInitRank(
        &state->comm, world_size, unique_id, rank));

    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    AWEX_NCCL_CHECK(ncclCommQueryProperties(state->comm, &properties));
    if (!properties.deviceApiSupport) {
      throw std::runtime_error(
          "The loaded NCCL communicator does not support the Device API");
    }
    const ncclTeam_t lsa_team = ncclTeamLsa(state->comm);
    if (lsa_team.nRanks != world_size || lsa_team.rank != rank ||
        lsa_team.stride != 1) {
      throw std::runtime_error(
          "nccl_device requires the transfer communicator to map directly "
          "onto one contiguous LSA domain");
    }

    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    const char* multicast_env = std::getenv("AWEX_NCCL_DEVICE_MULTICAST");
    const bool multicast_requested = multicast_env == nullptr ||
        (std::strcmp(multicast_env, "0") != 0 &&
         std::strcmp(multicast_env, "false") != 0 &&
         std::strcmp(multicast_env, "False") != 0);
    requirements.lsaMultimem = multicast_requested && properties.multimemSupport;
    size_t dev_comm_bytes = 0;
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    requirements.useRuntimeVersion = true;
    dev_comm_bytes = properties.devCommRuntimeVersionSize;
#else
    dev_comm_bytes = sizeof(ncclDevComm_t);
#endif
    state->dev_comm = reinterpret_cast<ncclDevComm_t*>(std::malloc(dev_comm_bytes));
    if (state->dev_comm == nullptr) {
      throw std::bad_alloc();
    }
    AWEX_NCCL_CHECK(ncclDevCommCreate(
        state->comm, &requirements, state->dev_comm));

    const RingLayout ring = make_ring_layout(data_bytes, world_size);
    state->logical_data_bytes = data_bytes;
    state->ring_slot_bytes = ring.slot_bytes;
    state->ring_slots_per_peer = ring.slots_per_peer;
    state->ring_data_offset = ring.data_offset;
    state->window_bytes = ring.window_bytes;
    AWEX_NCCL_CHECK(ncclMemAlloc(&state->local_base, state->window_bytes));
    AWEX_NCCL_CHECK(ncclCommWindowRegister(
        state->comm,
        state->local_base,
        state->window_bytes,
        &state->window,
        NCCL_WIN_COLL_SYMMETRIC));
    if (requirements.lsaMultimem) {
      AWEX_NCCL_CHECK(ncclGetLsaMultimemDevicePointer(
          state->window, 0, &state->multimem_base));
    }
    AWEX_CUDA_CHECK(cudaMemset(
        state->local_base, 0, state->ring_data_offset));

    state->remote_bases.resize(world_size, nullptr);
    state->remote_bases[rank] = state->local_base;
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) {
        continue;
      }
      AWEX_NCCL_CHECK(ncclGetLsaDevicePointer(
          state->window, 0, peer, &state->remote_bases[peer]));
    }
  } catch (...) {
    if (state->window != nullptr) {
      ncclCommWindowDeregister(state->comm, state->window);
    }
    if (state->local_base != nullptr) {
      ncclMemFree(state->local_base);
    }
    if (state->dev_comm != nullptr) {
      ncclDevCommDestroy(state->comm, state->dev_comm);
      std::free(state->dev_comm);
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
  AWEX_CUDA_CHECK(cudaSetDevice(state->device));
  release_launch_resources(state);
  if (state->window != nullptr) {
    AWEX_NCCL_CHECK(ncclCommWindowDeregister(state->comm, state->window));
    state->window = nullptr;
  }
  if (state->local_base != nullptr) {
    AWEX_NCCL_CHECK(ncclMemFree(state->local_base));
    state->local_base = nullptr;
  }
  if (state->dev_comm != nullptr) {
    AWEX_NCCL_CHECK(ncclDevCommDestroy(state->comm, state->dev_comm));
    std::free(state->dev_comm);
    state->dev_comm = nullptr;
  }
  if (state->comm != nullptr) {
    AWEX_NCCL_CHECK(ncclCommAbort(state->comm));
    state->comm = nullptr;
  }
}

struct RingLaunchDescriptor {
  std::vector<dt::DeviceTask> tasks;
  std::vector<uint32_t> expected_counts;
  std::vector<uint32_t> peer_offsets;
  std::vector<uint32_t> active_peers;
  std::vector<uintptr_t> target_bases;
  uint64_t injected_payload_bytes = 0;
  uint32_t multicast_group_count = 0;
  uint32_t multicast_segment_count = 0;
};

RingLaunchDescriptor build_ring_launch_descriptor(
    const DeviceState& state,
    const py::list& tensors,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& lengths,
    const std::vector<int64_t>& peers,
    const std::vector<int64_t>& ordinals,
    const std::vector<int64_t>& expected_counts,
    const std::vector<std::vector<int64_t>>& multicast_groups,
    bool sender) {
  if (tensors.size() != offsets.size() || tensors.size() != lengths.size() ||
      tensors.size() != peers.size() || tensors.size() != ordinals.size()) {
    throw std::runtime_error("Tensor/task descriptor lengths do not match");
  }
  if (expected_counts.size() != static_cast<size_t>(state.world_size)) {
    throw std::runtime_error("Expected-count vector does not match world_size");
  }
  if (!multicast_groups.empty() &&
      multicast_groups.size() != static_cast<size_t>(state.world_size)) {
    throw std::runtime_error("Multicast-group vector does not match world_size");
  }

  std::vector<uint32_t> original_expected_counts;
  original_expected_counts.reserve(expected_counts.size());
  for (const auto count : expected_counts) {
    if (count < 0 || static_cast<uint64_t>(count) > UINT32_MAX) {
      throw std::runtime_error("nccl_device expected count is invalid");
    }
    original_expected_counts.push_back(static_cast<uint32_t>(count));
  }

  struct HostTensorTask {
    uintptr_t tensor_ptr = 0;
    uint64_t nbytes = 0;
    bool present = false;
  };

  std::vector<uint32_t> actual_counts(state.world_size, 0);
  std::vector<std::vector<HostTensorTask>> ordered_tasks(state.world_size);
  for (int peer = 0; peer < state.world_size; ++peer) {
    ordered_tasks[peer].resize(original_expected_counts[peer]);
  }

  for (size_t index = 0; index < tensors.size(); ++index) {
    const auto tensor = tensors[index].cast<torch::Tensor>();
    if (!tensor.is_cuda() || !tensor.is_contiguous()) {
      throw std::runtime_error(
          "nccl_device kernel inputs must be contiguous CUDA tensors");
    }
    if (tensor.get_device() != state.device) {
      throw std::runtime_error(
          "nccl_device tensor is on a different CUDA device");
    }

    const int64_t peer = peers[index];
    const int64_t ordinal = ordinals[index];
    if (peer < 0 || peer >= state.world_size || peer == state.rank) {
      throw std::runtime_error("nccl_device task peer is invalid");
    }
    if (ordinal < 0 ||
        static_cast<uint64_t>(ordinal) >= original_expected_counts[peer]) {
      throw std::runtime_error("nccl_device task ordinal is invalid");
    }
    if (offsets[index] < 0 || lengths[index] < 0) {
      throw std::runtime_error("nccl_device task byte range is invalid");
    }

    const auto logical_offset = static_cast<uint64_t>(offsets[index]);
    const auto task_bytes = static_cast<uint64_t>(lengths[index]);
    if (logical_offset > state.logical_data_bytes ||
        task_bytes > state.logical_data_bytes - logical_offset) {
      throw std::runtime_error(
          "nccl_device task exceeds the logical transfer layout");
    }
    const auto tensor_bytes = static_cast<uint64_t>(tensor.numel()) *
        static_cast<uint64_t>(tensor.element_size());
    if (task_bytes > tensor_bytes) {
      throw std::runtime_error(
          "nccl_device task length exceeds its tensor storage");
    }

    auto& ordered_task = ordered_tasks[peer][ordinal];
    if (ordered_task.present) {
      throw std::runtime_error("nccl_device task ordinals must be unique");
    }
    ordered_task = HostTensorTask{
        reinterpret_cast<uintptr_t>(tensor.data_ptr()), task_bytes, true};
    ++actual_counts[peer];
  }
  if (actual_counts != original_expected_counts) {
    throw std::runtime_error(
        "nccl_device task count does not match expected peer counts");
  }

  RingLaunchDescriptor descriptor;
  descriptor.expected_counts.resize(state.world_size, 0);
  descriptor.peer_offsets.resize(state.world_size + 1, 0);
  std::vector<bool> suppressed_peer(state.world_size, false);
  const bool use_multicast =
      sender && state.multimem_base != nullptr && !multicast_groups.empty();

  if (use_multicast) {
    for (int canonical_peer = 0; canonical_peer < state.world_size;
         ++canonical_peer) {
      const auto& targets = multicast_groups[canonical_peer];
      if (targets.empty()) {
        continue;
      }
      if (targets.size() < 2 || targets.front() != canonical_peer) {
        throw std::runtime_error(
            "nccl_device multicast group must start with its canonical peer");
      }
      const auto& canonical_tasks = ordered_tasks[canonical_peer];
      if (canonical_tasks.empty()) {
        throw std::runtime_error(
            "nccl_device multicast group has no canonical tasks");
      }
      int64_t previous_target = -1;
      for (const int64_t target : targets) {
        if (target < 0 || target >= state.world_size || target == state.rank ||
            target <= previous_target) {
          throw std::runtime_error(
              "nccl_device multicast targets must be sorted, unique peers");
        }
        previous_target = target;
        const auto& target_tasks = ordered_tasks[target];
        if (target_tasks.size() != canonical_tasks.size()) {
          throw std::runtime_error(
              "nccl_device multicast target task count differs from canonical");
        }
        for (size_t task = 0; task < canonical_tasks.size(); ++task) {
          if (target_tasks[task].tensor_ptr != canonical_tasks[task].tensor_ptr ||
              target_tasks[task].nbytes != canonical_tasks[task].nbytes) {
            throw std::runtime_error(
                "nccl_device multicast target task stream is not identical");
          }
        }
        if (target != canonical_peer) {
          if (suppressed_peer[target]) {
            throw std::runtime_error(
                "nccl_device peer belongs to multiple multicast groups");
          }
          suppressed_peer[target] = true;
        }
      }
      ++descriptor.multicast_group_count;
    }
  }

  for (int peer = 0; peer < state.world_size; ++peer) {
    descriptor.peer_offsets[peer] =
        static_cast<uint32_t>(descriptor.tasks.size());
    if (suppressed_peer[peer]) {
      descriptor.peer_offsets[peer + 1] =
          static_cast<uint32_t>(descriptor.tasks.size());
      continue;
    }

    const bool peer_multicast =
        use_multicast && !multicast_groups[peer].empty();
    const auto target_begin = static_cast<uint32_t>(descriptor.target_bases.size());
    if (peer_multicast) {
      for (const int64_t target : multicast_groups[peer]) {
        descriptor.target_bases.push_back(
            reinterpret_cast<uintptr_t>(state.remote_bases[target]));
      }
    } else if (!ordered_tasks[peer].empty()) {
      descriptor.target_bases.push_back(
          reinterpret_cast<uintptr_t>(state.remote_bases[peer]));
    }
    const auto target_count = static_cast<uint32_t>(
        descriptor.target_bases.size() - target_begin);
    uint64_t segment_ordinal = 0;
    for (const auto& task : ordered_tasks[peer]) {
      for (uint64_t offset = 0; offset < task.nbytes;
           offset += state.ring_slot_bytes) {
        if (segment_ordinal >= UINT32_MAX) {
          throw std::runtime_error(
              "nccl_device peer has too many ring segments");
        }
        const auto segment_bytes = std::min<uint64_t>(
            state.ring_slot_bytes, task.nbytes - offset);
        descriptor.tasks.push_back(dt::DeviceTask{
            task.tensor_ptr + static_cast<uintptr_t>(offset),
            reinterpret_cast<uintptr_t>(
                peer_multicast ? state.multimem_base : state.remote_bases[peer]),
            segment_bytes,
            static_cast<uint32_t>(peer),
            static_cast<uint32_t>(segment_ordinal),
            target_begin,
            target_count,
            peer_multicast ? dt::kTaskMulticast : 0U,
        });
        descriptor.injected_payload_bytes += segment_bytes;
        if (peer_multicast) {
          ++descriptor.multicast_segment_count;
        }
        ++segment_ordinal;
      }
    }

    descriptor.expected_counts[peer] =
        static_cast<uint32_t>(segment_ordinal);
    if (segment_ordinal != 0) {
      descriptor.active_peers.push_back(static_cast<uint32_t>(peer));
    }
    if (descriptor.tasks.size() > UINT32_MAX) {
      throw std::runtime_error("nccl_device launch has too many ring segments");
    }
    descriptor.peer_offsets[peer + 1] =
        static_cast<uint32_t>(descriptor.tasks.size());
  }
  return descriptor;
}

py::dict launch(
    int64_t handle,
    const py::list& tensors,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& lengths,
    const std::vector<int64_t>& peers,
    const std::vector<int64_t>& ordinals,
    const std::vector<int64_t>& expected_counts,
    const std::vector<std::vector<int64_t>>& multicast_groups,
    bool sender,
    int64_t sequence) {
  using Clock = std::chrono::steady_clock;
  const auto launch_start = Clock::now();
  // Fair backend benchmarks use the common host-side timer without adding
  // device-only events and profile-buffer copies. Preserve the historical
  // AWEX_PROFILE behavior unless an explicit detail override is provided.
  const char* profile_env = std::getenv("AWEX_PROFILE_DEVICE_DETAILS");
  if (profile_env == nullptr) {
    profile_env = std::getenv("AWEX_PROFILE");
  }
  const bool collect_profile = profile_env != nullptr &&
      (std::strcmp(profile_env, "1") == 0 ||
       std::strcmp(profile_env, "true") == 0 ||
       std::strcmp(profile_env, "True") == 0);
  py::dict metrics;
  auto* state = reinterpret_cast<DeviceState*>(handle);
  if (state == nullptr) {
    throw std::runtime_error("Invalid nccl_device state handle");
  }
  if (sequence <= 0 || static_cast<uint64_t>(sequence) > UINT32_MAX) {
    throw std::runtime_error(
        "nccl_device sequence must fit in a positive 32-bit value");
  }
  if (static_cast<uint64_t>(sequence) <= state->last_sequence) {
    throw std::runtime_error(
        "nccl_device sequence must increase between ring launches");
  }
  const auto descriptor = build_ring_launch_descriptor(
      *state,
      tensors,
      offsets,
      lengths,
      peers,
      ordinals,
      expected_counts,
      multicast_groups,
      sender);
  metrics["ring_segment_count"] = py::int_(descriptor.tasks.size());
  metrics["ring_slot_bytes"] = py::int_(state->ring_slot_bytes);
  metrics["ring_slots_per_peer"] = py::int_(state->ring_slots_per_peer);
  metrics["registered_window_bytes"] = py::int_(state->window_bytes);
  metrics["multimem_available"] = py::bool_(state->multimem_base != nullptr);
  metrics["multicast_group_count"] = py::int_(
      descriptor.multicast_group_count);
  metrics["multicast_segment_count"] = py::int_(
      descriptor.multicast_segment_count);
  metrics["injected_payload_bytes"] = py::int_(
      descriptor.injected_payload_bytes);
  metrics["host_descriptor_time_ms"] =
      std::chrono::duration<double, std::milli>(Clock::now() - launch_start)
          .count();

  AWEX_CUDA_CHECK(cudaSetDevice(state->device));
  auto stream = at::cuda::getCurrentCUDAStream(state->device).stream();
  auto* local_control =
      reinterpret_cast<dt::ControlBlock*>(state->local_base);

  const auto allocation_start = Clock::now();
  ensure_launch_resources(
      state,
      descriptor.tasks.size(),
      descriptor.expected_counts.size(),
      descriptor.target_bases.size(),
      collect_profile);
  metrics["buffer_allocation_time_ms"] =
      std::chrono::duration<double, std::milli>(Clock::now() - allocation_start)
          .count();

  auto* device_tasks = state->device_tasks;
  auto* device_expected_counts = state->device_expected_counts;
  auto* device_peer_offsets = state->device_peer_offsets;
  auto* device_active_peers = state->device_active_peers;
  auto* device_target_bases = state->device_target_bases;
  auto* device_profile = collect_profile ? state->device_profile : nullptr;
  const auto metadata_start =
      collect_profile ? state->metadata_start : nullptr;
  const auto metadata_end = collect_profile ? state->metadata_end : nullptr;
  const auto kernel_end = collect_profile ? state->kernel_end : nullptr;

  dt::DeviceProfile initial_profile{};
  initial_profile.first_ready_ns = ULLONG_MAX;
  initial_profile.first_copy_ns = ULLONG_MAX;
  initial_profile.publish_start_ns = ULLONG_MAX;
  if (collect_profile) {
    AWEX_CUDA_CHECK(cudaEventRecord(metadata_start, stream));
  }
  AWEX_CUDA_CHECK(cudaMemsetAsync(
      &local_control->error, 0, sizeof(local_control->error), stream));
  if (!descriptor.tasks.empty()) {
    AWEX_CUDA_CHECK(cudaMemcpyAsync(
        device_tasks,
        descriptor.tasks.data(),
        descriptor.tasks.size() * sizeof(dt::DeviceTask),
        cudaMemcpyHostToDevice,
        stream));
  }
  AWEX_CUDA_CHECK(cudaMemcpyAsync(
      device_expected_counts,
      descriptor.expected_counts.data(),
      descriptor.expected_counts.size() * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream));
  AWEX_CUDA_CHECK(cudaMemcpyAsync(
      device_peer_offsets,
      descriptor.peer_offsets.data(),
      descriptor.peer_offsets.size() * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream));
  if (!descriptor.active_peers.empty()) {
    AWEX_CUDA_CHECK(cudaMemcpyAsync(
        device_active_peers,
        descriptor.active_peers.data(),
        descriptor.active_peers.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice,
        stream));
  }
  if (!descriptor.target_bases.empty()) {
    AWEX_CUDA_CHECK(cudaMemcpyAsync(
        device_target_bases,
        descriptor.target_bases.data(),
        descriptor.target_bases.size() * sizeof(uintptr_t),
        cudaMemcpyHostToDevice,
        stream));
  }
  if (device_profile != nullptr) {
    AWEX_CUDA_CHECK(cudaMemcpyAsync(
        device_profile,
        &initial_profile,
        sizeof(dt::DeviceProfile),
        cudaMemcpyHostToDevice,
        stream));
  }
  if (collect_profile) {
    AWEX_CUDA_CHECK(cudaEventRecord(metadata_end, stream));
  }

  const unsigned int blocks = static_cast<unsigned int>(std::max<size_t>(
      1, descriptor.active_peers.size() * state->ring_slots_per_peer));
  state->last_sequence = static_cast<uint64_t>(sequence);
  const dt::DeviceTransferArgs kernel_args{
      device_tasks,
      static_cast<uint32_t>(descriptor.tasks.size()),
      sender ? dt::TransferRole::kSender : dt::TransferRole::kReceiver,
      static_cast<unsigned long long>(sequence),
      state->rank,
      reinterpret_cast<uint8_t*>(state->local_base),
      device_expected_counts,
      device_peer_offsets,
      device_active_peers,
      device_target_bases,
      static_cast<uint32_t>(descriptor.active_peers.size()),
      state->ring_slots_per_peer,
      state->ring_slot_bytes,
      state->ring_data_offset,
      device_profile,
      state->timeout_cycles,
  };
  dt::transfer_impl<<<blocks, dt::kThreadsPerBlock, 0, stream>>>(kernel_args);
  AWEX_CUDA_CHECK(cudaGetLastError());
  if (collect_profile) {
    AWEX_CUDA_CHECK(cudaEventRecord(kernel_end, stream));
  }
  AWEX_CUDA_CHECK(cudaStreamSynchronize(stream));

  dt::DeviceProfile host_profile{};
  if (collect_profile) {
    float metadata_upload_time_ms = 0.0F;
    float kernel_transfer_time_ms = 0.0F;
    AWEX_CUDA_CHECK(cudaEventElapsedTime(
        &metadata_upload_time_ms, metadata_start, metadata_end));
    AWEX_CUDA_CHECK(cudaEventElapsedTime(
        &kernel_transfer_time_ms, metadata_end, kernel_end));
    metrics["metadata_upload_time_ms"] = metadata_upload_time_ms;
    metrics["kernel_transfer_time_ms"] = kernel_transfer_time_ms;
    AWEX_CUDA_CHECK(cudaMemcpy(
        &host_profile,
        device_profile,
        sizeof(dt::DeviceProfile),
        cudaMemcpyDeviceToHost));
  }

  dt::ControlBlock host_control;
  const auto control_download_start = Clock::now();
  AWEX_CUDA_CHECK(cudaMemcpy(
      &host_control,
      local_control,
      sizeof(host_control),
      cudaMemcpyDeviceToHost));
  metrics["control_download_time_ms"] =
      std::chrono::duration<double, std::milli>(
          Clock::now() - control_download_start)
          .count();

  metrics["buffer_cleanup_time_ms"] = 0.0;

  auto device_delta_ms = [](unsigned long long start,
                            unsigned long long end) -> double {
    if (start == 0 || start == ULLONG_MAX || end < start) {
      return 0.0;
    }
    return static_cast<double>(end - start) / 1.0e6;
  };
  if (collect_profile) {
    metrics["device_copy_span_time_ms"] = device_delta_ms(
        host_profile.first_copy_ns, host_profile.last_copy_ns);
    if (sender) {
      metrics["sender_publish_time_ms"] = device_delta_ms(
          host_profile.publish_start_ns, host_profile.publish_done_ns);
      metrics["sender_reader_ack_wait_time_ms"] = device_delta_ms(
          host_profile.publish_done_ns, host_profile.peer_done_ns);
    } else {
      metrics["reader_first_ready_time_ms"] = device_delta_ms(
          host_profile.kernel_start_ns, host_profile.first_ready_ns);
      metrics["reader_wait_time_ms"] = device_delta_ms(
          host_profile.kernel_start_ns, host_profile.last_ready_ns);
      metrics["reader_copyback_time_ms"] = device_delta_ms(
          host_profile.first_copy_ns, host_profile.last_copy_ns);
      metrics["reader_copyback_tail_time_ms"] = device_delta_ms(
          host_profile.last_ready_ns, host_profile.last_copy_ns);
    }
  }
  metrics["extension_total_time_ms"] =
      std::chrono::duration<double, std::milli>(Clock::now() - launch_start)
          .count();
  if (host_control.error != 0) {
    throw std::runtime_error(
        "nccl_device kernel aborted while waiting for the peer");
  }
  return metrics;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("unique_id_size", []() { return sizeof(ncclUniqueId); });
  module.def("get_unique_id", []() {
    ncclUniqueId unique_id;
    AWEX_NCCL_CHECK(ncclGetUniqueId(&unique_id));
    return py::bytes(
        reinterpret_cast<const char*>(&unique_id), sizeof(unique_id));
  });
  module.def("create", [](const py::bytes& id,
                          int world_size,
                          int rank,
                          int64_t data_bytes,
                          int device,
                          int timeout_ms) {
    if (data_bytes < 0) {
      throw std::runtime_error(
          "nccl_device logical transfer size must be non-negative");
    }
    const std::string unique_id = id;
    auto state = make_state(
        unique_id,
        world_size,
        rank,
        static_cast<size_t>(data_bytes),
        device,
        timeout_ms);
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

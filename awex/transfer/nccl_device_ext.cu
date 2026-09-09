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

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kMaxRanks = 256;
constexpr uint64_t kVectorBytes = sizeof(uint4);
static_assert(
    kVectorBytes == 16, "nccl_device vector copy requires 16-byte uint4");

struct alignas(64) ControlBlock {
  unsigned long long ready_count[kMaxRanks];
  unsigned long long done_count[kMaxRanks];
  unsigned long long ready_base[kMaxRanks];
  unsigned long long done_base[kMaxRanks];
  unsigned long long epoch;
  unsigned int error;
};

struct DeviceTask {
  uintptr_t tensor_ptr;
  uintptr_t remote_base;
  uint64_t byte_offset;
  uint64_t nbytes;
  uint32_t peer;
  uint32_t ordinal;
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

struct DeviceState {
  ncclComm_t comm = nullptr;
  ncclDevComm_t* dev_comm = nullptr;
  ncclWindow_t window = nullptr;
  void* local_base = nullptr;
  // Cached NCCL LSA device pointers; the transfer kernel uses direct
  // load/store operations instead of issuing a host-side NCCL P2P operation.
  std::vector<void*> remote_bases;
  size_t window_bytes = 0;
  uint64_t timeout_cycles = 0;
  int rank = 0;
  int world_size = 0;
  int device = 0;
  DeviceTask* device_tasks = nullptr;
  uint32_t* device_expected_counts = nullptr;
  unsigned long long* device_task_done = nullptr;
  DeviceProfile* device_profile = nullptr;
  size_t task_capacity = 0;
  size_t expected_count_capacity = 0;
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

void ensure_launch_resources(
    DeviceState* state,
    size_t task_count,
    size_t expected_count_count,
    bool collect_profile) {
  if (task_count > state->task_capacity) {
    DeviceTask* new_tasks = nullptr;
    unsigned long long* new_task_done = nullptr;
    try {
      AWEX_CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&new_tasks),
          task_count * sizeof(DeviceTask)));
      AWEX_CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&new_task_done),
          task_count * sizeof(unsigned long long)));
    } catch (...) {
      if (new_tasks != nullptr) {
        cudaFree(new_tasks);
      }
      if (new_task_done != nullptr) {
        cudaFree(new_task_done);
      }
      throw;
    }
    if (state->device_tasks != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_tasks));
    }
    if (state->device_task_done != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_task_done));
    }
    state->device_tasks = new_tasks;
    state->device_task_done = new_task_done;
    state->task_capacity = task_count;
  }

  if (expected_count_count > state->expected_count_capacity) {
    uint32_t* new_expected_counts = nullptr;
    AWEX_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&new_expected_counts),
        expected_count_count * sizeof(uint32_t)));
    if (state->device_expected_counts != nullptr) {
      AWEX_CUDA_CHECK(cudaFree(state->device_expected_counts));
    }
    state->device_expected_counts = new_expected_counts;
    state->expected_count_capacity = expected_count_count;
  }

  if (collect_profile && state->device_profile == nullptr) {
    DeviceProfile* new_profile = nullptr;
    cudaEvent_t new_metadata_start = nullptr;
    cudaEvent_t new_metadata_end = nullptr;
    cudaEvent_t new_kernel_end = nullptr;
    try {
      AWEX_CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&new_profile), sizeof(DeviceProfile)));
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
  if (state->device_task_done != nullptr) {
    AWEX_CUDA_CHECK(cudaFree(state->device_task_done));
    state->device_task_done = nullptr;
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
}

__device__ __forceinline__ unsigned long long global_timer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ bool wait_for_value(
    volatile unsigned long long* address,
    unsigned long long expected,
    volatile unsigned int* error,
    unsigned long long timeout_cycles,
    bool monotonic = false) {
  const unsigned long long start = clock64();
  while (true) {
    const auto value = atomicAdd_system(
        const_cast<unsigned long long*>(address), 0ULL);
    if ((monotonic && value >= expected) || (!monotonic && value == expected)) {
      return true;
    }
    if (atomicAdd(const_cast<unsigned int*>(error), 0U) != 0U) {
      return false;
    }
    if (clock64() - start > timeout_cycles) {
      atomicExch_system(const_cast<unsigned int*>(error), 1U);
      return false;
    }
  }
}

__device__ __forceinline__ void copy_task_bytes(
    uint8_t* __restrict__ destination,
    const uint8_t* __restrict__ source,
    uint64_t nbytes) {
  const auto address_bits = reinterpret_cast<uintptr_t>(destination) |
      reinterpret_cast<uintptr_t>(source);
  if ((address_bits & (kVectorBytes - 1)) == 0) {
    auto* destination_vectors = reinterpret_cast<uint4*>(destination);
    const auto* source_vectors = reinterpret_cast<const uint4*>(source);
    const uint64_t vector_count = nbytes / kVectorBytes;
    for (uint64_t vector = threadIdx.x; vector < vector_count;
         vector += blockDim.x) {
      destination_vectors[vector] = source_vectors[vector];
    }

    const uint64_t tail_offset = vector_count * kVectorBytes;
    for (uint64_t byte = tail_offset + threadIdx.x; byte < nbytes;
         byte += blockDim.x) {
      destination[byte] = source[byte];
    }
    return;
  }

  for (uint64_t byte = threadIdx.x; byte < nbytes; byte += blockDim.x) {
    destination[byte] = source[byte];
  }
}

__global__ void awex_transfer_kernel(
    const DeviceTask* tasks,
    uint32_t task_count,
    bool sender,
    unsigned long long sequence,
    int local_rank,
    uint8_t* local_base,
    int world_size,
    const uint32_t* expected_counts,
    unsigned long long* task_done,
    DeviceProfile* profile,
    unsigned long long timeout_cycles) {
  auto* local = reinterpret_cast<ControlBlock*>(local_base);

  // Capture the current counter values once per update.  Counters are never
  // reset, so a new update can safely signal over a previous update's window.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (profile != nullptr) {
      profile->kernel_start_ns = global_timer_ns();
    }
    for (int peer = 0; peer < world_size; ++peer) {
      local->ready_base[peer] =
          atomicAdd_system(&local->ready_count[peer], 0ULL);
      local->done_base[peer] =
          atomicAdd_system(&local->done_count[peer], 0ULL);
    }
    __threadfence_system();
    atomicExch_system(&local->epoch, sequence);
  }

  __shared__ int initialized;
  if (threadIdx.x == 0) {
    initialized = wait_for_value(
        &local->epoch, sequence, &local->error, timeout_cycles);
    if (initialized) {
      for (uint32_t index = 0; index < task_count; ++index) {
        auto* remote = reinterpret_cast<ControlBlock*>(tasks[index].remote_base);
        if (!wait_for_value(
                &remote->epoch, sequence, &local->error, timeout_cycles)) {
          initialized = 0;
          break;
        }
      }
    }
  }
  __syncthreads();
  if (!initialized) {
    return;
  }

  for (uint32_t index = blockIdx.x; index < task_count; index += gridDim.x) {
    const DeviceTask task = tasks[index];
    uint8_t* local_data =
        local_base + sizeof(ControlBlock) + task.byte_offset;
    uint8_t* remote_data = reinterpret_cast<uint8_t*>(task.remote_base) +
        sizeof(ControlBlock) + task.byte_offset;
    uint8_t* source = reinterpret_cast<uint8_t*>(task.tensor_ptr);

    if (!sender) {
      if (threadIdx.x == 0) {
        const auto expected =
            local->ready_base[task.peer] + task.ordinal + 1ULL;
        initialized = wait_for_value(
            &local->ready_count[task.peer],
            expected,
            &local->error,
            timeout_cycles,
            true);
      }
      __syncthreads();
      if (!initialized) {
        return;
      }
      if (profile != nullptr) {
        const auto ready_ns = global_timer_ns();
        atomicMin(&profile->first_ready_ns, ready_ns);
        atomicMax(&profile->last_ready_ns, ready_ns);
      }
      __threadfence_system();
    }

    if (threadIdx.x == 0 && profile != nullptr) {
      atomicMin(&profile->first_copy_ns, global_timer_ns());
    }
    __syncthreads();
    if (sender) {
      copy_task_bytes(remote_data, source, task.nbytes);
    } else {
      copy_task_bytes(source, local_data, task.nbytes);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (profile != nullptr) {
        atomicMax(&profile->last_copy_ns, global_timer_ns());
      }
      if (sender) {
        __threadfence_system();
        atomicExch(&task_done[index], 1ULL);
      } else {
        auto* remote = reinterpret_cast<ControlBlock*>(task.remote_base);
        __threadfence_system();
        atomicAdd_system(&remote->done_count[local_rank], 1ULL);
      }
    }
    __syncthreads();
  }

  if (sender && blockIdx.x == 0 && threadIdx.x == 0) {
    // Blocks may finish copies out of order.  Publish ready counters in task
    // ordinal order so the receiver's counter wait identifies the right task.
    if (profile != nullptr) {
      profile->publish_start_ns = global_timer_ns();
    }
    for (uint32_t index = 0; index < task_count; ++index) {
      if (!wait_for_value(
              &task_done[index], 1ULL, &local->error, timeout_cycles)) {
        return;
      }
      auto* remote = reinterpret_cast<ControlBlock*>(tasks[index].remote_base);
      __threadfence_system();
      atomicAdd_system(&remote->ready_count[local_rank], 1ULL);
    }
    if (profile != nullptr) {
      profile->publish_done_ns = global_timer_ns();
    }
    for (int peer = 0; peer < world_size; ++peer) {
      const auto expected = local->done_base[peer] + expected_counts[peer];
      if (!wait_for_value(
              &local->done_count[peer],
              expected,
              &local->error,
              timeout_cycles,
              true)) {
        return;
      }
    }
    if (profile != nullptr) {
      profile->peer_done_ns = global_timer_ns();
    }
    __threadfence_system();
  }
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

    state->window_bytes =
        ((sizeof(ControlBlock) + data_bytes + 4095) / 4096) * 4096;
    AWEX_NCCL_CHECK(ncclMemAlloc(&state->local_base, state->window_bytes));
    AWEX_NCCL_CHECK(ncclCommWindowRegister(
        state->comm,
        state->local_base,
        state->window_bytes,
        &state->window,
        NCCL_WIN_COLL_SYMMETRIC));
    AWEX_CUDA_CHECK(cudaMemset(
        state->local_base, 0, sizeof(ControlBlock)));

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

py::dict launch(
    int64_t handle,
    const py::list& tensors,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& lengths,
    const std::vector<int64_t>& peers,
    const std::vector<int64_t>& ordinals,
    const std::vector<int64_t>& expected_counts,
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
  if (sequence <= 0) {
    throw std::runtime_error("nccl_device sequence must be positive");
  }
  if (tensors.size() != offsets.size() || tensors.size() != lengths.size() ||
      tensors.size() != peers.size() || tensors.size() != ordinals.size()) {
    throw std::runtime_error("Tensor/task descriptor lengths do not match");
  }
  if (expected_counts.size() != static_cast<size_t>(state->world_size)) {
    throw std::runtime_error("Expected-count vector does not match world_size");
  }

  std::vector<uint32_t> host_expected_counts;
  host_expected_counts.reserve(expected_counts.size());
  for (const auto count : expected_counts) {
    if (count < 0 || static_cast<uint64_t>(count) > UINT32_MAX) {
      throw std::runtime_error("nccl_device expected count is invalid");
    }
    host_expected_counts.push_back(static_cast<uint32_t>(count));
  }

  std::vector<DeviceTask> host_tasks;
  host_tasks.reserve(tensors.size());
  std::vector<uint32_t> actual_counts(state->world_size, 0);
  std::vector<std::vector<uint8_t>> seen_ordinals(state->world_size);
  for (int peer = 0; peer < state->world_size; ++peer) {
    seen_ordinals[peer].resize(host_expected_counts[peer], 0);
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto tensor = tensors[i].cast<torch::Tensor>();
    if (!tensor.is_cuda() || !tensor.is_contiguous()) {
      throw std::runtime_error(
          "nccl_device kernel inputs must be contiguous CUDA tensors");
    }
    if (peers[i] < 0 || peers[i] >= state->world_size || peers[i] == state->rank) {
      throw std::runtime_error("nccl_device task peer is invalid");
    }
    if (ordinals[i] < 0) {
      throw std::runtime_error("nccl_device task ordinal must be non-negative");
    }
    if (offsets[i] < 0 || lengths[i] < 0 ||
        static_cast<size_t>(offsets[i]) + static_cast<size_t>(lengths[i]) >
            state->window_bytes - sizeof(ControlBlock)) {
      throw std::runtime_error("nccl_device task exceeds registered window");
    }
    if (static_cast<size_t>(ordinals[i]) >=
        static_cast<size_t>(host_expected_counts[peers[i]])) {
      throw std::runtime_error("nccl_device task ordinal exceeds expected count");
    }
    if (seen_ordinals[peers[i]][ordinals[i]] != 0) {
      throw std::runtime_error("nccl_device task ordinals must be unique");
    }
    seen_ordinals[peers[i]][ordinals[i]] = 1;
    actual_counts[peers[i]] += 1;
    host_tasks.push_back(DeviceTask{
        reinterpret_cast<uintptr_t>(tensor.data_ptr()),
        reinterpret_cast<uintptr_t>(state->remote_bases[peers[i]]),
        static_cast<uint64_t>(offsets[i]),
        static_cast<uint64_t>(lengths[i]),
        static_cast<uint32_t>(peers[i]),
        static_cast<uint32_t>(ordinals[i]),
    });
  }
  if (actual_counts != host_expected_counts) {
    throw std::runtime_error(
        "nccl_device task count does not match expected peer counts");
  }
  metrics["host_descriptor_time_ms"] =
      std::chrono::duration<double, std::milli>(Clock::now() - launch_start)
          .count();

  AWEX_CUDA_CHECK(cudaSetDevice(state->device));
  auto stream = at::cuda::getCurrentCUDAStream(state->device).stream();
  auto* local_control = reinterpret_cast<ControlBlock*>(state->local_base);

  const auto allocation_start = Clock::now();
  ensure_launch_resources(
      state,
      host_tasks.size(),
      host_expected_counts.size(),
      collect_profile);
  metrics["buffer_allocation_time_ms"] =
      std::chrono::duration<double, std::milli>(Clock::now() - allocation_start)
          .count();

  auto* device_tasks = state->device_tasks;
  auto* device_expected_counts = state->device_expected_counts;
  auto* device_task_done = state->device_task_done;
  auto* device_profile = collect_profile ? state->device_profile : nullptr;
  const auto metadata_start =
      collect_profile ? state->metadata_start : nullptr;
  const auto metadata_end = collect_profile ? state->metadata_end : nullptr;
  const auto kernel_end = collect_profile ? state->kernel_end : nullptr;

  DeviceProfile initial_profile{};
  initial_profile.first_ready_ns = ULLONG_MAX;
  initial_profile.first_copy_ns = ULLONG_MAX;
  if (collect_profile) {
    AWEX_CUDA_CHECK(cudaEventRecord(metadata_start, stream));
  }
  AWEX_CUDA_CHECK(cudaMemsetAsync(
      &local_control->error, 0, sizeof(local_control->error), stream));
  if (!host_tasks.empty()) {
    AWEX_CUDA_CHECK(cudaMemcpyAsync(
        device_tasks,
        host_tasks.data(),
        host_tasks.size() * sizeof(DeviceTask),
        cudaMemcpyHostToDevice,
        stream));
  }
  AWEX_CUDA_CHECK(cudaMemcpyAsync(
      device_expected_counts,
      host_expected_counts.data(),
      host_expected_counts.size() * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream));
  if (!host_tasks.empty()) {
    AWEX_CUDA_CHECK(cudaMemsetAsync(
        device_task_done,
        0,
        host_tasks.size() * sizeof(unsigned long long),
        stream));
  }
  if (device_profile != nullptr) {
    AWEX_CUDA_CHECK(cudaMemcpyAsync(
        device_profile,
        &initial_profile,
        sizeof(DeviceProfile),
        cudaMemcpyHostToDevice,
        stream));
  }
  if (collect_profile) {
    AWEX_CUDA_CHECK(cudaEventRecord(metadata_end, stream));
  }

  const unsigned int blocks = static_cast<unsigned int>(
      std::max<size_t>(1, std::min<size_t>(host_tasks.size(), 1024)));
  awex_transfer_kernel<<<blocks, 256, 0, stream>>>(
      device_tasks,
      static_cast<uint32_t>(host_tasks.size()),
      sender,
      static_cast<unsigned long long>(sequence),
      state->rank,
      reinterpret_cast<uint8_t*>(state->local_base),
      state->world_size,
      device_expected_counts,
      device_task_done,
      device_profile,
      state->timeout_cycles);
  AWEX_CUDA_CHECK(cudaGetLastError());
  if (collect_profile) {
    AWEX_CUDA_CHECK(cudaEventRecord(kernel_end, stream));
  }
  AWEX_CUDA_CHECK(cudaStreamSynchronize(stream));

  DeviceProfile host_profile{};
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
        sizeof(DeviceProfile),
        cudaMemcpyDeviceToHost));
  }

  ControlBlock host_control;
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

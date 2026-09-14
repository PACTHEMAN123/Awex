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

#include "device_v2_types.cuh"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace awex {
namespace nccl_device_v2 {

struct V2PeerPath {
  std::uint32_t nvlink_count = 0;
  std::uint32_t raw_channels = 2;
  std::uint32_t channels = 2;
  float bandwidth_gbps = 0.0F;
};

struct V2Topology {
  bool nvml_available = false;
  std::uint32_t total_channels = 1;
  std::uint32_t requested_channels_per_peer = 1;
  std::uint32_t channels_per_peer = 1;
  std::vector<std::uint32_t> peer_channels;
  std::vector<V2PeerPath> peer_paths;
};

namespace topology_detail {

inline void checkCuda(cudaError_t result, const char* expression) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed: " + cudaGetErrorString(result));
  }
}

inline void checkNccl(ncclResult_t result, const char* expression) {
  if (result != ncclSuccess) {
    throw std::runtime_error(std::string(expression) + " failed: " + ncclGetErrorString(result));
  }
}

inline std::uint32_t powerOfTwoDown(std::uint32_t value) {
  std::uint32_t result = 1;
  while (result <= value / 2) result *= 2;
  return result;
}

inline std::uint32_t powerOfTwoUp(std::uint32_t value) {
  std::uint32_t result = 1;
  while (result < value && result < kMaxChannels) result *= 2;
  return result;
}

inline float nvlinkBandwidth(int compute_capability) {
  // Keep these synchronized with NCCL graph/topo.h.
  if (compute_capability >= 100) return 40.1F;
  if (compute_capability >= 90) return 20.6F;
  if (compute_capability == 86) return 12.0F;
  if (compute_capability >= 80) return 20.0F;
  if (compute_capability >= 70) return 20.0F;
  return 18.0F;
}

inline std::uint64_t encodePciBus(unsigned int domain, unsigned int bus, unsigned int device) {
  return (static_cast<std::uint64_t>(domain) << 32) | (static_cast<std::uint64_t>(bus) << 16) |
         static_cast<std::uint64_t>(device);
}

inline std::uint64_t parsePciBus(const char* bus_id) {
  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
  if (std::sscanf(bus_id, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
    throw std::runtime_error(std::string("cannot parse CUDA PCI bus id: ") + bus_id);
  }
  return encodePciBus(domain, bus, device);
}

struct alignas(16) TopologyRecord {
  std::uint64_t pci_bus;
  std::uint32_t compute_capability;
  std::uint32_t switch_links;
  std::uint32_t nvml_available;
  std::uint32_t reserved;
};

template <typename T>
inline std::vector<T> allGather(ncclComm_t comm, const T* local, std::size_t count, int world_size,
                                cudaStream_t stream) {
  T* device_send = nullptr;
  T* device_recv = nullptr;
  const std::size_t send_bytes = count * sizeof(T);
  const std::size_t recv_bytes = send_bytes * world_size;
  try {
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_send), send_bytes), "cudaMalloc(topology send)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_recv), recv_bytes), "cudaMalloc(topology recv)");
    checkCuda(cudaMemcpyAsync(device_send, local, send_bytes, cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(topology send)");
    checkNccl(ncclAllGather(device_send, device_recv, send_bytes, ncclUint8, comm, stream), "ncclAllGather(topology)");
    std::vector<T> gathered(count * world_size);
    checkCuda(cudaMemcpyAsync(gathered.data(), device_recv, recv_bytes, cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(topology recv)");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(topology)");
    checkCuda(cudaFree(device_recv), "cudaFree(topology recv)");
    checkCuda(cudaFree(device_send), "cudaFree(topology send)");
    return gathered;
  } catch (...) {
    if (device_recv != nullptr) cudaFree(device_recv);
    if (device_send != nullptr) cudaFree(device_send);
    throw;
  }
}

#if defined(__linux__)

using NvmlDevice = struct nvmlDevice_st*;
using NvmlReturn = int;
constexpr NvmlReturn kNvmlSuccess = 0;
constexpr int kNvmlFeatureEnabled = 1;
constexpr int kNvmlLinkP2pSupported = 0;
constexpr int kNvmlRemoteGpu = 0;
constexpr int kNvmlRemoteSwitch = 2;
constexpr unsigned int kNvmlFieldLinkState = 165;

struct NvmlPciInfo {
  char bus_id_legacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pci_device_id;
  unsigned int pci_subsystem_id;
  char bus_id[32];
};

union NvmlValue {
  double d;
  unsigned int ui;
  unsigned long ul;
  unsigned long long ull;
  long long sll;
};

struct NvmlFieldValue {
  unsigned int field_id;
  unsigned int scope_id;
  long long timestamp;
  long long latency_usec;
  int value_type;
  NvmlReturn result;
  NvmlValue value;
};

class NvmlApi {
public:
  NvmlApi() {
    library_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (library_ == nullptr) return;
    init_ = load<InitFn>("nvmlInit_v2");
    shutdown_ = load<ShutdownFn>("nvmlShutdown");
    get_handle_ = load<GetHandleFn>("nvmlDeviceGetHandleByPciBusId_v2");
    if (get_handle_ == nullptr) get_handle_ = load<GetHandleFn>("nvmlDeviceGetHandleByPciBusId");
    get_capability_ = load<GetCapabilityFn>("nvmlDeviceGetNvLinkCapability");
    get_state_ = load<GetStateFn>("nvmlDeviceGetNvLinkState");
    get_remote_type_ = load<GetRemoteTypeFn>("nvmlDeviceGetNvLinkRemoteDeviceType");
    get_remote_pci_ = load<GetRemotePciFn>("nvmlDeviceGetNvLinkRemotePciInfo_v2");
    if (get_remote_pci_ == nullptr) get_remote_pci_ = load<GetRemotePciFn>("nvmlDeviceGetNvLinkRemotePciInfo");
    get_fields_ = load<GetFieldsFn>("nvmlDeviceGetFieldValues");
    available_ = init_ != nullptr && get_handle_ != nullptr && get_capability_ != nullptr &&
                 get_remote_type_ != nullptr && (get_state_ != nullptr || get_fields_ != nullptr);
    if (available_) available_ = init_() == kNvmlSuccess;
  }

  ~NvmlApi() {
    if (available_ && shutdown_ != nullptr) shutdown_();
    if (library_ != nullptr) dlclose(library_);
  }

  NvmlApi(const NvmlApi&) = delete;
  NvmlApi& operator=(const NvmlApi&) = delete;

  bool available() const {
    return available_;
  }

  NvmlDevice device(const char* pci_bus_id) const {
    NvmlDevice result = nullptr;
    return available_ && get_handle_(pci_bus_id, &result) == kNvmlSuccess ? result : nullptr;
  }

  bool activeP2pLink(NvmlDevice device, unsigned int link) const {
    unsigned int p2p = 0;
    if (get_capability_(device, link, kNvmlLinkP2pSupported, &p2p) != kNvmlSuccess || p2p == 0) return false;
    if (get_fields_ != nullptr) {
      NvmlFieldValue value{};
      value.field_id = kNvmlFieldLinkState;
      value.scope_id = link;
      if (get_fields_(device, 1, &value) == kNvmlSuccess && value.result == kNvmlSuccess) {
        return value.value.ui == kNvmlFeatureEnabled;
      }
    }
    int state = 0;
    return get_state_ != nullptr && get_state_(device, link, &state) == kNvmlSuccess && state == kNvmlFeatureEnabled;
  }

  int remoteType(NvmlDevice device, unsigned int link) const {
    int type = -1;
    return get_remote_type_(device, link, &type) == kNvmlSuccess ? type : -1;
  }

  std::uint64_t remotePci(NvmlDevice device, unsigned int link) const {
    if (get_remote_pci_ == nullptr) return 0;
    NvmlPciInfo pci{};
    if (get_remote_pci_(device, link, &pci) != kNvmlSuccess) return 0;
    return encodePciBus(pci.domain, pci.bus, pci.device);
  }

private:
  template <typename T>
  T load(const char* name) {
    return reinterpret_cast<T>(dlsym(library_, name));
  }

  using InitFn = NvmlReturn (*)();
  using ShutdownFn = NvmlReturn (*)();
  using GetHandleFn = NvmlReturn (*)(const char*, NvmlDevice*);
  using GetCapabilityFn = NvmlReturn (*)(NvmlDevice, unsigned int, int, unsigned int*);
  using GetStateFn = NvmlReturn (*)(NvmlDevice, unsigned int, int*);
  using GetRemoteTypeFn = NvmlReturn (*)(NvmlDevice, unsigned int, int*);
  using GetRemotePciFn = NvmlReturn (*)(NvmlDevice, unsigned int, NvmlPciInfo*);
  using GetFieldsFn = NvmlReturn (*)(NvmlDevice, int, NvmlFieldValue*);

  void* library_ = nullptr;
  InitFn init_ = nullptr;
  ShutdownFn shutdown_ = nullptr;
  GetHandleFn get_handle_ = nullptr;
  GetCapabilityFn get_capability_ = nullptr;
  GetStateFn get_state_ = nullptr;
  GetRemoteTypeFn get_remote_type_ = nullptr;
  GetRemotePciFn get_remote_pci_ = nullptr;
  GetFieldsFn get_fields_ = nullptr;
  bool available_ = false;
};

#endif

struct LocalLinks {
  bool available = false;
  std::uint32_t switch_links = 0;
  std::vector<std::uint64_t> direct_peers;
};

inline LocalLinks queryLocalLinks(const char* pci_bus_id, int compute_capability) {
  LocalLinks result;
#if defined(__linux__)
  NvmlApi nvml;
  NvmlDevice device = nvml.device(pci_bus_id);
  if (device == nullptr) return result;
  const int max_links = compute_capability < 60 ? 0 :
                        compute_capability < 70 ? 4 :
                        compute_capability < 80 ? 6 :
                        compute_capability < 90 ? 12 :
                                                  18;
  result.available = true;
  for (int link = 0; link < max_links; ++link) {
    if (!nvml.activeP2pLink(device, link)) continue;
    const int remote_type = nvml.remoteType(device, link);
    if (remote_type == kNvmlRemoteSwitch) {
      ++result.switch_links;
    } else if (remote_type == kNvmlRemoteGpu) {
      const std::uint64_t remote_pci = nvml.remotePci(device, link);
      if (remote_pci != 0) result.direct_peers.push_back(remote_pci);
    }
  }
#else
  (void)pci_bus_id;
  (void)compute_capability;
#endif
  return result;
}

}  // namespace topology_detail

inline V2Topology discoverV2Topology(ncclComm_t comm, int world_size, int rank, int device,
                                     std::uint32_t channel_limit) {
  using namespace topology_detail;
  V2Topology topology;
  const std::uint32_t channel_ceiling =
    powerOfTwoDown(std::max<std::uint32_t>(1, std::min(channel_limit, static_cast<std::uint32_t>(kMaxChannels))));
  topology.total_channels = channel_ceiling;
  topology.peer_channels.assign(world_size, 1);
  topology.peer_paths.resize(world_size);

  cudaDeviceProp properties{};
  checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties(topology)");
  char pci_bus_id[32] = {};
  checkCuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device), "cudaDeviceGetPCIBusId(topology)");
  const int compute_capability = properties.major * 10 + properties.minor;
  const LocalLinks local_links = queryLocalLinks(pci_bus_id, compute_capability);

  cudaStream_t stream = nullptr;
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate(topology)");
  try {
    const TopologyRecord local_record{
      parsePciBus(pci_bus_id),
      static_cast<std::uint32_t>(compute_capability),
      local_links.switch_links,
      local_links.available ? 1U : 0U,
      0,
    };
    const std::vector<TopologyRecord> records = allGather(comm, &local_record, 1, world_size, stream);

    std::vector<std::uint8_t> local_path_links(world_size, 0);
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) continue;
      std::uint32_t links = 0;
      if (local_links.switch_links != 0 && records[peer].switch_links != 0) {
        links = std::min(local_links.switch_links, records[peer].switch_links);
      } else {
        links = static_cast<std::uint32_t>(std::count(local_links.direct_peers.begin(), local_links.direct_peers.end(),
                                                      records[peer].pci_bus));
      }
      local_path_links[peer] = static_cast<std::uint8_t>(std::min<std::uint32_t>(links, 255));
    }
    const std::vector<std::uint8_t> link_matrix =
      allGather(comm, local_path_links.data(), local_path_links.size(), world_size, stream);

    topology.nvml_available = true;
    for (const TopologyRecord& record : records) topology.nvml_available &= record.nvml_available != 0;
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) continue;
      const std::uint32_t links =
        std::min<std::uint32_t>(link_matrix[static_cast<std::size_t>(rank) * world_size + peer],
                                link_matrix[static_cast<std::size_t>(peer) * world_size + rank]);
      const float link_bw = std::min(nvlinkBandwidth(compute_capability),
                                     nvlinkBandwidth(static_cast<int>(records[peer].compute_capability)));
      V2PeerPath path;
      path.nvlink_count = links;
      path.bandwidth_gbps = links * link_bw;
      path.raw_channels = links == 0 ? 2 : 2 * std::max(1, static_cast<int>(path.bandwidth_gbps / link_bw));
      path.channels = std::min(channel_ceiling, powerOfTwoUp(path.raw_channels));
      topology.peer_paths[peer] = path;
    }

    // NCCL uses the communicator-wide minimum so both ends of every peer pair
    // choose the same p2pnChannelsPerPeer value.
    std::uint32_t communicator_raw_channels = kMaxChannels;
    for (int source = 0; source < world_size; ++source) {
      for (int peer = source + 1; peer < world_size; ++peer) {
        const std::uint32_t links =
          std::min<std::uint32_t>(link_matrix[static_cast<std::size_t>(source) * world_size + peer],
                                  link_matrix[static_cast<std::size_t>(peer) * world_size + source]);
        const float link_bw = std::min(nvlinkBandwidth(static_cast<int>(records[source].compute_capability)),
                                       nvlinkBandwidth(static_cast<int>(records[peer].compute_capability)));
        const float path_bw = links * link_bw;
        const std::uint32_t raw_channels = links == 0 ? 2 : 2 * std::max(1, static_cast<int>(path_bw / link_bw));
        communicator_raw_channels = std::min(communicator_raw_channels, raw_channels);
      }
    }
    // NCCL rounds the path demand up, then caps it by the communicator's
    // channel pool. Its public communicator properties do not expose that
    // internal pool, so v2 derives a non-oversubscribed pool from the same
    // path bandwidth budget by rounding the raw capacity down.
    communicator_raw_channels = std::max<std::uint32_t>(1, communicator_raw_channels);
    topology.requested_channels_per_peer = std::min(channel_ceiling, powerOfTwoUp(communicator_raw_channels));
    topology.total_channels = std::min(channel_ceiling, powerOfTwoDown(communicator_raw_channels));
    topology.channels_per_peer = std::min(topology.total_channels, topology.requested_channels_per_peer);
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer != rank) {
        topology.peer_paths[peer].channels = std::min(topology.peer_paths[peer].channels, topology.total_channels);
        topology.peer_channels[peer] = topology.channels_per_peer;
      }
    }
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy(topology)");
    return topology;
  } catch (...) {
    cudaStreamDestroy(stream);
    throw;
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

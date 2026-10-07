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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace awex {
namespace nccl_device_v2 {

struct V2GinState {
  bool enabled = false;
  bool fp8_source_pipeline = false;
  std::uint32_t signal_count = 0;
  std::uint32_t connection_count = 0;
  std::uint32_t context_count = 0;
  std::uint32_t channel_budget = 0;
  std::uint32_t channels_per_peer = 1;
  std::vector<std::uint64_t> peer_payload_bytes;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  ncclDevComm dev_comm{};
  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  bool created = false;
#endif
};

inline std::uint32_t v2PowerOfTwoUp(std::uint32_t value) {
  std::uint32_t result = 1;
  while (result < value && result < kMaxChannels) result *= 2;
  return result;
}

inline std::uint64_t v2DivideUp(std::uint64_t value, std::uint64_t divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

inline bool v2HasGinPeer(const std::vector<std::uint32_t>& peers,
                         const std::vector<std::uint8_t>& transports) {
  return std::any_of(peers.begin(), peers.end(), [&](std::uint32_t peer) {
    return transports[peer] == static_cast<std::uint8_t>(V2Transport::kGin);
  });
}

inline void v2SetGinType(V2GinState* state, const ncclCommProperties_t& properties) {
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  state->type = properties.ginType;
#else
  (void)state;
  (void)properties;
#endif
}

inline void v2ValidateGinSupport(const V2GinState& state, int nccl_version) {
  if (!state.enabled) return;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  if (nccl_version < NCCL_VERSION(2, 30, 4)) {
    throw std::runtime_error("nccl_device_v2 GIN transport requires NCCL 2.30.4 or newer at runtime");
  }
  if (state.type == NCCL_GIN_TYPE_NONE) {
    throw std::runtime_error("nccl_device_v2 found non-LSA peers, but the NCCL communicator has no GIN support");
  }
#else
  (void)nccl_version;
  throw std::runtime_error(
    "nccl_device_v2 found non-LSA peers, but this extension was not built with NCCL 2.30.4+ GIN headers");
#endif
}

inline void v2ConfigureGinChannels(V2GinState* state, std::uint32_t total_channels,
                                   std::uint32_t gin_fifo_depth, std::size_t network_step_bytes,
                                   const std::vector<std::uint32_t>& active_peers,
                                   const std::vector<std::uint8_t>& transports,
                                   std::vector<std::uint32_t>* peer_channels,
                                   std::uint32_t channel_budget_factor = 6) {
  const std::uint32_t base_channels =
    std::min(total_channels, v2PowerOfTwoUp(2 * state->connection_count));
  // Direct BF16->FP8 adds substantial copy-worker computation to each rail.
  // Use more of the existing channel/FIFO registration to overlap that work;
  // the copy-only and ring budgets stay unchanged. In a many-sender receiver,
  // the network-only budget otherwise negotiates just one or two CTAs per
  // pair, even while most SMs and already registered slots remain unused.
  state->channel_budget = std::min(total_channels, channel_budget_factor * state->connection_count);

  std::vector<std::uint32_t> gin_peers;
  for (const std::uint32_t peer : active_peers) {
    if (transports[peer] == static_cast<std::uint8_t>(V2Transport::kGin)) {
      gin_peers.push_back(peer);
    }
  }
  if (gin_peers.empty()) return;

  if (gin_peers.size() == 1) {
    (*peer_channels)[gin_peers.front()] =
      std::min(total_channels, v2PowerOfTwoUp(state->channel_budget));
  } else {
    const std::uint64_t fifo_window_bytes =
      static_cast<std::uint64_t>(network_step_bytes) * gin_fifo_depth;
    std::vector<std::uint32_t> desired_channels(peer_channels->size(), 1);
    for (const std::uint32_t peer : gin_peers) {
      const std::uint64_t window_demand =
        std::max<std::uint64_t>(1, v2DivideUp(state->peer_payload_bytes[peer], fifo_window_bytes));
      desired_channels[peer] = v2PowerOfTwoUp(
        static_cast<std::uint32_t>(std::min<std::uint64_t>(base_channels, window_demand)));
      (*peer_channels)[peer] = 1;
    }

    // Start with one channel per peer, then spend the shared budget on the
    // highest remaining bytes-per-channel pressure while preserving powers of two.
    std::uint32_t assigned_channels = static_cast<std::uint32_t>(gin_peers.size());
    while (assigned_channels < state->channel_budget) {
      std::uint32_t selected_peer = std::numeric_limits<std::uint32_t>::max();
      for (const std::uint32_t peer : gin_peers) {
        const std::uint32_t channels = (*peer_channels)[peer];
        if (channels >= desired_channels[peer] || assigned_channels + channels > state->channel_budget) {
          continue;
        }
        if (selected_peer == std::numeric_limits<std::uint32_t>::max()) {
          selected_peer = peer;
          continue;
        }
        const std::uint32_t selected_channels = (*peer_channels)[selected_peer];
        const std::uint64_t peer_pressure = state->peer_payload_bytes[peer] * selected_channels;
        const std::uint64_t selected_pressure = state->peer_payload_bytes[selected_peer] * channels;
        if (peer_pressure > selected_pressure ||
            (peer_pressure == selected_pressure && peer < selected_peer)) {
          selected_peer = peer;
        }
      }
      if (selected_peer == std::numeric_limits<std::uint32_t>::max()) {
        break;
      }
      assigned_channels += (*peer_channels)[selected_peer];
      (*peer_channels)[selected_peer] *= 2;
    }
  }

  state->channels_per_peer = 1;
  for (const std::uint32_t peer : gin_peers) {
    state->channels_per_peer = std::max(state->channels_per_peer, (*peer_channels)[peer]);
  }
}

inline void v2InitializeGin(V2GinState* state, ncclComm_t comm, int world_size,
                            std::uint32_t total_channels, std::uint32_t gin_fifo_depth,
                            std::size_t network_step_bytes, std::uint32_t context_count,
                            const std::vector<std::uint32_t>& active_peers,
                            const std::vector<std::uint8_t>& transports,
                            std::vector<std::uint64_t> peer_payload_bytes,
                            std::vector<std::uint32_t>* peer_channels,
                            std::uint32_t channel_budget_factor = 6) {
  if (!state->enabled) return;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  state->signal_count = 2U * static_cast<std::uint32_t>(world_size) * total_channels;
  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = static_cast<int>(std::min(context_count, total_channels));
  requirements.ginSignalCount = static_cast<int>(state->signal_count);
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  requirements.worldGinBarrierCount = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 7)
  requirements.ginStrongSignalsRequired = true;
  requirements.ginVaSignalsRequired = false;
#endif
  const ncclResult_t result = ncclDevCommCreate(comm, &requirements, &state->dev_comm);
  if (result != ncclSuccess) {
    throw std::runtime_error(std::string("ncclDevCommCreate failed: ") + ncclGetErrorString(result));
  }
  state->created = true;
  state->context_count = static_cast<std::uint32_t>(state->dev_comm.ginContextCount);
  state->connection_count = static_cast<std::uint32_t>(state->dev_comm.ginConnectionCount);
  if (state->context_count == 0 || state->connection_count == 0) {
    throw std::runtime_error("nccl_device_v2 GIN initialization returned no contexts or connections");
  }
  state->peer_payload_bytes = std::move(peer_payload_bytes);
  v2ConfigureGinChannels(state, total_channels, gin_fifo_depth, network_step_bytes, active_peers,
                         transports, peer_channels, channel_budget_factor);
#else
  (void)comm;
  (void)world_size;
  (void)total_channels;
  (void)gin_fifo_depth;
  (void)network_step_bytes;
  (void)context_count;
  (void)active_peers;
  (void)transports;
  (void)peer_payload_bytes;
  (void)peer_channels;
  (void)channel_budget_factor;
#endif
}

inline ncclResult_t v2DestroyGin(V2GinState* state, ncclComm_t comm) {
  ncclResult_t result = ncclSuccess;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  if (state->created) {
    result = ncclDevCommDestroy(comm, &state->dev_comm);
  }
#else
  (void)comm;
#endif
  *state = V2GinState{};
  return result;
}

inline std::uint32_t v2GinCreditBatch(std::uint32_t active_gin_peers, std::uint32_t fifo_depth) {
  return active_gin_peers > 1 ? 1 : std::min<std::uint32_t>(4, std::max<std::uint32_t>(1, fifo_depth / 2));
}

inline int v2GinType(const V2GinState& state) {
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  return static_cast<int>(state.type);
#else
  (void)state;
  return 0;
#endif
}

inline std::uint32_t v2GinContextCount(const V2GinState& state) {
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  return state.created ? state.context_count : 0;
#else
  (void)state;
  return 0;
#endif
}

inline void v2SetGinKernelArgs(const V2GinState& state, ncclWindow_t window,
                               std::uint32_t active_gin_peers, std::uint32_t fifo_depth,
                               V2KernelArgs* args) {
  args->gin_enabled = state.enabled ? 1U : 0U;
  args->gin_credit_batch = v2GinCreditBatch(active_gin_peers, fifo_depth);
  args->gin_signal_count = state.signal_count;
  args->gin_fp8_source_pipeline = state.fp8_source_pipeline ? 1U : 0U;
#if AWEX_NCCL_DEVICE_V2_HAS_GIN
  args->window = window;
  if (state.created) args->dev_comm = state.dev_comm;
#else
  (void)window;
#endif
}

}  // namespace nccl_device_v2
}  // namespace awex

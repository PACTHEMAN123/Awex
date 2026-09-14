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
#include <utility>
#include <vector>

namespace awex {
namespace nccl_device_v2 {

struct V2LoweringConfig {
  std::uint32_t world_size = 0;
  std::uint32_t max_channels = 32;
  std::uint32_t fifo_depth = kDefaultFifoDepth;
  std::size_t chunk_bytes = kDefaultChunkBytes;
  std::size_t step_bytes = kDefaultStepBytes;
  // Indexed by peer * max_channels + channel. Each connection owns an
  // independent NCCL-style monotonically increasing FIFO step stream.
  std::vector<std::uint64_t> initial_steps;
};

struct V2LoweringTask {
  std::uintptr_t tensor_ptr = 0;
  std::uint64_t nbytes = 0;
  std::uint64_t tensor_offset = 0;
  std::uint64_t tensor_row_bytes = 0;
  std::uint64_t tensor_row_stride = 0;
  std::uint32_t peer = 0;
  std::uint32_t ordinal = 0;
};

struct V2Schedule {
  std::vector<V2Work> works;
  std::vector<V2WorkBatch> batches;
  std::vector<V2ChannelQueue> channels;
  std::uint32_t channel_count = 0;
  std::uint32_t chunk_count = 0;
  std::uint64_t next_step = 1;
  std::vector<std::uint64_t> next_steps;
};

inline V2WindowLayout makeV2WindowLayout(std::uint32_t world_size, std::uint32_t channel_count,
                                         std::uint32_t fifo_depth, std::size_t slot_bytes) {
  if (world_size == 0 || channel_count == 0 || fifo_depth == 0 || slot_bytes == 0) {
    throw std::invalid_argument("invalid v2 window dimensions");
  }
  const std::size_t connection_count = static_cast<std::size_t>(world_size) * channel_count;
  const std::size_t slot_count = connection_count * fifo_depth;
  const std::size_t state_offset = ((sizeof(V2WindowHeader) + kFifoAlignment - 1) / kFifoAlignment) * kFifoAlignment;
  const std::size_t payload_offset =
    ((state_offset + slot_count * sizeof(V2FifoSlot) + kFifoAlignment - 1) / kFifoAlignment) * kFifoAlignment;
  const std::size_t window_bytes =
    ((payload_offset + slot_count * slot_bytes + kWindowAlignment - 1) / kWindowAlignment) * kWindowAlignment;
  return V2WindowLayout{
    state_offset, payload_offset, slot_bytes, fifo_depth, channel_count, world_size, window_bytes,
  };
}

inline std::pair<std::uint64_t, std::uint64_t> v2PartBounds(std::uint32_t parts, std::uint32_t part,
                                                            std::uint64_t bytes) {
  if (parts == 0 || part >= parts) {
    throw std::invalid_argument("invalid v2 channel partition");
  }
  // This is the same balanced contiguous partition shape used by NCCL P2P
  // work: every channel gets a deterministic interval and no byte is lost.
  const std::uint64_t begin = (bytes * part) / parts;
  const std::uint64_t end = (bytes * (part + 1)) / parts;
  return {begin, end};
}

inline std::uint64_t v2DivUp(std::uint64_t value, std::uint64_t divisor) {
  return value == 0 ? 0 : (value - 1) / divisor + 1;
}

inline std::uint32_t v2ChannelsForBytes(std::uint64_t bytes, std::uint32_t min_channels,
                                        std::uint32_t max_channels, std::size_t step_bytes) {
  if (bytes == 0) return 1;

  // Mirrors NCCL addP2pToPlan for an intra-node SIMPLE P2P operation.
  const std::uint64_t min_part_bytes = std::max<std::uint64_t>(1, step_bytes / 8);
  const std::uint64_t max_part_bytes = static_cast<std::uint64_t>(step_bytes) * 32;
  const std::uint64_t initial_channels = std::min<std::uint64_t>(min_channels, v2DivUp(bytes, min_part_bytes));
  std::uint32_t channels = static_cast<std::uint32_t>(initial_channels);
  std::uint64_t part_bytes = std::max<std::uint64_t>(min_part_bytes, v2DivUp(bytes, channels));
  while (part_bytes > max_part_bytes && channels <= max_channels / 2) {
    channels *= 2;
    part_bytes = v2DivUp(bytes, channels);
  }
  return channels;
}

inline void v2SetSide(V2WorkSide* side, const V2LoweringTask& task, std::uint64_t chunk_offset,
                      std::uint64_t chunk_bytes, std::uint32_t channel_count, const V2LoweringConfig& config,
                      std::vector<std::uint64_t>* next_steps) {
  side->enabled = 1;
  side->channel_base = 0;
  side->channel_count = channel_count;
  side->tensor_ptr = task.tensor_ptr;
  side->nbytes = chunk_bytes;
  side->tensor_offset = task.tensor_offset + chunk_offset;
  side->tensor_row_bytes = task.tensor_row_bytes;
  side->tensor_row_stride = task.tensor_row_stride;

  const std::size_t peer_base = static_cast<std::size_t>(task.peer) * config.max_channels;
  for (std::uint32_t part = 0; part < channel_count; ++part) {
    const auto bounds = v2PartBounds(channel_count, part, chunk_bytes);
    const std::uint64_t part_bytes = bounds.second - bounds.first;
    side->step_begin[part] = (*next_steps)[peer_base + part];
    (*next_steps)[peer_base + part] += std::max<std::uint64_t>(1, v2DivUp(part_bytes, config.step_bytes));
  }
}

// Lower exactly one fixed task stream. The input vector is intentionally not
// sorted or fused: its iteration order is the TransferPlan order.
inline V2Schedule lowerFixedTasks(const std::vector<V2LoweringTask>& tasks,
                                  const std::vector<std::uint32_t>& active_peers, V2Direction direction,
                                  const V2LoweringConfig& config) {
  if (config.world_size == 0 || config.world_size > 256) {
    throw std::invalid_argument("invalid v2 world size");
  }
  if (config.max_channels == 0 || config.max_channels > kMaxChannelsPerPeer) {
    throw std::invalid_argument("invalid v2 max_channels");
  }
  if (config.fifo_depth == 0 || config.step_bytes == 0) {
    throw std::invalid_argument("v2 FIFO depth and step size must be positive");
  }
  if (config.chunk_bytes != 0 && config.chunk_bytes < config.step_bytes) {
    throw std::invalid_argument("v2 chunk_bytes must be zero or at least step_bytes");
  }

  const std::size_t step_count = static_cast<std::size_t>(config.world_size) * config.max_channels;
  if (!config.initial_steps.empty() && config.initial_steps.size() != step_count) {
    throw std::invalid_argument("invalid v2 initial step table");
  }

  V2Schedule schedule;
  schedule.next_steps = config.initial_steps.empty() ? std::vector<std::uint64_t>(step_count, 1)
                                                      : config.initial_steps;

  std::vector<std::int32_t> peer_index(config.world_size, -1);
  for (std::size_t index = 0; index < active_peers.size(); ++index) {
    const std::uint32_t peer = active_peers[index];
    if (peer >= config.world_size || peer_index[peer] != -1) {
      throw std::invalid_argument("v2 active peer list is invalid");
    }
    peer_index[peer] = static_cast<std::int32_t>(index);
  }

  std::uint32_t min_channels = config.max_channels;
  while (static_cast<std::uint64_t>(min_channels) * config.world_size > config.max_channels && min_channels > 1) {
    min_channels /= 2;
  }
  for (const V2LoweringTask& task : tasks) {
    if (task.peer >= config.world_size || peer_index[task.peer] < 0) {
      throw std::invalid_argument("v2 task peer is not active");
    }
    if (task.tensor_row_bytes == 0 || task.tensor_row_stride < task.tensor_row_bytes) {
      throw std::invalid_argument("invalid v2 tensor row layout");
    }
    const std::uint64_t effective_chunk_bytes = config.chunk_bytes == 0 ? task.nbytes : config.chunk_bytes;
    const std::uint64_t chunk_count = task.nbytes == 0 ? 1 : v2DivUp(task.nbytes, effective_chunk_bytes);
    if (chunk_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("v2 task has too many chunks");
    }
    const std::uint32_t channel_count =
      v2ChannelsForBytes(task.nbytes, min_channels, config.max_channels, config.step_bytes);
    schedule.channel_count = std::max(schedule.channel_count, channel_count);

    for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
      const std::uint64_t chunk_offset = std::min<std::uint64_t>(chunk * effective_chunk_bytes, task.nbytes);
      const std::uint64_t chunk_bytes = std::min<std::uint64_t>(effective_chunk_bytes, task.nbytes - chunk_offset);
      V2Work work{};
      work.peer = task.peer;
      work.task_ordinal = task.ordinal;
      work.chunk_ordinal = static_cast<std::uint32_t>(chunk);
      work.chunk_count = static_cast<std::uint32_t>(chunk_count);
      if (direction == V2Direction::kSend) {
        v2SetSide(&work.send, task, chunk_offset, chunk_bytes, channel_count, config, &schedule.next_steps);
      } else {
        v2SetSide(&work.recv, task, chunk_offset, chunk_bytes, channel_count, config, &schedule.next_steps);
      }
      schedule.works.push_back(work);
      ++schedule.chunk_count;
    }
  }

  // Preserve the FIFO pipeline across work records. Only the last work seen
  // for a peer/channel partition waits for its final consumer acknowledgement.
  schedule.channels.resize(schedule.channel_count);
  std::vector<std::uint8_t> seen_final(step_count, 0);
  for (std::size_t index = schedule.works.size(); index > 0; --index) {
    V2WorkSide* side =
      direction == V2Direction::kSend ? &schedule.works[index - 1].send : &schedule.works[index - 1].recv;
    if (!side->enabled) {
      continue;
    }
    for (std::uint32_t part = 0; part < side->channel_count; ++part) {
      const std::size_t connection = static_cast<std::size_t>(schedule.works[index - 1].peer) * config.max_channels +
                                     side->channel_base + part;
      if (seen_final[connection] == 0) {
        side->final_parts |= 1U << part;
        seen_final[connection] = 1;
      }
    }
  }

  for (std::uint32_t begin = 0; begin < schedule.works.size(); begin += kMaxWorksPerBatch) {
    schedule.batches.push_back(V2WorkBatch{
      begin,
      std::min<std::uint32_t>(kMaxWorksPerBatch, static_cast<std::uint32_t>(schedule.works.size() - begin)),
    });
  }
  for (V2ChannelQueue& channel : schedule.channels) {
    channel.first_batch = 0;
    channel.batch_count = static_cast<std::uint32_t>(schedule.batches.size());
  }
  schedule.next_step = 1;
  for (const std::uint64_t step : schedule.next_steps) {
    schedule.next_step = std::max(schedule.next_step, step);
  }
  return schedule;
}

}  // namespace nccl_device_v2
}  // namespace awex

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
  std::uint32_t local_rank = 0;
  std::uint32_t world_size = 0;
  std::uint32_t total_channels = 0;
  std::uint32_t fifo_depth = kDefaultFifoDepth;
  std::size_t chunk_bytes = kDefaultChunkBytes;
  std::size_t step_bytes = kDefaultStepBytes;
  // Topology-derived upper bound for each peer, indexed by rank.
  std::vector<std::uint32_t> peer_channels;
  // Indexed by peer * total_channels + channel.
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
  std::vector<V2Fragment> fragments;
  std::vector<V2WorkBatch> batches;
  std::vector<V2ChannelQueue> channels;
  std::vector<std::uint32_t> channel_ids;
  std::vector<std::uint32_t> peer_channel_counts;
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

inline std::uint64_t v2DivUp(std::uint64_t value, std::uint64_t divisor) {
  return value == 0 ? 0 : (value - 1) / divisor + 1;
}

inline std::uint64_t v2AlignUp(std::uint64_t value, std::uint64_t alignment) {
  return v2DivUp(value, alignment) * alignment;
}

// Match ncclP2pPartBounds so vector alignment survives channel partitioning.
inline std::pair<std::uint64_t, std::uint64_t> v2PartBounds(std::uint32_t parts, std::uint32_t part,
                                                            std::uint64_t bytes) {
  if (parts == 0 || part >= parts) {
    throw std::invalid_argument("invalid v2 channel partition");
  }
  const std::uint64_t part_bytes = v2AlignUp(v2DivUp(bytes, parts), 4 * 1024);
  return {
    std::min<std::uint64_t>(static_cast<std::uint64_t>(part) * part_bytes, bytes),
    std::min<std::uint64_t>(static_cast<std::uint64_t>(part + 1) * part_bytes, bytes),
  };
}

inline std::uint32_t v2ChannelsForBytes(std::uint64_t bytes, std::uint32_t min_channels,
                                        std::uint32_t max_channels, std::size_t step_bytes) {
  if (bytes == 0) return 1;

  // Mirrors NCCL addP2pToPlan for an intra-node SIMPLE P2P operation. The
  // min/max inputs are supplied by the topology layer, not constants.
  const std::uint64_t min_part_bytes = std::max<std::uint64_t>(1, step_bytes / 8);
  const std::uint64_t max_part_bytes = static_cast<std::uint64_t>(step_bytes) * 32;
  const std::uint64_t initial_channels = std::min<std::uint64_t>(min_channels, v2DivUp(bytes, min_part_bytes));
  std::uint32_t channels = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(initial_channels));
  std::uint64_t part_bytes = std::max<std::uint64_t>(min_part_bytes, v2DivUp(bytes, channels));
  while (part_bytes > max_part_bytes && channels <= max_channels / 2) {
    channels *= 2;
    part_bytes = v2DivUp(bytes, channels);
  }
  return channels;
}

inline std::uint32_t v2Log2(std::uint32_t value) {
  std::uint32_t bits = 0;
  while ((1U << bits) < value) ++bits;
  return bits;
}

inline std::uint32_t v2ReverseBits(std::uint64_t value, std::uint32_t bits) {
  std::uint32_t result = 0;
  for (std::uint32_t bit = 0; bit < bits; ++bit) {
    result = (result << 1) | static_cast<std::uint32_t>((value >> bit) & 1ULL);
  }
  return result;
}

inline std::uint32_t v2ChannelBase(std::uint32_t local_rank, std::uint32_t peer, std::uint32_t world_size,
                                   std::uint32_t total_channels) {
  const std::uint32_t low = std::min(local_rank, peer);
  const std::uint32_t high = std::max(local_rank, peer);
  const std::uint64_t pair = static_cast<std::uint64_t>(low) * world_size + high;
  return v2ReverseBits(pair, v2Log2(total_channels));
}

struct V2StreamSpan {
  const V2LoweringTask* task;
  std::uint64_t begin;
  std::uint64_t end;
};

inline void v2AppendFragments(const std::vector<V2StreamSpan>& spans, std::uint64_t work_begin,
                              std::uint64_t work_end, V2Schedule* schedule, V2Work* work) {
  work->fragment_begin = static_cast<std::uint32_t>(schedule->fragments.size());
  for (const V2StreamSpan& span : spans) {
    if (span.end <= work_begin || work_end <= span.begin) continue;
    const std::uint64_t begin = std::max(span.begin, work_begin);
    const std::uint64_t end = std::min(span.end, work_end);
    const V2LoweringTask& task = *span.task;
    schedule->fragments.push_back(V2Fragment{
      task.tensor_ptr,
      end - begin,
      task.tensor_offset + begin - span.begin,
      task.tensor_row_bytes,
      task.tensor_row_stride,
      begin - work_begin,
    });
  }
  const std::size_t fragment_count = schedule->fragments.size() - work->fragment_begin;
  if (fragment_count == 0 || fragment_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("v2 work has an invalid fragment range");
  }
  work->fragment_count = static_cast<std::uint32_t>(fragment_count);
}

// Treat each peer's ordered TransferPlan spans as one virtual tensor. Channel
// partitioning happens on that complete stream; only then are channel parts
// divided into transport chunks and FIFO steps.
inline V2Schedule lowerFixedTasks(const std::vector<V2LoweringTask>& tasks,
                                  const std::vector<std::uint32_t>& active_peers, V2Direction direction,
                                  const V2LoweringConfig& config) {
  if (config.world_size == 0 || config.world_size > 256 || config.local_rank >= config.world_size) {
    throw std::invalid_argument("invalid v2 rank configuration");
  }
  if (config.total_channels == 0 || config.total_channels > kMaxChannels ||
      (config.total_channels & (config.total_channels - 1)) != 0) {
    throw std::invalid_argument("v2 total channel count must be a power of two");
  }
  if (config.peer_channels.size() != config.world_size) {
    throw std::invalid_argument("v2 peer channel table does not match world size");
  }
  if (config.fifo_depth == 0 || config.step_bytes == 0) {
    throw std::invalid_argument("v2 FIFO depth and step size must be positive");
  }
  if (config.chunk_bytes != 0 && config.chunk_bytes < config.step_bytes) {
    throw std::invalid_argument("v2 chunk_bytes must be zero or at least step_bytes");
  }

  const std::size_t step_count = static_cast<std::size_t>(config.world_size) * config.total_channels;
  if (!config.initial_steps.empty() && config.initial_steps.size() != step_count) {
    throw std::invalid_argument("invalid v2 initial step table");
  }

  V2Schedule schedule;
  schedule.next_steps = config.initial_steps.empty() ? std::vector<std::uint64_t>(step_count, 1) : config.initial_steps;
  schedule.peer_channel_counts.assign(config.world_size, 0);

  std::vector<std::int32_t> peer_index(config.world_size, -1);
  for (std::size_t index = 0; index < active_peers.size(); ++index) {
    const std::uint32_t peer = active_peers[index];
    if (peer >= config.world_size || peer == config.local_rank || peer_index[peer] != -1) {
      throw std::invalid_argument("v2 active peer list is invalid");
    }
    peer_index[peer] = static_cast<std::int32_t>(index);
  }

  std::vector<std::vector<const V2LoweringTask*>> peer_tasks(config.world_size);
  for (const V2LoweringTask& task : tasks) {
    if (task.peer >= config.world_size || peer_index[task.peer] < 0) {
      throw std::invalid_argument("v2 task peer is not active");
    }
    if (task.tensor_row_bytes == 0 || task.tensor_row_stride < task.tensor_row_bytes) {
      throw std::invalid_argument("invalid v2 tensor row layout");
    }
    if (task.ordinal != peer_tasks[task.peer].size()) {
      throw std::invalid_argument("v2 task order is not dense within its peer stream");
    }
    peer_tasks[task.peer].push_back(&task);
  }

  using PeerWorkQueues = std::vector<std::vector<V2Work>>;
  std::vector<PeerWorkQueues> channel_work(config.total_channels, PeerWorkQueues(config.world_size));
  for (const std::uint32_t peer : active_peers) {
    std::vector<V2StreamSpan> spans;
    std::uint64_t stream_bytes = 0;
    for (const V2LoweringTask* task : peer_tasks[peer]) {
      if (task->nbytes > std::numeric_limits<std::uint64_t>::max() - stream_bytes) {
        throw std::invalid_argument("v2 peer stream size overflows");
      }
      if (task->nbytes != 0) {
        spans.push_back(V2StreamSpan{task, stream_bytes, stream_bytes + task->nbytes});
      }
      stream_bytes += task->nbytes;
    }
    if (stream_bytes == 0) continue;

    const std::uint32_t max_channels =
      std::max<std::uint32_t>(1, std::min(config.peer_channels[peer], config.total_channels));
    std::uint32_t min_channels = max_channels;
    while (static_cast<std::uint64_t>(min_channels) * config.world_size > config.total_channels && min_channels > 1) {
      min_channels /= 2;
    }
    const std::uint32_t channel_count =
      v2ChannelsForBytes(stream_bytes, min_channels, max_channels, config.step_bytes);
    schedule.peer_channel_counts[peer] = channel_count;
    const std::uint32_t channel_base =
      v2ChannelBase(config.local_rank, peer, config.world_size, config.total_channels);

    for (std::uint32_t part = 0; part < channel_count; ++part) {
      const auto bounds = v2PartBounds(channel_count, part, stream_bytes);
      if (bounds.first == bounds.second) continue;
      const std::uint32_t channel = (channel_base + part) & (config.total_channels - 1);
      const std::uint64_t part_bytes = bounds.second - bounds.first;
      const std::uint64_t effective_chunk_bytes = config.chunk_bytes == 0 ? part_bytes : config.chunk_bytes;
      const std::uint64_t chunk_count = v2DivUp(part_bytes, effective_chunk_bytes);
      if (chunk_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("v2 channel part has too many chunks");
      }

      auto& queue = channel_work[channel][peer];
      for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const std::uint64_t chunk_begin = bounds.first + chunk * effective_chunk_bytes;
        const std::uint64_t chunk_end = std::min<std::uint64_t>(chunk_begin + effective_chunk_bytes, bounds.second);
        V2Work work{};
        work.peer = peer;
        work.chunk_ordinal = static_cast<std::uint32_t>(chunk);
        work.chunk_count = static_cast<std::uint32_t>(chunk_count);
        work.stream_offset = chunk_begin;
        work.nbytes = chunk_end - chunk_begin;
        const std::size_t connection = static_cast<std::size_t>(peer) * config.total_channels + channel;
        work.step_begin = schedule.next_steps[connection];
        schedule.next_steps[connection] += v2DivUp(work.nbytes, config.step_bytes);
        v2AppendFragments(spans, chunk_begin, chunk_end, &schedule, &work);
        queue.push_back(work);
        ++schedule.chunk_count;
      }
      queue.back().final = 1;
    }
  }

  // Each batch contains at most one work per peer, so concurrent warp groups
  // never race on the same (peer, channel) FIFO step stream.
  for (std::uint32_t channel = 0; channel < config.total_channels; ++channel) {
    std::size_t remaining = 0;
    std::vector<std::size_t> cursors(config.world_size, 0);
    for (const std::uint32_t peer : active_peers) remaining += channel_work[channel][peer].size();
    if (remaining == 0) continue;

    schedule.channel_ids.push_back(channel);
    V2ChannelQueue channel_queue{};
    channel_queue.first_batch = static_cast<std::uint32_t>(schedule.batches.size());
    std::size_t peer_cursor = 0;
    while (remaining != 0) {
      V2WorkBatch batch{};
      batch.work_begin = static_cast<std::uint32_t>(schedule.works.size());
      const std::size_t batch_peer_begin = peer_cursor;
      std::size_t last_peer_index = batch_peer_begin;
      std::size_t scanned = 0;
      while (scanned < active_peers.size() && batch.work_count < kMaxWorksPerBatch) {
        const std::size_t index = (batch_peer_begin + scanned) % active_peers.size();
        const std::uint32_t peer = active_peers[index];
        auto& queue = channel_work[channel][peer];
        if (cursors[peer] < queue.size()) {
          schedule.works.push_back(queue[cursors[peer]++]);
          ++batch.work_count;
          --remaining;
          last_peer_index = index;
        }
        ++scanned;
      }
      if (batch.work_count == 0) {
        throw std::logic_error("v2 channel scheduler made no progress");
      }
      peer_cursor = (last_peer_index + 1) % active_peers.size();
      schedule.batches.push_back(batch);
      ++channel_queue.batch_count;
    }
    schedule.channels.push_back(channel_queue);
  }

  schedule.channel_count = static_cast<std::uint32_t>(schedule.channel_ids.size());
  for (const std::uint64_t step : schedule.next_steps) schedule.next_step = std::max(schedule.next_step, step);
  (void)direction;
  return schedule;
}

}  // namespace nccl_device_v2
}  // namespace awex

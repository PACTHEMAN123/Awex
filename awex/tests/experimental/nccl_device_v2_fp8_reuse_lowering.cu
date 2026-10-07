// Licensed to the Awex developers under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with it.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "awex/transfer/nccl_device_v2/device_v2_lowering.h"

#include <cassert>
#include <iostream>

namespace v2 = awex::nccl_device_v2;

int main() {
  v2::V2LoweringConfig config;
  config.world_size = 3;
  config.local_rank = 2;
  config.total_channels = 8;
  config.peer_channels = {4, 4, 0};
  config.peer_transports = {1, 1, 0};
  std::vector<v2::V2LoweringTask> tasks(2);
  for (unsigned peer = 0; peer < 2; ++peer) {
    auto& task = tasks[peer];
    task.tensor_ptr = 0x100000;
    task.nbytes = 16400ULL * 1024;
    task.tensor_row_bytes = 512;
    task.tensor_row_stride = 640;
    task.peer = peer;
    task.block_rows = 128;
    task.block_cols = 128;
  }
  const std::vector<std::uint32_t> peers{0, 1};
  const auto original = v2::lowerFixedTasks(tasks, peers, v2::V2Direction::kSend, config);
  auto folded = original;
  v2::reuseIdenticalFp8Source(&folded, config, peers, v2::V2Direction::kSend);
  assert(folded.works.size() * 2 == original.works.size());
  assert(folded.fp8_reused_source_bytes == tasks[0].nbytes);
  assert(folded.next_steps == original.next_steps);
  assert(folded.peer_channel_counts == original.peer_channel_counts);
  // Every removed stream still has its exact original channel and counter.
  for (const auto& work : folded.works) {
    bool found = false;
    for (std::size_t c = 0; c < original.channels.size(); ++c) {
      const auto& queue = original.channels[c];
      for (unsigned b = 0; b < queue.batch_count; ++b) {
        const auto& batch = original.batches[queue.first_batch + b];
        for (unsigned w = 0; w < batch.work_count; ++w) {
          const auto& other = original.works[batch.work_begin + w];
          if (other.peer == 1 && other.stream_offset == work.stream_offset && other.nbytes == work.nbytes) {
            assert(work.duplicate_peer == other.peer);
            assert(work.duplicate_channel == original.channel_ids[c]);
            assert(work.duplicate_step_begin == other.step_begin);
            found = true;
          }
        }
      }
    }
    assert(found);
  }
  for (int mismatch = 0; mismatch < 5; ++mismatch) {
    auto incompatible_tasks = tasks;
    auto incompatible_config = config;
    if (mismatch == 0) incompatible_tasks[1].tensor_ptr += 16;
    if (mismatch == 1) incompatible_tasks[1].tensor_row_stride += 128;
    if (mismatch == 2) incompatible_config.peer_channels[1] = 2;
    if (mismatch == 3) for (auto& task : incompatible_tasks) task.block_rows = task.block_cols = 0;
    auto plan = v2::lowerFixedTasks(incompatible_tasks, peers, v2::V2Direction::kSend, incompatible_config);
    if (mismatch == 4) plan.works[0].ring_id = 2;
    const auto before = plan.works.size();
    v2::reuseIdenticalFp8Source(&plan, incompatible_config, peers, v2::V2Direction::kSend);
    assert(plan.works.size() == before && plan.fp8_reused_source_bytes == 0);
  }
  auto recv = original;
  v2::reuseIdenticalFp8Source(&recv, config, peers, v2::V2Direction::kRecv);
  assert(recv.works.size() == original.works.size() && recv.fp8_reused_source_bytes == 0);
  std::cout << "FP8 source reuse lowering invariants passed\n";
}

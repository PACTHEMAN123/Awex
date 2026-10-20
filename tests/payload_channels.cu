#include <shardstream/lowering.h>
#include <cassert>
#include <numeric>

using namespace shardstream::transport;

int main() {
  LoweringConfig config;
  config.world_size = 4;
  config.total_channels = 64;
  config.peer_transports = {0, 1, 0, 1};
  Schedule schedule;
  schedule.channel_ids = {3, 57};
  schedule.channels = {{0, 1}, {1, 1}};
  schedule.batches = {{0, 2}, {2, 1}};
  Work receive{};
  receive.peer = 1;
  receive.forward_peer = 2;
  Work local{};
  local.peer = 2;
  local.forward_peer = kNoPeer;
  Work remote{};
  remote.peer = 1;
  remote.forward_peer = 3;
  schedule.works = {receive, local, remote};
  const auto mask = payloadChannelMask(schedule, config, Direction::kRecv);
  assert(mask[1 * 64 + 3] && mask[2 * 64 + 3]);
  assert(mask[1 * 64 + 57] && mask[3 * 64 + 57]);
  assert(!mask[2 * 64 + 57]);
  assert(std::accumulate(mask.begin(), mask.end(), 0U) == 4);
  Schedule local_only;
  local_only.channel_ids = {7};
  local_only.channels = {{0, 1}};
  local_only.batches = {{0, 1}};
  local_only.works = {local};
  auto local_mask = payloadChannelMask(local_only, config, Direction::kRecv);
  assert(std::accumulate(local_mask.begin(), local_mask.end(), 0U) == 0);
  local_mask = payloadChannelMask(local_only, config, Direction::kSend);
  assert(local_mask[2 * 64 + 7]);
  const auto full = makeWindowLayout(4, 64, 16, 131072, 3);
  const auto compact = makeWindowLayout(4, 64, 16, 131072, 3, 4);
  assert(full.payload_offset == compact.payload_offset);
  assert(full.payload_connection_count == 192);
  assert(compact.payload_connection_count == 4);
  assert(full.window_bytes - compact.window_bytes == 188ULL * 16 * 131072);
  bool rejected = false;
  try { (void)makeWindowLayout(4, 64, 16, 131072, 3, 193); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

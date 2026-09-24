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

#include "device_v2_primitives.cuh"

#include <cuda/barrier>
#include <cuda/ptx>

namespace awex {
namespace nccl_device_v2 {

namespace ptx = cuda::ptx;
using V2TmaBarrier = cuda::barrier<cuda::thread_scope_block>;
constexpr int kTmaControlWarps = 1;
constexpr int kTmaWorkerThreads = kTmaQuantThreads - kTmaControlWarps * kWarpSize;
constexpr int kTmaWorkerWarps = kTmaWorkerThreads / kWarpSize;
constexpr int kTmaWorkerGroups = 2;
constexpr int kTmaFirstGroupWarps = kTmaWorkerWarps / kTmaWorkerGroups;
constexpr int kTmaSecondGroupWarps = kTmaWorkerWarps - kTmaFirstGroupWarps;
constexpr int kTmaMaxGroupWarps = kTmaSecondGroupWarps;
constexpr int kTmaWorkerBarrierBase = 1;
constexpr int kTmaHandoffBarrierBase = kTmaWorkerBarrierBase + kTmaWorkerGroups;
constexpr int kTmaReceiverHandoffBarrier = 1;

__device__ __forceinline__ bool v2TmaElected(bool control) {
  return control && ptx::elect_sync(0xffffffffU);
}

__device__ __forceinline__ float v2TmaScaleForFp8(float value, float scale, float inverse_scale) {
  const float approximate = value * inverse_scale;
  const std::uint32_t magnitude_bits = __float_as_uint(fabsf(approximate));
  if (magnitude_bits == 0) return approximate;

  constexpr std::uint32_t kFp8MinNormalBits = 121U << 23U;  // 2^-6
  constexpr std::uint32_t kFloatInfinityBits = 0x7f800000U;
  if (magnitude_bits < kFp8MinNormalBits || magnitude_bits >= kFloatInfinityBits) {
    return value / scale;
  }

  // E4M3 keeps three of the float mantissa bits. Reciprocal multiplication can
  // differ from division by a few float ulps, so refine only near a rounding tie.
  constexpr std::uint32_t kDiscardedMask = (1U << 20U) - 1U;
  constexpr std::uint32_t kRoundingTie = 1U << 19U;
  constexpr std::uint32_t kRefineUlps = 16U;
  const std::uint32_t discarded = magnitude_bits & kDiscardedMask;
  const std::uint32_t tie_distance =
    discarded > kRoundingTie ? discarded - kRoundingTie : kRoundingTie - discarded;
  return tie_distance <= kRefineUlps ? value / scale : approximate;
}

__device__ __forceinline__ bool v2TmaWaitForPeers(const V2KernelArgs& args, bool control_lane) {
  auto* local_header = reinterpret_cast<V2WindowHeader*>(args.local_window);
  __shared__ int peers_ready;
  if (control_lane) {
    peers_ready = 1;
    for (std::uint32_t index = 0; index < args.active_peer_count && peers_ready; ++index) {
      const std::uint32_t peer = args.active_peers[index];
      auto* remote_header = reinterpret_cast<V2WindowHeader*>(args.peer_windows[peer]);
      unsigned long long cache = 0;
      peers_ready = v2WaitReady(&remote_header->epoch, args.epoch, &cache, &local_header->error,
                                args.timeout_cycles);
    }
    if (!peers_ready) atomicExch_system(&local_header->error, 4U);
  }
  __syncthreads();
  return peers_ready != 0;
}

__device__ __forceinline__ void v2TmaIssueLoad(const V2TmaKernelArgs& args, const V2TmaQuantTile& tile,
                                               __nv_bfloat16* shared_tile, V2TmaBarrier& barrier) {
#if __CUDA_ARCH__ >= 900
  const CUtensorMap* tensor_map = args.tensor_maps + tile.tensor_map_index;
  constexpr auto descriptor_bytes = ptx::n32_t<128>();
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, descriptor_bytes);
  const std::int32_t coordinates[2] = {
    static_cast<std::int32_t>(tile.tile_col * kTmaQuantBlockCols),
    static_cast<std::int32_t>(tile.tile_row * kTmaQuantBlockRows),
  };
  ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, shared_tile, tensor_map, coordinates,
                            cuda::device::barrier_native_handle(barrier));
  (void)cuda::device::barrier_arrive_tx(barrier, 1, kTmaQuantElements * sizeof(__nv_bfloat16));
#else
  (void)args;
  (void)tile;
  (void)shared_tile;
  (void)barrier;
#endif
}

__global__ void __launch_bounds__(kTmaQuantThreads, 1) tma_quant_send_kernel(V2TmaKernelArgs args) {
#if __CUDA_ARCH__ >= 900
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto* shared_tiles = reinterpret_cast<__nv_bfloat16*>(shared_bytes);
  auto* shared_quantized =
    shared_bytes + kTmaWorkerGroups * kTmaQuantElements * sizeof(__nv_bfloat16);
  __shared__ V2TmaBarrier barriers[2];
  __shared__ float warp_max[kTmaWorkerGroups][kTmaMaxGroupWarps];
  __shared__ float block_scale[kTmaWorkerGroups];
  __shared__ float block_inv_scale[kTmaWorkerGroups];
  __shared__ int fifo_ready;
  __shared__ unsigned long long fifo_cache;

  const V2KernelArgs& transport = args.transport;
  auto* local_header = reinterpret_cast<V2WindowHeader*>(transport.local_window);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const bool control = warp == kTmaWorkerWarps;
  const bool worker = !control;
  const int worker_warp = warp;
  const int worker_group = worker_warp < kTmaFirstGroupWarps ? 0 : 1;
  const int group_warp_base = worker_group == 0 ? 0 : kTmaFirstGroupWarps;
  const int group_warps = worker_group == 0 ? kTmaFirstGroupWarps : kTmaSecondGroupWarps;
  const int group_threads = group_warps * kWarpSize;
  const int group_warp = worker_warp - group_warp_base;
  const int group_tid = group_warp * kWarpSize + lane;
  const bool control_lane = control && lane == 0;
  if (blockIdx.x == 0 && control_lane) v2Publish(&local_header->epoch, transport.epoch);
  if (control_lane) {
    init(&barriers[0], 1);
    init(&barriers[1], 1);
    fifo_ready = 0;
    fifo_cache = 0;
  }
  if (!v2TmaWaitForPeers(transport, control_lane)) return;

  const V2TmaChannelQueue queue = args.queues[blockIdx.x];
  const std::uint32_t preload = queue.tile_count < 2 ? queue.tile_count : 2;
  if (v2TmaElected(control)) {
    for (std::uint32_t stage = 0; stage < preload; ++stage) {
      const V2TmaQuantTile& tile = args.tiles[queue.tile_begin + stage];
      v2TmaIssueLoad(args, tile, shared_tiles + static_cast<std::size_t>(stage) * kTmaQuantElements,
                     barriers[stage]);
    }
  }

  if (worker) {
    for (std::uint32_t index = worker_group; index < queue.tile_count; index += kTmaWorkerGroups) {
      const std::uint32_t stage = static_cast<std::uint32_t>(worker_group);
      const std::uint32_t parity = (index >> 1U) & 1U;
      while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                            cuda::device::barrier_native_handle(barriers[stage]), parity)) {
      }

      const V2TmaQuantTile tile = args.tiles[queue.tile_begin + index];
      const auto* shared_tile = shared_tiles + static_cast<std::size_t>(stage) * kTmaQuantElements;
      float local_max = 0.0F;
      for (std::uint32_t element = group_tid; element < kTmaQuantElements; element += group_threads) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(shared_tile[element])));
      }
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffU, local_max, offset));
      }
      if (lane == 0) warp_max[worker_group][group_warp] = local_max;
      v2GroupBarrier(kTmaWorkerBarrierBase + worker_group, group_threads);
      if (group_warp == 0) {
        local_max = lane < group_warps ? warp_max[worker_group][lane] : 0.0F;
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
          local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffU, local_max, offset));
        }
        if (lane == 0) {
          const float scale = fmaxf(local_max, 1.0e-4F) / 448.0F;
          block_scale[worker_group] = scale;
          block_inv_scale[worker_group] = 1.0F / scale;
        }
      }
      v2GroupBarrier(kTmaHandoffBarrierBase + worker_group, group_threads + kWarpSize);
      if (!fifo_ready) return;

      auto* shared_output =
        shared_quantized + static_cast<std::size_t>(worker_group) * kTmaQuantPayloadBytes;
      constexpr std::uint32_t pair_count = kTmaQuantElements / 2;
      for (std::uint32_t pair = group_tid; pair < pair_count; pair += group_threads) {
        const float2 values = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(shared_tile)[pair]);
        const float2 scaled = {
          v2TmaScaleForFp8(values.x, block_scale[worker_group], block_inv_scale[worker_group]),
          v2TmaScaleForFp8(values.y, block_scale[worker_group], block_inv_scale[worker_group]),
        };
        const __nv_fp8x2_e4m3 encoded(scaled);
        reinterpret_cast<std::uint16_t*>(shared_output)[pair] = encoded.__x;
      }
      v2GroupBarrier(kTmaHandoffBarrierBase + worker_group, group_threads + kWarpSize);
    }
  } else {
    for (std::uint32_t index = 0; index < queue.tile_count; index += kTmaWorkerGroups) {
      const std::uint32_t active_groups = queue.tile_count - index < kTmaWorkerGroups
                                            ? queue.tile_count - index
                                            : kTmaWorkerGroups;
      if (control_lane) {
        for (std::uint32_t group = 0; group < active_groups; ++group) {
          const V2TmaQuantTile& tile = args.tiles[queue.tile_begin + index + group];
          if (tile.payload_offset == 0) {
            V2FifoSlot* slot =
              v2FifoSlot(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true);
            fifo_ready = v2WaitFree(slot, tile.step, transport.layout.fifo_depth, &fifo_cache,
                                    &local_header->error, transport.timeout_cycles);
          }
        }
      }

      for (std::uint32_t group = 0; group < active_groups; ++group) {
        const int active_group_warps = group == 0 ? kTmaFirstGroupWarps : kTmaSecondGroupWarps;
        v2GroupBarrier(kTmaHandoffBarrierBase + group, (active_group_warps + 1) * kWarpSize);
      }
      if (!fifo_ready) return;

      if (control_lane) {
        for (std::uint32_t group = 0; group < active_groups; ++group) {
          const V2TmaQuantTile& tile = args.tiles[queue.tile_begin + index + group];
          auto* payload =
            v2FifoPayload(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true) +
            tile.payload_offset;
          reinterpret_cast<V2TmaPacketHeader*>(payload)->scale = block_scale[group];
          auto* scale = reinterpret_cast<float*>(tile.scale_ptr);
          scale[static_cast<std::uint64_t>(tile.tile_row) * (tile.scale_row_stride / sizeof(float)) +
                tile.tile_col] = block_scale[group];
        }
      }

      for (std::uint32_t group = 0; group < active_groups; ++group) {
        const int active_group_warps = group == 0 ? kTmaFirstGroupWarps : kTmaSecondGroupWarps;
        v2GroupBarrier(kTmaHandoffBarrierBase + group, (active_group_warps + 1) * kWarpSize);
        const std::uint32_t tile_index = index + group;
        const V2TmaQuantTile& tile = args.tiles[queue.tile_begin + tile_index];
        if (control_lane) {
          auto* payload =
            v2FifoPayload(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true) +
            tile.payload_offset;
          ptx::fence_proxy_async(ptx::space_shared);
          ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, payload + kTmaQuantHeaderBytes,
                             shared_quantized + static_cast<std::size_t>(group) * kTmaQuantPayloadBytes,
                             static_cast<std::uint32_t>(kTmaQuantPayloadBytes));
          ptx::cp_async_bulk_commit_group();
        }
        const std::uint32_t fetch = tile_index + kTmaWorkerGroups;
        if (v2TmaElected(control) && fetch < queue.tile_count) {
          const V2TmaQuantTile& next = args.tiles[queue.tile_begin + fetch];
          v2TmaIssueLoad(args, next, shared_tiles + static_cast<std::size_t>(group) * kTmaQuantElements,
                         barriers[group]);
        }
      }
      if (control_lane) ptx::cp_async_bulk_wait_group(ptx::n32_t<0>());

      for (std::uint32_t group = 0; group < active_groups; ++group) {
        const std::uint32_t tile_index = index + group;
        const V2TmaQuantTile& tile = args.tiles[queue.tile_begin + tile_index];
        const bool packet_end =
          tile_index + 1 == queue.tile_count || args.tiles[queue.tile_begin + tile_index + 1].step != tile.step;
        if (packet_end && control_lane && v2LoadError(&local_header->error) == 0) {
          V2FifoSlot* slot =
            v2FifoSlot(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true);
          slot->bytes = tile.payload_offset + static_cast<std::uint32_t>(kTmaQuantPacketBytes);
          v2Publish(&slot->ready_step, tile.step);
        }
      }
    }
  }

  if (queue.tile_count != 0 && control_lane) {
    const V2TmaQuantTile& tile = args.tiles[queue.tile_begin + queue.tile_count - 1];
    V2FifoSlot* slot = v2FifoSlot(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true);
    (void)v2WaitConsumed(slot, tile.step, &fifo_cache, &local_header->error, transport.timeout_cycles);
  }
#else
  (void)args;
#endif
}

__global__ void __launch_bounds__(kTmaQuantThreads, 1) tma_quant_recv_kernel(V2TmaKernelArgs args) {
  const V2KernelArgs& transport = args.transport;
  auto* local_header = reinterpret_cast<V2WindowHeader*>(transport.local_window);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const bool control = warp == kTmaWorkerWarps;
  const bool worker = !control;
  const int worker_tid = threadIdx.x;
  const bool control_lane = control && lane == 0;
  if (blockIdx.x == 0 && control_lane) v2Publish(&local_header->epoch, transport.epoch);
  __shared__ int fifo_ready;
  __shared__ unsigned long long fifo_cache;
  if (control_lane) fifo_cache = 0;
  if (!v2TmaWaitForPeers(transport, control_lane)) return;

  const V2TmaChannelQueue queue = args.queues[blockIdx.x];
  for (std::uint32_t index = 0; index < queue.tile_count; ++index) {
    const V2TmaQuantTile tile = args.tiles[queue.tile_begin + index];
    V2FifoSlot* slot = v2FifoSlot(transport, queue.peer, transport.local_rank, queue.channel, tile.step, false);
    const bool packet_begin = tile.payload_offset == 0;
    const bool packet_end =
      index + 1 == queue.tile_count || args.tiles[queue.tile_begin + index + 1].step != tile.step;
    if (packet_begin && control_lane) {
      fifo_ready = v2WaitReady(&slot->ready_step, tile.step, &fifo_cache, &local_header->error,
                               transport.timeout_cycles);
      if (fifo_ready &&
          (slot->bytes == 0 || slot->bytes > transport.layout.slot_bytes ||
           slot->bytes % kTmaQuantPacketBytes != 0)) {
        atomicExch_system(&local_header->error, 5U);
        fifo_ready = 0;
      }
    }
    if (packet_begin) v2GroupBarrier(kTmaReceiverHandoffBarrier, kTmaQuantThreads);
    if (!fifo_ready) return;

    const auto* payload =
      v2FifoPayload(transport, queue.peer, transport.local_rank, queue.channel, tile.step, false) +
      tile.payload_offset;
    if (control_lane) {
      auto* scale = reinterpret_cast<float*>(tile.scale_ptr);
      scale[static_cast<std::uint64_t>(tile.tile_row) * (tile.scale_row_stride / sizeof(float)) + tile.tile_col] =
        reinterpret_cast<const V2TmaPacketHeader*>(payload)->scale;
    }
    auto* tensor = reinterpret_cast<std::uint8_t*>(tile.tensor_ptr);
    constexpr std::uint32_t packs_per_row = kTmaQuantBlockCols / kCopyPackBytes;
    constexpr std::uint32_t pack_count = kTmaQuantBlockRows * packs_per_row;
    if (worker) {
      for (std::uint32_t pack = worker_tid; pack < pack_count; pack += kTmaWorkerThreads) {
        const std::uint32_t row = pack / packs_per_row;
        const std::uint32_t col_pack = pack - row * packs_per_row;
        const auto value = v2Load128(payload + kTmaQuantHeaderBytes + pack * kCopyPackBytes);
        auto* destination = tensor + static_cast<std::uint64_t>(tile.tile_row * kTmaQuantBlockRows + row) *
                                      tile.tensor_row_stride +
                            static_cast<std::uint64_t>(tile.tile_col * kTmaQuantBlockCols) +
                            col_pack * kCopyPackBytes;
        v2Store128(destination, value);
      }
    }
    if (packet_end) {
      v2GroupBarrier(kTmaReceiverHandoffBarrier, kTmaQuantThreads);
      if (control_lane && v2LoadError(&local_header->error) == 0) {
        v2Publish(&slot->consumed_step, tile.step);
      }
    }
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

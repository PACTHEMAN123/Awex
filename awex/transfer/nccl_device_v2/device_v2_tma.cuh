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
constexpr int kTmaWorkerBarrier = 1;
constexpr int kTmaHandoffBarrier = 2;
constexpr std::uint32_t kTmaFp8PairsPerStore = kCopyPackBytes / sizeof(std::uint16_t);

union alignas(16) V2TmaEncodedPack {
  std::uint16_t pairs[kTmaFp8PairsPerStore];
  V2Pack128 vector;
};

static_assert((kTmaQuantElements / 2) % kTmaFp8PairsPerStore == 0, "FP8 tile must contain full vector stores");

__device__ __forceinline__ bool v2TmaElected(bool control) {
  return control && ptx::elect_sync(0xffffffffU);
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
  __shared__ V2TmaBarrier barriers[2];
  __shared__ float warp_max[kTmaWorkerWarps];
  __shared__ float block_scale;
  __shared__ int fifo_ready;
  __shared__ unsigned long long fifo_cache;

  const V2KernelArgs& transport = args.transport;
  auto* local_header = reinterpret_cast<V2WindowHeader*>(transport.local_window);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const bool control = warp == kTmaWorkerWarps;
  const bool worker = !control;
  const int worker_warp = warp;
  const int worker_tid = threadIdx.x;
  const bool control_lane = control && lane == 0;
  if (blockIdx.x == 0 && control_lane) v2Publish(&local_header->epoch, transport.epoch);
  if (control_lane) {
    init(&barriers[0], 1);
    init(&barriers[1], 1);
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

  for (std::uint32_t index = 0; index < queue.tile_count; ++index) {
    const std::uint32_t stage = index & 1U;
    const std::uint32_t parity = (index >> 1U) & 1U;
    if (worker) {
      while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                            cuda::device::barrier_native_handle(barriers[stage]), parity)) {
      }
    }

    const V2TmaQuantTile tile = args.tiles[queue.tile_begin + index];
    V2FifoSlot* slot = v2FifoSlot(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true);
    if (control_lane && tile.payload_offset == 0) {
      fifo_ready = v2WaitFree(slot, tile.step, transport.layout.fifo_depth, &fifo_cache, &local_header->error,
                              transport.timeout_cycles);
    }

    const auto* shared_tile = shared_tiles + static_cast<std::size_t>(stage) * kTmaQuantElements;
    if (worker) {
      float local_max = 0.0F;
      constexpr std::uint32_t pair_count = kTmaQuantElements / 2;
      constexpr std::uint32_t pack_count = pair_count / kTmaFp8PairsPerStore;
      const auto* shared_pairs = reinterpret_cast<const __nv_bfloat162*>(shared_tile);
      for (std::uint32_t pack = worker_tid; pack < pack_count; pack += kTmaWorkerThreads) {
#pragma unroll
        for (std::uint32_t pair = 0; pair < kTmaFp8PairsPerStore; ++pair) {
          const float2 values = __bfloat1622float2(shared_pairs[pack * kTmaFp8PairsPerStore + pair]);
          local_max = fmaxf(local_max, fmaxf(fabsf(values.x), fabsf(values.y)));
        }
      }
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffU, local_max, offset));
      }
      if (lane == 0) warp_max[worker_warp] = local_max;
      v2GroupBarrier(kTmaWorkerBarrier, kTmaWorkerThreads);
      if (worker_warp == 0) {
        local_max = lane < kTmaWorkerWarps ? warp_max[lane] : 0.0F;
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
          local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffU, local_max, offset));
        }
        if (lane == 0) block_scale = fmaxf(local_max, 1.0e-4F) / 448.0F;
      }
    }
    v2GroupBarrier(kTmaHandoffBarrier, kTmaQuantThreads);
    if (!fifo_ready) return;

    auto* payload = v2FifoPayload(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true) +
                    tile.payload_offset;
    if (control_lane) {
      reinterpret_cast<V2TmaPacketHeader*>(payload)->scale = block_scale;
      auto* scale = reinterpret_cast<float*>(tile.scale_ptr);
      scale[static_cast<std::uint64_t>(tile.tile_row) * (tile.scale_row_stride / sizeof(float)) + tile.tile_col] =
        block_scale;
    }
    if (worker) {
      constexpr std::uint32_t pair_count = kTmaQuantElements / 2;
      constexpr std::uint32_t pack_count = pair_count / kTmaFp8PairsPerStore;
      const auto* shared_pairs = reinterpret_cast<const __nv_bfloat162*>(shared_tile);
      for (std::uint32_t pack = worker_tid; pack < pack_count; pack += kTmaWorkerThreads) {
        V2TmaEncodedPack encoded_pack;
#pragma unroll
        for (std::uint32_t pair = 0; pair < kTmaFp8PairsPerStore; ++pair) {
          const float2 values = __bfloat1622float2(shared_pairs[pack * kTmaFp8PairsPerStore + pair]);
          const float2 scaled = {values.x / block_scale, values.y / block_scale};
          encoded_pack.pairs[pair] = __nv_fp8x2_e4m3(scaled).__x;
        }
        v2Store128(payload + kTmaQuantHeaderBytes + pack * kCopyPackBytes, encoded_pack.vector);
      }
    }
    v2GroupBarrier(kTmaHandoffBarrier, kTmaQuantThreads);

    const std::uint32_t fetch = index + 2;
    if (v2TmaElected(control) && fetch < queue.tile_count) {
      const V2TmaQuantTile& next = args.tiles[queue.tile_begin + fetch];
      v2TmaIssueLoad(args, next, shared_tiles + static_cast<std::size_t>(stage) * kTmaQuantElements,
                     barriers[stage]);
    }
    const bool packet_end =
      index + 1 == queue.tile_count || args.tiles[queue.tile_begin + index + 1].step != tile.step;
    if (packet_end && control_lane && v2LoadError(&local_header->error) == 0) {
      slot->bytes = tile.payload_offset + static_cast<std::uint32_t>(kTmaQuantPacketBytes);
      v2Publish(&slot->ready_step, tile.step);
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
    if (packet_begin) v2GroupBarrier(kTmaHandoffBarrier, kTmaQuantThreads);
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
      v2GroupBarrier(kTmaHandoffBarrier, kTmaQuantThreads);
      if (control_lane && v2LoadError(&local_header->error) == 0) {
        v2Publish(&slot->consumed_step, tile.step);
      }
    }
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

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

__device__ __forceinline__ bool v2TmaElected() {
  const unsigned int warp = __shfl_sync(0xffffffffU, threadIdx.x / kWarpSize, 0);
  return warp == 0 && ptx::elect_sync(0xffffffffU);
}

__device__ __forceinline__ bool v2TmaWaitForPeers(const V2KernelArgs& args) {
  auto* local_header = reinterpret_cast<V2WindowHeader*>(args.local_window);
  __shared__ int peers_ready;
  if (threadIdx.x == 0) {
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
  __shared__ float warp_max[kTmaQuantThreads / kWarpSize];
  __shared__ float block_scale;
  __shared__ int fifo_ready;
  __shared__ unsigned long long fifo_cache;

  const V2KernelArgs& transport = args.transport;
  auto* local_header = reinterpret_cast<V2WindowHeader*>(transport.local_window);
  if (blockIdx.x == 0 && threadIdx.x == 0) v2Publish(&local_header->epoch, transport.epoch);
  if (threadIdx.x == 0) {
    init(&barriers[0], 1);
    init(&barriers[1], 1);
    fifo_cache = 0;
  }
  __syncthreads();
  if (!v2TmaWaitForPeers(transport)) return;

  const V2TmaChannelQueue queue = args.queues[blockIdx.x];
  const std::uint32_t preload = queue.tile_count < 2 ? queue.tile_count : 2;
  if (v2TmaElected()) {
    for (std::uint32_t stage = 0; stage < preload; ++stage) {
      const V2TmaQuantTile& tile = args.tiles[queue.tile_begin + stage];
      v2TmaIssueLoad(args, tile, shared_tiles + static_cast<std::size_t>(stage) * kTmaQuantElements,
                     barriers[stage]);
    }
  }

  for (std::uint32_t index = 0; index < queue.tile_count; ++index) {
    const std::uint32_t stage = index & 1U;
    const std::uint32_t parity = (index >> 1U) & 1U;
    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                          cuda::device::barrier_native_handle(barriers[stage]), parity)) {
    }
    __syncthreads();

    const V2TmaQuantTile tile = args.tiles[queue.tile_begin + index];
    V2FifoSlot* slot = v2FifoSlot(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true);
    if (threadIdx.x == 0) {
      fifo_ready = v2WaitFree(slot, tile.step, transport.layout.fifo_depth, &fifo_cache, &local_header->error,
                              transport.timeout_cycles);
    }
    __syncthreads();
    if (!fifo_ready) return;

    const auto* shared_tile = shared_tiles + static_cast<std::size_t>(stage) * kTmaQuantElements;
    float local_max = 0.0F;
    for (std::uint32_t element = threadIdx.x; element < kTmaQuantElements; element += blockDim.x) {
      local_max = fmaxf(local_max, fabsf(__bfloat162float(shared_tile[element])));
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffU, local_max, offset));
    }
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) warp_max[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
      local_max = lane < kTmaQuantThreads / kWarpSize ? warp_max[lane] : 0.0F;
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffU, local_max, offset));
      }
      if (lane == 0) block_scale = fmaxf(local_max, 1.0e-4F) / 448.0F;
    }
    __syncthreads();

    auto* payload = v2FifoPayload(transport, transport.local_rank, queue.peer, queue.channel, tile.step, true);
    if (threadIdx.x == 0) {
      reinterpret_cast<V2TmaPacketHeader*>(payload)->scale = block_scale;
      auto* scale = reinterpret_cast<float*>(tile.scale_ptr);
      scale[static_cast<std::uint64_t>(tile.tile_row) * (tile.scale_row_stride / sizeof(float)) + tile.tile_col] =
        block_scale;
    }
    constexpr std::uint32_t pair_count = kTmaQuantElements / 2;
    for (std::uint32_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const float2 values = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(shared_tile)[pair]);
      const float2 scaled = {values.x / block_scale, values.y / block_scale};
      const __nv_fp8x2_e4m3 encoded(scaled);
      v2Store16(payload + kTmaQuantHeaderBytes + pair * 2, encoded.__x);
    }
    __syncthreads();
    if (threadIdx.x == 0 && v2LoadError(&local_header->error) == 0) {
      slot->bytes = static_cast<std::uint32_t>(kTmaQuantPacketBytes);
      v2Publish(&slot->ready_step, tile.step);
    }
    __syncthreads();
    if (v2LoadError(&local_header->error) != 0) return;

    const std::uint32_t fetch = index + 2;
    if (v2TmaElected() && fetch < queue.tile_count) {
      const V2TmaQuantTile& next = args.tiles[queue.tile_begin + fetch];
      v2TmaIssueLoad(args, next, shared_tiles + static_cast<std::size_t>(stage) * kTmaQuantElements,
                     barriers[stage]);
    }
  }

  if (queue.tile_count != 0 && threadIdx.x == 0) {
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
  if (blockIdx.x == 0 && threadIdx.x == 0) v2Publish(&local_header->epoch, transport.epoch);
  if (!v2TmaWaitForPeers(transport)) return;

  __shared__ int fifo_ready;
  __shared__ unsigned long long fifo_cache;
  if (threadIdx.x == 0) fifo_cache = 0;
  __syncthreads();

  const V2TmaChannelQueue queue = args.queues[blockIdx.x];
  for (std::uint32_t index = 0; index < queue.tile_count; ++index) {
    const V2TmaQuantTile tile = args.tiles[queue.tile_begin + index];
    V2FifoSlot* slot = v2FifoSlot(transport, queue.peer, transport.local_rank, queue.channel, tile.step, false);
    if (threadIdx.x == 0) {
      fifo_ready = v2WaitReady(&slot->ready_step, tile.step, &fifo_cache, &local_header->error,
                               transport.timeout_cycles);
      if (fifo_ready && slot->bytes != kTmaQuantPacketBytes) {
        atomicExch_system(&local_header->error, 5U);
        fifo_ready = 0;
      }
    }
    __syncthreads();
    if (!fifo_ready) return;

    const auto* payload = v2FifoPayload(transport, queue.peer, transport.local_rank, queue.channel, tile.step, false);
    if (threadIdx.x == 0) {
      auto* scale = reinterpret_cast<float*>(tile.scale_ptr);
      scale[static_cast<std::uint64_t>(tile.tile_row) * (tile.scale_row_stride / sizeof(float)) + tile.tile_col] =
        reinterpret_cast<const V2TmaPacketHeader*>(payload)->scale;
    }
    auto* tensor = reinterpret_cast<std::uint8_t*>(tile.tensor_ptr);
    constexpr std::uint32_t packs_per_row = kTmaQuantBlockCols / kCopyPackBytes;
    constexpr std::uint32_t pack_count = kTmaQuantBlockRows * packs_per_row;
    for (std::uint32_t pack = threadIdx.x; pack < pack_count; pack += blockDim.x) {
      const std::uint32_t row = pack / packs_per_row;
      const std::uint32_t col_pack = pack - row * packs_per_row;
      const auto value = v2Load128(payload + kTmaQuantHeaderBytes + pack * kCopyPackBytes);
      auto* destination = tensor + static_cast<std::uint64_t>(tile.tile_row * kTmaQuantBlockRows + row) *
                                    tile.tensor_row_stride +
                          static_cast<std::uint64_t>(tile.tile_col * kTmaQuantBlockCols) +
                          col_pack * kCopyPackBytes;
      v2Store128(destination, value);
    }
    __syncthreads();
    if (threadIdx.x == 0 && v2LoadError(&local_header->error) == 0) {
      v2Publish(&slot->consumed_step, tile.step);
    }
    __syncthreads();
    if (v2LoadError(&local_header->error) != 0) return;
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

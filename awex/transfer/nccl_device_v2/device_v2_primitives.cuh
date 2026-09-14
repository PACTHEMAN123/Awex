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

namespace awex {
namespace nccl_device_v2 {

struct alignas(16) V2Pack128 {
  unsigned long long first;
  unsigned long long second;
};

__device__ __forceinline__ unsigned long long v2LoadSystem(volatile unsigned long long* address) {
  return atomicAdd_system(const_cast<unsigned long long*>(address), 0ULL);
}

__device__ __forceinline__ unsigned int v2LoadError(volatile unsigned int* address) {
  return atomicAdd(const_cast<unsigned int*>(address), 0U);
}

__device__ __forceinline__ bool v2WaitReady(volatile unsigned long long* ready, unsigned long long step,
                                            volatile unsigned int* error, unsigned long long timeout_cycles) {
  const unsigned long long start = clock64();
  while (v2LoadSystem(ready) < step) {
    if (v2LoadError(error) != 0) {
      return false;
    }
    if (clock64() - start > timeout_cycles) {
      atomicExch_system(const_cast<unsigned int*>(error), 2U);
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ bool v2WaitFree(volatile V2FifoSlot* slot, unsigned long long step, std::uint32_t fifo_depth,
                                           volatile unsigned int* error, unsigned long long timeout_cycles) {
  const unsigned long long start = clock64();
  // A zeroed slot is free for its first generation. The slot is blocked only
  // once the producer is more than fifo_depth steps ahead of the last
  // consumer acknowledgement; equality is still the initial use of a slot.
  while (v2LoadSystem(&slot->consumed_step) + fifo_depth < step) {
    if (v2LoadError(error) != 0) {
      return false;
    }
    if (clock64() - start > timeout_cycles) {
      atomicExch_system(const_cast<unsigned int*>(error), 3U);
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ bool v2WaitConsumed(volatile V2FifoSlot* slot, unsigned long long step,
                                               volatile unsigned int* error, unsigned long long timeout_cycles) {
  return v2WaitReady(&slot->consumed_step, step, error, timeout_cycles);
}

__device__ __forceinline__ void v2Publish(volatile unsigned long long* address, unsigned long long step) {
  __threadfence_system();
  atomicExch_system(const_cast<unsigned long long*>(address), step);
}

__device__ __forceinline__ void v2GroupBarrier(int group, int nthreads) {
  if (nthreads == kWarpSize) {
    __syncwarp();
  } else {
    // Keep barrier 0 for CTA-wide __syncthreads, as NCCL primitives do.
    const int barrier = 15 - group;
    asm volatile("barrier.sync.aligned %0, %1;" : : "r"(barrier), "r"(nthreads) : "memory");
  }
}

__device__ __forceinline__ V2Pack128 v2Load128(const void* address) {
  V2Pack128 value;
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
               : "=l"(value.first), "=l"(value.second)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ void v2Store128(void* address, const V2Pack128& value) {
  asm volatile("st.global.v2.u64 [%0], {%1,%2};" : : "l"(address), "l"(value.first), "l"(value.second) : "memory");
}

__device__ __forceinline__ V2FifoSlot* v2FifoSlot(const V2KernelArgs& args, std::uint32_t window_rank,
                                                  std::uint32_t connection_rank, std::uint32_t channel,
                                                  unsigned long long step, bool local_window) {
  auto* window = local_window ? args.local_window : reinterpret_cast<std::uint8_t*>(args.peer_windows[window_rank]);
  const std::size_t connection = (static_cast<std::size_t>(connection_rank) * args.layout.channel_count) + channel;
  const std::size_t slot =
    connection * args.layout.fifo_depth + static_cast<std::size_t>(step % args.layout.fifo_depth);
  return reinterpret_cast<V2FifoSlot*>(window + args.layout.state_offset + slot * sizeof(V2FifoSlot));
}

__device__ __forceinline__ std::uint8_t* v2FifoPayload(const V2KernelArgs& args, std::uint32_t window_rank,
                                                       std::uint32_t connection_rank, std::uint32_t channel,
                                                       unsigned long long step, bool local_window) {
  auto* window = local_window ? args.local_window : reinterpret_cast<std::uint8_t*>(args.peer_windows[window_rank]);
  const std::size_t connection = (static_cast<std::size_t>(connection_rank) * args.layout.channel_count) + channel;
  const std::size_t slot =
    connection * args.layout.fifo_depth + static_cast<std::size_t>(step % args.layout.fifo_depth);
  return window + args.layout.payload_offset + slot * args.layout.slot_bytes;
}

__device__ __forceinline__ void v2CopyContiguous(std::uint8_t* destination, const std::uint8_t* source,
                                                 std::uint64_t nbytes, int tid, int nthreads) {
  const std::uintptr_t source_address = reinterpret_cast<std::uintptr_t>(source);
  const std::uintptr_t destination_address = reinterpret_cast<std::uintptr_t>(destination);
  if ((source_address | destination_address) % kCopyPackBytes == 0) {
    const std::uint64_t pack_count = nbytes / kCopyPackBytes;
    const std::uint64_t packs_per_hunk = static_cast<std::uint64_t>(nthreads) * kCopyUnroll;
    const std::uint64_t unrolled_packs = (pack_count / packs_per_hunk) * packs_per_hunk;

    for (std::uint64_t hunk = 0; hunk < unrolled_packs; hunk += packs_per_hunk) {
      V2Pack128 values[kCopyUnroll];
#pragma unroll
      for (int unroll = 0; unroll < kCopyUnroll; ++unroll) {
        const std::uint64_t pack = hunk + tid + static_cast<std::uint64_t>(unroll) * nthreads;
        values[unroll] = v2Load128(source + pack * kCopyPackBytes);
      }
#pragma unroll
      for (int unroll = 0; unroll < kCopyUnroll; ++unroll) {
        const std::uint64_t pack = hunk + tid + static_cast<std::uint64_t>(unroll) * nthreads;
        v2Store128(destination + pack * kCopyPackBytes, values[unroll]);
      }
    }
    for (std::uint64_t pack = unrolled_packs + tid; pack < pack_count; pack += nthreads) {
      v2Store128(destination + pack * kCopyPackBytes, v2Load128(source + pack * kCopyPackBytes));
    }
    const std::uint64_t vector_bytes = pack_count * kCopyPackBytes;
    for (std::uint64_t byte = vector_bytes + tid; byte < nbytes; byte += nthreads) {
      destination[byte] = source[byte];
    }
    return;
  }

  for (std::uint64_t byte = tid; byte < nbytes; byte += nthreads) {
    destination[byte] = source[byte];
  }
}

__device__ __forceinline__ void v2CopyTensorToContiguous(std::uint8_t* destination, const std::uint8_t* tensor,
                                                         std::uint64_t tensor_offset, std::uint64_t nbytes,
                                                         std::uint64_t row_bytes, std::uint64_t row_stride, int tid,
                                                         int nthreads) {
  if (row_bytes == row_stride) {
    v2CopyContiguous(destination, tensor + tensor_offset, nbytes, tid, nthreads);
    return;
  }

  std::uint64_t row = tensor_offset / row_bytes;
  std::uint64_t column = tensor_offset % row_bytes;
  std::uint64_t copied = 0;
  while (copied < nbytes) {
    const std::uint64_t row_remaining = row_bytes - column;
    const std::uint64_t span = row_remaining < nbytes - copied ? row_remaining : nbytes - copied;
    v2CopyContiguous(destination + copied, tensor + row * row_stride + column, span, tid, nthreads);
    copied += span;
    ++row;
    column = 0;
  }
}

__device__ __forceinline__ void v2CopyContiguousToTensor(std::uint8_t* tensor, const std::uint8_t* source,
                                                         std::uint64_t tensor_offset, std::uint64_t nbytes,
                                                         std::uint64_t row_bytes, std::uint64_t row_stride, int tid,
                                                         int nthreads) {
  if (row_bytes == row_stride) {
    v2CopyContiguous(tensor + tensor_offset, source, nbytes, tid, nthreads);
    return;
  }

  std::uint64_t row = tensor_offset / row_bytes;
  std::uint64_t column = tensor_offset % row_bytes;
  std::uint64_t copied = 0;
  while (copied < nbytes) {
    const std::uint64_t row_remaining = row_bytes - column;
    const std::uint64_t span = row_remaining < nbytes - copied ? row_remaining : nbytes - copied;
    v2CopyContiguous(tensor + row * row_stride + column, source + copied, span, tid, nthreads);
    copied += span;
    ++row;
    column = 0;
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

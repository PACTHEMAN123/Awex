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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace awex {
namespace nccl_device_v2 {

struct alignas(16) V2Pack128 {
  unsigned long long first;
  unsigned long long second;
};

enum V2Role : std::uint32_t {
  kRoleWorker = 1U << 0,
  kRoleWaitSend = 1U << 1,
  kRoleWaitRecv = 1U << 2,
  kRolePostSend = 1U << 3,
  kRolePostRecv = 1U << 4,
};

__device__ __forceinline__ unsigned long long v2LoadStep(const volatile unsigned long long* address) {
  unsigned long long value;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ unsigned int v2LoadError(const volatile unsigned int* address) {
  unsigned int value;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void v2FenceSystem() {
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#else
  __threadfence_system();
#endif
}

__device__ __forceinline__ void v2StoreStep(volatile unsigned long long* address, unsigned long long step) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" : : "l"(address), "l"(step) : "memory");
#else
  atomicExch_system(const_cast<unsigned long long*>(address), step);
#endif
}

__device__ __forceinline__ bool v2WaitReady(const volatile unsigned long long* ready, unsigned long long step,
                                            unsigned long long* cache, volatile unsigned int* error,
                                            unsigned long long timeout_cycles) {
  const unsigned long long start = clock64();
  while (*cache < step) {
    *cache = v2LoadStep(ready);
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

__device__ __forceinline__ bool v2WaitFree(const volatile V2FifoSlot* slot, unsigned long long step,
                                           std::uint32_t fifo_depth, unsigned long long* cache,
                                           volatile unsigned int* error, unsigned long long timeout_cycles) {
  const unsigned long long start = clock64();
  // A zeroed slot is free for its first generation. The slot is blocked only
  // once the producer is more than fifo_depth steps ahead of the last
  // consumer acknowledgement; equality is still the initial use of a slot.
  while (*cache + fifo_depth < step) {
    *cache = v2LoadStep(&slot->consumed_step);
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

__device__ __forceinline__ bool v2WaitConsumed(const volatile V2FifoSlot* slot, unsigned long long step,
                                               unsigned long long* cache, volatile unsigned int* error,
                                               unsigned long long timeout_cycles) {
  return v2WaitReady(&slot->consumed_step, step, cache, error, timeout_cycles);
}

__device__ __forceinline__ void v2Publish(volatile unsigned long long* address, unsigned long long step) {
  v2FenceSystem();
  v2StoreStep(address, step);
}

__device__ __forceinline__ void v2GroupBarrier(int barrier, int nthreads) {
  if (nthreads == kWarpSize) {
    __syncwarp();
  } else {
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

__device__ __forceinline__ std::uint8_t v2Load8(const void* address) {
  std::uint32_t value;
  asm volatile("ld.volatile.global.u8 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return static_cast<std::uint8_t>(value);
}

__device__ __forceinline__ void v2Store128(void* address, const V2Pack128& value) {
  asm volatile("st.global.v2.u64 [%0], {%1,%2};" : : "l"(address), "l"(value.first), "l"(value.second) : "memory");
}

__device__ __forceinline__ void v2Store8(void* address, std::uint8_t value) {
  asm volatile("st.global.u8 [%0], %1;" : : "l"(address), "r"(static_cast<std::uint32_t>(value)) : "memory");
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
  const std::uint32_t payload_peer =
    args.payload_peer_slots[static_cast<std::size_t>(window_rank) * args.world_size + connection_rank];
  const std::size_t connection = (static_cast<std::size_t>(payload_peer) * args.layout.channel_count) + channel;
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
      v2Store8(destination + byte, v2Load8(source + byte));
    }
    return;
  }

  for (std::uint64_t byte = tid; byte < nbytes; byte += nthreads) {
    v2Store8(destination + byte, v2Load8(source + byte));
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

struct alignas(4) V2ScalarBytes {
  std::uint8_t bytes[4];
};

__device__ __forceinline__ const std::uint8_t* v2TensorAddress(const std::uint8_t* tensor,
                                                              std::uint64_t logical_offset,
                                                              std::uint64_t row_bytes,
                                                              std::uint64_t row_stride) {
  if (row_bytes == row_stride) return tensor + logical_offset;
  const std::uint64_t row = logical_offset / row_bytes;
  const std::uint64_t column = logical_offset % row_bytes;
  return tensor + row * row_stride + column;
}

__device__ __forceinline__ float v2LoadNumeric(const std::uint8_t* address, V2DataType dtype) {
  switch (dtype) {
    case V2DataType::kFloat16:
      return __half2float(*reinterpret_cast<const __half*>(address));
    case V2DataType::kBFloat16:
      return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(address));
    case V2DataType::kFloat32:
      return *reinterpret_cast<const float*>(address);
    case V2DataType::kFloat8E4M3:
      return static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(address));
    case V2DataType::kFloat8E5M2:
      return static_cast<float>(*reinterpret_cast<const __nv_fp8_e5m2*>(address));
    case V2DataType::kOpaque:
      break;
  }
  return 0.0F;
}

__device__ __forceinline__ V2ScalarBytes v2EncodeNumeric(float value, V2DataType dtype) {
  V2ScalarBytes encoded{};
  switch (dtype) {
    case V2DataType::kFloat16:
      *reinterpret_cast<__half*>(encoded.bytes) = __float2half_rn(value);
      break;
    case V2DataType::kBFloat16:
      *reinterpret_cast<__nv_bfloat16*>(encoded.bytes) = __float2bfloat16_rn(value);
      break;
    case V2DataType::kFloat32:
      *reinterpret_cast<float*>(encoded.bytes) = value;
      break;
    case V2DataType::kFloat8E4M3:
      *reinterpret_cast<__nv_fp8_e4m3*>(encoded.bytes) = __nv_fp8_e4m3(value);
      break;
    case V2DataType::kFloat8E5M2:
      *reinterpret_cast<__nv_fp8_e5m2*>(encoded.bytes) = __nv_fp8_e5m2(value);
      break;
    case V2DataType::kOpaque:
      break;
  }
  return encoded;
}

// Encode directly into a FIFO slice. The slice may begin or end inside one
// wire element, so an element at a step boundary is encoded by both adjacent
// steps and each step publishes only its own bytes.
__device__ __forceinline__ void v2CastTensorToContiguous(
  std::uint8_t* destination, const std::uint8_t* tensor, std::uint64_t tensor_offset,
  std::uint64_t wire_offset, std::uint64_t nbytes, std::uint64_t row_bytes, std::uint64_t row_stride,
  V2DataType tensor_dtype, V2DataType wire_dtype, std::uint32_t tensor_element_bytes,
  std::uint32_t wire_element_bytes, std::uintptr_t quant_scale_ptr, std::uint64_t quant_rows,
  std::uint64_t quant_cols, std::uint64_t quant_row_offset, std::uint64_t quant_col_offset,
  std::uint64_t quant_scale_row_stride, V2QuantMode quant_mode, std::uint32_t quant_block_rows,
  std::uint32_t quant_block_cols, int tid, int nthreads) {
  const std::uint64_t wire_end = wire_offset + nbytes;
  const std::uint64_t first_element = wire_offset / wire_element_bytes;
  const std::uint64_t element_end = (wire_end + wire_element_bytes - 1) / wire_element_bytes;
  for (std::uint64_t element = first_element + tid; element < element_end; element += nthreads) {
    const std::uint64_t logical_offset = tensor_offset + element * tensor_element_bytes;
    const auto* source = v2TensorAddress(tensor, logical_offset, row_bytes, row_stride);
    float value = v2LoadNumeric(source, tensor_dtype);
    if (quant_mode == V2QuantMode::kBlockwiseFloat8E4M3) {
      const std::uint64_t row = element / quant_cols;
      const std::uint64_t col = element % quant_cols;
      if (row < quant_rows) {
        const std::uint64_t scale_row = (quant_row_offset + row) / quant_block_rows;
        const std::uint64_t scale_col = (quant_col_offset + col) / quant_block_cols;
        const auto* scales = reinterpret_cast<const float*>(quant_scale_ptr);
        value /= scales[scale_row * quant_scale_row_stride + scale_col];
      }
    }
    const V2ScalarBytes encoded = v2EncodeNumeric(value, wire_dtype);
    const std::uint64_t element_wire_begin = element * wire_element_bytes;
    const std::uint64_t begin = element_wire_begin < wire_offset ? wire_offset : element_wire_begin;
    const std::uint64_t element_wire_end = element_wire_begin + wire_element_bytes;
    const std::uint64_t end = element_wire_end < wire_end ? element_wire_end : wire_end;
    for (std::uint64_t byte = begin; byte < end; ++byte) {
      v2Store8(destination + byte - wire_offset, encoded.bytes[byte - element_wire_begin]);
    }
  }
}

__device__ __forceinline__ void v2CopyFragmentsToContiguous(const V2KernelArgs& args, const V2Work& work,
                                                            std::uint8_t* destination, std::uint64_t work_offset,
                                                            std::uint64_t nbytes, int tid, int nthreads) {
  const std::uint64_t copy_end = work_offset + nbytes;
  for (std::uint32_t index = 0; index < work.fragment_count; ++index) {
    const V2Fragment& fragment = args.fragments[work.fragment_begin + index];
    const std::uint64_t fragment_end = fragment.work_offset + fragment.nbytes;
    if (fragment_end <= work_offset || copy_end <= fragment.work_offset) continue;
    const std::uint64_t begin = fragment.work_offset < work_offset ? work_offset : fragment.work_offset;
    const std::uint64_t end = fragment_end < copy_end ? fragment_end : copy_end;
    const auto* tensor = reinterpret_cast<const std::uint8_t*>(fragment.tensor_ptr);
    const std::uint64_t wire_offset = fragment.wire_offset + begin - fragment.work_offset;
    if (fragment.tensor_dtype == fragment.wire_dtype &&
        fragment.tensor_element_bytes == fragment.wire_element_bytes) {
      v2CopyTensorToContiguous(destination + begin - work_offset, tensor, fragment.tensor_offset + wire_offset,
                               end - begin, fragment.tensor_row_bytes, fragment.tensor_row_stride, tid, nthreads);
    } else {
      v2CastTensorToContiguous(destination + begin - work_offset, tensor, fragment.tensor_offset, wire_offset,
                               end - begin, fragment.tensor_row_bytes, fragment.tensor_row_stride,
                               fragment.tensor_dtype, fragment.wire_dtype, fragment.tensor_element_bytes,
                               fragment.wire_element_bytes, fragment.quant_scale_ptr, fragment.quant_rows,
                               fragment.quant_cols, fragment.quant_row_offset, fragment.quant_col_offset,
                               fragment.quant_scale_row_stride, fragment.quant_mode, fragment.quant_block_rows,
                               fragment.quant_block_cols, tid, nthreads);
    }
  }
}

__device__ __forceinline__ void v2CopyContiguousToFragments(const V2KernelArgs& args, const V2Work& work,
                                                            const std::uint8_t* source, std::uint64_t work_offset,
                                                            std::uint64_t nbytes, int tid, int nthreads) {
  const std::uint64_t copy_end = work_offset + nbytes;
  for (std::uint32_t index = 0; index < work.fragment_count; ++index) {
    const V2Fragment& fragment = args.fragments[work.fragment_begin + index];
    const std::uint64_t fragment_end = fragment.work_offset + fragment.nbytes;
    if (fragment_end <= work_offset || copy_end <= fragment.work_offset) continue;
    const std::uint64_t begin = fragment.work_offset < work_offset ? work_offset : fragment.work_offset;
    const std::uint64_t end = fragment_end < copy_end ? fragment_end : copy_end;
    auto* tensor = reinterpret_cast<std::uint8_t*>(fragment.tensor_ptr);
    const std::uint64_t wire_offset = fragment.wire_offset + begin - fragment.work_offset;
    v2CopyContiguousToTensor(tensor, source + begin - work_offset,
                             fragment.tensor_offset + wire_offset, end - begin, fragment.tensor_row_bytes,
                             fragment.tensor_row_stride, tid, nthreads);
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

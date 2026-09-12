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

#include "device_transfer_types.cuh"

namespace awex {
namespace device_transfer {

__device__ __forceinline__ unsigned long long global_timer_ns() {
#if defined(__CUDA_ARCH__)
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
#else
  return 0;
#endif
}

// Keep the data mover behind one primitive so a Hopper TMA implementation can
// replace it without changing slot ownership or completion semantics.
__device__ __forceinline__ void copy_contiguous_1d(
    std::uint8_t* __restrict__ destination,
    const std::uint8_t* __restrict__ source,
    std::uint64_t nbytes) {
  const auto address_bits = reinterpret_cast<std::uintptr_t>(destination) |
      reinterpret_cast<std::uintptr_t>(source);
  if ((address_bits & (kVectorBytes - 1)) == 0) {
    auto* destination_vectors = reinterpret_cast<uint4*>(destination);
    const auto* source_vectors = reinterpret_cast<const uint4*>(source);
    const std::uint64_t vector_count = nbytes / kVectorBytes;
    for (std::uint64_t vector = threadIdx.x; vector < vector_count;
         vector += blockDim.x) {
      destination_vectors[vector] = source_vectors[vector];
    }

    const std::uint64_t tail_offset = vector_count * kVectorBytes;
    for (std::uint64_t byte = tail_offset + threadIdx.x; byte < nbytes;
         byte += blockDim.x) {
      destination[byte] = source[byte];
    }
    return;
  }

  for (std::uint64_t byte = threadIdx.x; byte < nbytes; byte += blockDim.x) {
    destination[byte] = source[byte];
  }
}

__device__ __forceinline__ void fence_proxy_alias() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("fence.proxy.alias;" ::: "memory");
#endif
}

__device__ __forceinline__ void multimem_store_u32(
    std::uint8_t* destination,
    std::uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "multimem.st.weak.global.b32 [%0], %1;"
      :
      : "l"(destination), "r"(value)
      : "memory");
#else
  (void)destination;
  (void)value;
#endif
}

__device__ __forceinline__ void multimem_store_release_u64(
    unsigned long long* destination,
    unsigned long long value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "multimem.st.release.sys.global.u64 [%0], %1;"
      :
      : "l"(destination), "l"(value)
      : "memory");
#else
  (void)destination;
  (void)value;
#endif
}

// A multimem address may only be accessed by multimem instructions. Keep the
// primitive here so the transfer protocol never accidentally uses a regular
// CUDA store on the multicast mapping.
__device__ __forceinline__ void copy_contiguous_1d_multicast(
    std::uint8_t* destination,
    const std::uint8_t* source,
    std::uint64_t nbytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const auto address_bits = reinterpret_cast<std::uintptr_t>(destination) |
      reinterpret_cast<std::uintptr_t>(source);
  std::uint64_t copied = 0;
  if ((address_bits & (kVectorBytes - 1)) == 0) {
    const auto* source_vectors = reinterpret_cast<const uint4*>(source);
    const std::uint64_t vector_count = nbytes / kVectorBytes;
    for (std::uint64_t vector = threadIdx.x; vector < vector_count;
         vector += blockDim.x) {
      const uint4 value = source_vectors[vector];
      auto* vector_destination = destination + vector * kVectorBytes;
      multimem_store_u32(vector_destination, value.x);
      multimem_store_u32(vector_destination + 4, value.y);
      multimem_store_u32(vector_destination + 8, value.z);
      multimem_store_u32(vector_destination + 12, value.w);
    }
    copied = vector_count * kVectorBytes;
  }

  // Pack the uncommon unaligned path and the tail into 32-bit stores. The
  // slot has room through round_up(nbytes, 4), so padding never crosses it.
  const std::uint64_t word_count = (nbytes - copied + 3) / 4;
  for (std::uint64_t word = threadIdx.x; word < word_count;
       word += blockDim.x) {
    const std::uint64_t offset = copied + word * 4;
    std::uint32_t value = 0;
    auto* bytes = reinterpret_cast<std::uint8_t*>(&value);
#pragma unroll
    for (std::uint64_t byte = 0; byte < 4; ++byte) {
      if (offset + byte < nbytes) {
        bytes[byte] = source[offset + byte];
      }
    }
    multimem_store_u32(destination + offset, value);
  }
#else
  (void)destination;
  (void)source;
  (void)nbytes;
#endif
}

}  // namespace device_transfer
}  // namespace awex

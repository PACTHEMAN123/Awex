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

}  // namespace device_transfer
}  // namespace awex

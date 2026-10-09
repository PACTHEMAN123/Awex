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

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "types.cuh"

namespace shardstream {
namespace transport {

// Each record is [FP32 dequantization scale, 12 zero bytes, E4M3 tile].
// It lives only in the existing FIFO. A single warp owns one quantization
// block, even when lowering/FIFO steps split the record. No CTA barrier,
// full-weight staging allocation, or separate quantization launch is needed.
template <int BlockRows, int BlockCols>
__device__ __forceinline__ void quantizeToFifoImpl(
  const Fragment& f, std::uint8_t* destination, std::uint64_t offset,
  std::uint64_t nbytes, int tid, int nthreads) {
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int warps = nthreads / kWarpSize;
  constexpr std::uint32_t elements = BlockRows * BlockCols;
  constexpr std::uint32_t record = elements + 16;
  const std::uint64_t first = offset / record;
  const std::uint64_t last = (offset + nbytes + record - 1) / record;
  const std::uint32_t block_columns = f.tensor_row_bytes / (2 * BlockCols);
  const auto* base = reinterpret_cast<const std::uint8_t*>(f.tensor_ptr);
  for (std::uint64_t tile = first + warp; tile < last; tile += warps) {
    const auto* matrix = base + (static_cast<std::uint32_t>(tile) / block_columns) * BlockRows * f.tensor_row_stride
                             + (static_cast<std::uint32_t>(tile) % block_columns) * BlockCols * 2;
    // Finite BF16 magnitudes preserve unsigned-bit ordering. Accumulate two
    // lanes at once instead of converting all 16K inputs to FP32 for amax.
    // Independent accumulators expose integer-ALU parallelism even when a
    // FIFO step has too few tiles to occupy every copy warp. A serial chain
    // of four dependent vmax operations otherwise stalls each vector load.
    unsigned int maximum_x = 0, maximum_y = 0, maximum_z = 0, maximum_w = 0;
    for (std::uint32_t i = lane * 8; i < elements; i += kWarpSize * 8) {
      const auto* address = matrix + (i / BlockCols) * f.tensor_row_stride + (i % BlockCols) * 2;
      const uint4 packed = *reinterpret_cast<const uint4*>(address);
      maximum_x = __vmaxu2(maximum_x, packed.x & 0x7fff7fffU);
      maximum_y = __vmaxu2(maximum_y, packed.y & 0x7fff7fffU);
      maximum_z = __vmaxu2(maximum_z, packed.z & 0x7fff7fffU);
      maximum_w = __vmaxu2(maximum_w, packed.w & 0x7fff7fffU);
    }
    const unsigned int packed_maximum = __vmaxu2(
      __vmaxu2(maximum_x, maximum_y), __vmaxu2(maximum_z, maximum_w));
    unsigned int maximum_bits = max(packed_maximum & 0xffffU, packed_maximum >> 16);
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      maximum_bits = max(maximum_bits, __shfl_xor_sync(0xffffffff, maximum_bits, delta));
    const float maximum = __uint_as_float(maximum_bits << 16);
    // Match the finite E4M3 block reference (epsilon prevents zero/denormal
    // scale division). The receiver consumes this scale without recomputing it.
    const float scale = fmaxf(maximum, 1.0e-12f) / 448.0f;
    const float inverse = 1.0f / scale;
    const std::uint64_t record_begin = tile * record;
    for (int i = lane; i < 16; i += kWarpSize) {
      const std::uint64_t wire = record_begin + i;
      if (wire >= offset && wire < offset + nbytes)
        destination[wire - offset] = i < 4 ? reinterpret_cast<const std::uint8_t*>(&scale)[i] : 0;
    }
    for (std::uint32_t i = lane * 8; i < elements; i += kWarpSize * 8) {
      const std::uint64_t wire = record_begin + 16 + i;
      if (wire + 8 <= offset || wire >= offset + nbytes) continue;
      const auto* address = matrix + (i / BlockCols) * f.tensor_row_stride + (i % BlockCols) * 2;
      const uint4 packed = *reinterpret_cast<const uint4*>(address);
      const auto* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
      unsigned long long encoded = 0;
      #pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 scaled = make_float2(__bfloat162float(values[pair * 2]) * inverse,
                                          __bfloat162float(values[pair * 2 + 1]) * inverse);
        encoded |= static_cast<unsigned long long>(__nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3))
                   << (pair * 16);
      }
      if (wire >= offset && wire + 8 <= offset + nbytes && (wire - offset) % 8 == 0) {
        *reinterpret_cast<unsigned long long*>(destination + wire - offset) = encoded;
      } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
          if (wire + j >= offset && wire + j < offset + nbytes)
            destination[wire + j - offset] = static_cast<std::uint8_t>(encoded >> (j * 8));
      }
    }
  }
}

template <int BlockRows, int BlockCols>
__device__ __forceinline__ void scatterFp8FromFifoImpl(
  const Fragment& f, const std::uint8_t* source, std::uint8_t* forward,
  std::uint64_t offset, std::uint64_t nbytes, int tid, int nthreads) {
  constexpr std::uint32_t elements = BlockRows * BlockCols;
  constexpr std::uint32_t record = elements + 16;
  const std::uint32_t block_columns = f.tensor_row_bytes / BlockCols;
  auto* base = reinterpret_cast<std::uint8_t*>(f.tensor_ptr);
  auto* scales = reinterpret_cast<std::uint8_t*>(f.scale_ptr);
  if (offset % 16 == 0 && nbytes % 16 == 0 &&
      reinterpret_cast<std::uintptr_t>(source) % 16 == 0 &&
      reinterpret_cast<std::uintptr_t>(base) % 16 == 0 && f.tensor_row_stride % 16 == 0 &&
      (forward == nullptr || reinterpret_cast<std::uintptr_t>(forward) % 16 == 0)) {
    // Stripe vectors over every worker. Whole-tile ownership leaves workers
    // idle when a step contains fewer tiles than the copy group's warp count.
    for (std::uint64_t i = tid * 16; i < nbytes; i += nthreads * 16) {
      const std::uint64_t wire = offset + i;
      const std::uint32_t tile = wire / record;
      const std::uint32_t within = wire % record;
      const uint4 value = *reinterpret_cast<const uint4*>(source + i);
      if (forward != nullptr) *reinterpret_cast<uint4*>(forward + i) = value;
      if (within == 0) {
        *reinterpret_cast<unsigned int*>(scales + (tile / block_columns) * f.scale_row_stride
                                         + (tile % block_columns) * 4) = value.x;
      } else {
        const std::uint32_t element = within - 16;
        auto* address = base + ((tile / block_columns) * BlockRows + element / BlockCols) * f.tensor_row_stride
                              + (tile % block_columns) * BlockCols + element % BlockCols;
        *reinterpret_cast<uint4*>(address) = value;
      }
    }
    return;
  }
  // The relay forwards the exact FIFO bytes, including scales, before giving
  // the existing credit back; it never dequantizes or requantizes the payload.
  for (std::uint64_t i = tid; i < nbytes; i += nthreads) {
    const std::uint64_t wire = offset + i;
    const std::uint32_t tile = wire / record;
    const std::uint64_t within = wire % record;
    const std::uint8_t value = source[i];
    if (forward != nullptr) forward[i] = value;
    if (within < 4) {
      scales[(tile / block_columns) * f.scale_row_stride + (tile % block_columns) * 4 + within] = value;
    } else if (within >= 16) {
      const std::uint64_t element = within - 16;
      base[((tile / block_columns) * BlockRows + element / BlockCols) * f.tensor_row_stride
           + (tile % block_columns) * BlockCols + element % BlockCols] = value;
    }
  }
}

__device__ __forceinline__ void quantizeToFifo(
  const Fragment& f, std::uint8_t* destination, std::uint64_t offset,
  std::uint64_t nbytes, int tid, int nthreads) {
  if (f.block_rows == 128 && f.block_cols == 128)
    quantizeToFifoImpl<128, 128>(f, destination, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64 && f.block_cols == 64)
    quantizeToFifoImpl<64, 64>(f, destination, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64)
    quantizeToFifoImpl<64, 128>(f, destination, offset, nbytes, tid, nthreads);
  else
    quantizeToFifoImpl<128, 64>(f, destination, offset, nbytes, tid, nthreads);
}

__device__ __forceinline__ void scatterFp8FromFifo(
  const Fragment& f, const std::uint8_t* source, std::uint8_t* forward,
  std::uint64_t offset, std::uint64_t nbytes, int tid, int nthreads) {
  if (f.block_rows == 128 && f.block_cols == 128)
    scatterFp8FromFifoImpl<128, 128>(f, source, forward, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64 && f.block_cols == 64)
    scatterFp8FromFifoImpl<64, 64>(f, source, forward, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64)
    scatterFp8FromFifoImpl<64, 128>(f, source, forward, offset, nbytes, tid, nthreads);
  else
    scatterFp8FromFifoImpl<128, 64>(f, source, forward, offset, nbytes, tid, nthreads);
}

}  // namespace transport
}  // namespace shardstream

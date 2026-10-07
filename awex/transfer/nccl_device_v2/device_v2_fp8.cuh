#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "device_v2_types.cuh"

namespace awex {
namespace nccl_device_v2 {

// Each record is [FP32 dequantization scale, 12 zero bytes, E4M3 tile].
// It lives only in the existing FIFO. A single warp owns one quantization
// block, even when lowering/FIFO steps split the record. No CTA barrier,
// full-weight staging allocation, or separate quantization launch is needed.
template <int BlockRows, int BlockCols>
__device__ __forceinline__ void v2QuantizeToFifoImpl(
  const V2Fragment& f, std::uint8_t* destination, std::uint64_t offset,
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
    unsigned int packed_maximum = 0;
    for (std::uint32_t i = lane * 8; i < elements; i += kWarpSize * 8) {
      const auto* address = matrix + (i / BlockCols) * f.tensor_row_stride + (i % BlockCols) * 2;
      const uint4 packed = *reinterpret_cast<const uint4*>(address);
      packed_maximum = __vmaxu2(packed_maximum, packed.x & 0x7fff7fffU);
      packed_maximum = __vmaxu2(packed_maximum, packed.y & 0x7fff7fffU);
      packed_maximum = __vmaxu2(packed_maximum, packed.z & 0x7fff7fffU);
      packed_maximum = __vmaxu2(packed_maximum, packed.w & 0x7fff7fffU);
    }
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
__device__ __forceinline__ void v2ScatterFp8FromFifoImpl(
  const V2Fragment& f, const std::uint8_t* source, std::uint8_t* forward,
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
    // Assign a tile to a warp, calculating its matrix/scale address once.
    // The byte-striped scatter paid a runtime block-column division for every
    // 16-byte vector. All row/column arithmetic below is compile-time shifts.
    const int lane = tid % kWarpSize;
    const int warp = tid / kWarpSize;
    const int warps = nthreads / kWarpSize;
    const std::uint32_t first = offset / record;
    const std::uint32_t last = (offset + nbytes + record - 1) / record;
    for (std::uint32_t tile = first + warp; tile < last; tile += warps) {
      const std::uint32_t tile_row = tile / block_columns;
      const std::uint32_t tile_column = tile % block_columns;
      auto* matrix = base + tile_row * BlockRows * f.tensor_row_stride + tile_column * BlockCols;
      const std::uint64_t record_begin = static_cast<std::uint64_t>(tile) * record;
      if (lane == 0 && record_begin >= offset && record_begin + 16 <= offset + nbytes) {
        const uint4 header = *reinterpret_cast<const uint4*>(source + record_begin - offset);
        if (forward != nullptr) *reinterpret_cast<uint4*>(forward + record_begin - offset) = header;
        *reinterpret_cast<unsigned int*>(scales + tile_row * f.scale_row_stride + tile_column * 4) = header.x;
      }
      for (std::uint32_t i = lane * 16; i < elements; i += kWarpSize * 16) {
        const std::uint64_t wire = record_begin + 16 + i;
        if (wire < offset || wire + 16 > offset + nbytes) continue;
        const uint4 value = *reinterpret_cast<const uint4*>(source + wire - offset);
        if (forward != nullptr) *reinterpret_cast<uint4*>(forward + wire - offset) = value;
        auto* address = matrix + (i / BlockCols) * f.tensor_row_stride + i % BlockCols;
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

__device__ __forceinline__ void v2QuantizeToFifo(
  const V2Fragment& f, std::uint8_t* destination, std::uint64_t offset,
  std::uint64_t nbytes, int tid, int nthreads) {
  if (f.block_rows == 128 && f.block_cols == 128)
    v2QuantizeToFifoImpl<128, 128>(f, destination, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64 && f.block_cols == 64)
    v2QuantizeToFifoImpl<64, 64>(f, destination, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64)
    v2QuantizeToFifoImpl<64, 128>(f, destination, offset, nbytes, tid, nthreads);
  else
    v2QuantizeToFifoImpl<128, 64>(f, destination, offset, nbytes, tid, nthreads);
}

__device__ __forceinline__ void v2ScatterFp8FromFifo(
  const V2Fragment& f, const std::uint8_t* source, std::uint8_t* forward,
  std::uint64_t offset, std::uint64_t nbytes, int tid, int nthreads) {
  if (f.block_rows == 128 && f.block_cols == 128)
    v2ScatterFp8FromFifoImpl<128, 128>(f, source, forward, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64 && f.block_cols == 64)
    v2ScatterFp8FromFifoImpl<64, 64>(f, source, forward, offset, nbytes, tid, nthreads);
  else if (f.block_rows == 64)
    v2ScatterFp8FromFifoImpl<64, 128>(f, source, forward, offset, nbytes, tid, nthreads);
  else
    v2ScatterFp8FromFifoImpl<128, 64>(f, source, forward, offset, nbytes, tid, nthreads);
}

}  // namespace nccl_device_v2
}  // namespace awex

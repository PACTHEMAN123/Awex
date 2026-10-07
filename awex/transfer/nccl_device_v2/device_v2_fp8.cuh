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
__device__ __forceinline__ void v2QuantizeToFifo(
  const V2Fragment& f, std::uint8_t* destination, std::uint64_t offset,
  std::uint64_t nbytes, int tid, int nthreads) {
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int warps = nthreads / kWarpSize;
  const std::uint64_t elements = f.block_rows * f.block_cols;
  const std::uint64_t record = elements + 16;
  const std::uint64_t first = offset / record;
  const std::uint64_t last = (offset + nbytes + record - 1) / record;
  const std::uint64_t block_columns = f.tensor_row_bytes / (2 * f.block_cols);
  const auto* base = reinterpret_cast<const std::uint8_t*>(f.tensor_ptr);
  for (std::uint64_t tile = first + warp; tile < last; tile += warps) {
    const auto* matrix = base + (tile / block_columns) * f.block_rows * f.tensor_row_stride
                             + (tile % block_columns) * f.block_cols * 2;
    float maximum = 0.0f;
    for (std::uint32_t i = lane * 8; i < elements; i += kWarpSize * 8) {
      const auto* address = matrix + (i / f.block_cols) * f.tensor_row_stride + (i % f.block_cols) * 2;
      const uint4 packed = *reinterpret_cast<const uint4*>(address);
      const auto* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        maximum = fmaxf(maximum, fabsf(__bfloat162float(values[j])));
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
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
      const auto* address = matrix + (i / f.block_cols) * f.tensor_row_stride + (i % f.block_cols) * 2;
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

__device__ __forceinline__ void v2ScatterFp8FromFifo(
  const V2Fragment& f, const std::uint8_t* source, std::uint8_t* forward,
  std::uint64_t offset, std::uint64_t nbytes, int tid, int nthreads) {
  const std::uint64_t elements = f.block_rows * f.block_cols;
  const std::uint64_t record = elements + 16;
  const std::uint64_t block_columns = f.tensor_row_bytes / f.block_cols;
  auto* base = reinterpret_cast<std::uint8_t*>(f.tensor_ptr);
  auto* scales = reinterpret_cast<std::uint8_t*>(f.scale_ptr);
  if (offset % 16 == 0 && nbytes % 16 == 0 &&
      reinterpret_cast<std::uintptr_t>(source) % 16 == 0 &&
      reinterpret_cast<std::uintptr_t>(base) % 16 == 0 && f.tensor_row_stride % 16 == 0 &&
      (forward == nullptr || reinterpret_cast<std::uintptr_t>(forward) % 16 == 0)) {
    for (std::uint64_t i = tid * 16; i < nbytes; i += nthreads * 16) {
      const std::uint64_t wire = offset + i;
      const std::uint64_t tile = wire / record;
      const std::uint32_t within = wire % record;
      const uint4 value = *reinterpret_cast<const uint4*>(source + i);
      if (forward != nullptr) *reinterpret_cast<uint4*>(forward + i) = value;
      if (within == 0) {
        *reinterpret_cast<unsigned int*>(scales + (tile / block_columns) * f.scale_row_stride
                                         + (tile % block_columns) * 4) = value.x;
      } else {
        const std::uint32_t element = within - 16;
        auto* address = base + ((tile / block_columns) * f.block_rows + element / f.block_cols) * f.tensor_row_stride
                              + (tile % block_columns) * f.block_cols + element % f.block_cols;
        *reinterpret_cast<uint4*>(address) = value;
      }
    }
    return;
  }
  // The relay forwards the exact FIFO bytes, including scales, before giving
  // the existing credit back; it never dequantizes or requantizes the payload.
  for (std::uint64_t i = tid; i < nbytes; i += nthreads) {
    const std::uint64_t wire = offset + i;
    const std::uint64_t tile = wire / record;
    const std::uint64_t within = wire % record;
    const std::uint8_t value = source[i];
    if (forward != nullptr) forward[i] = value;
    if (within < 4) {
      scales[(tile / block_columns) * f.scale_row_stride + (tile % block_columns) * 4 + within] = value;
    } else if (within >= 16) {
      const std::uint64_t element = within - 16;
      base[((tile / block_columns) * f.block_rows + element / f.block_cols) * f.tensor_row_stride
           + (tile % block_columns) * f.block_cols + element % f.block_cols] = value;
    }
  }
}

}  // namespace nccl_device_v2
}  // namespace awex

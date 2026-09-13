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
  while (v2LoadSystem(&slot->consumed_step) + fifo_depth <= step) {
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
  for (std::uint64_t byte = static_cast<std::uint64_t>(tid); byte < nbytes;
       byte += static_cast<std::uint64_t>(nthreads)) {
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

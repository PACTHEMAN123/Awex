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

#include "device_v2_kernel.cuh"
#include "device_v2_launch.cuh"
#include "device_v2_tma.cuh"

namespace awex {
namespace nccl_device_v2 {

cudaError_t launchDeviceV2(const V2KernelArgs& args, cudaStream_t stream) {
  if (args.channel_count == 0) {
    return cudaSuccess;
  }
  device_v2_kernel<<<args.channel_count, kThreadsPerBlock, 0, stream>>>(args);
  return cudaGetLastError();
}

cudaError_t launchDeviceV2Tma(const V2TmaKernelArgs& args, V2Direction direction, cudaStream_t stream) {
  if (args.queue_count == 0) {
    return cudaSuccess;
  }
  if (direction == V2Direction::kSend) {
    constexpr int shared_bytes = 2 * kTmaQuantElements * sizeof(__nv_bfloat16);
    cudaError_t result = cudaFuncSetAttribute(tma_quant_send_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             shared_bytes);
    if (result != cudaSuccess) return result;
    tma_quant_send_kernel<<<args.queue_count, kTmaQuantThreads, shared_bytes, stream>>>(args);
  } else {
    tma_quant_recv_kernel<<<args.queue_count, kTmaQuantThreads, 0, stream>>>(args);
  }
  return cudaGetLastError();
}

cudaError_t launchBlockwiseFp8Scales(const V2QuantMatrix* matrices, const V2QuantBlock* blocks,
                                     std::uint32_t block_count, cudaStream_t stream) {
  if (block_count == 0) {
    return cudaSuccess;
  }
  blockwise_fp8_scale_kernel<<<block_count, 256, 0, stream>>>(matrices, blocks, block_count);
  return cudaGetLastError();
}

}  // namespace nccl_device_v2
}  // namespace awex

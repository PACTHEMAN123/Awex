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

namespace awex {
namespace nccl_device_v2 {

cudaError_t launchDeviceV2(const V2KernelArgs& args, cudaStream_t stream) {
  if (args.channel_count == 0) {
    return cudaSuccess;
  }
  device_v2_kernel<<<args.channel_count, kThreadsPerBlock, 0, stream>>>(args);
  return cudaGetLastError();
}

cudaError_t launchBlockwiseFp8Scales(const V2QuantMatrix* matrices, std::uint32_t matrix_count,
                                     std::uint32_t max_blocks, cudaStream_t stream) {
  if (matrix_count == 0 || max_blocks == 0) {
    return cudaSuccess;
  }
  const dim3 grid(max_blocks, matrix_count);
  blockwise_fp8_scale_kernel<<<grid, 256, 0, stream>>>(matrices, matrix_count, max_blocks);
  return cudaGetLastError();
}

}  // namespace nccl_device_v2
}  // namespace awex

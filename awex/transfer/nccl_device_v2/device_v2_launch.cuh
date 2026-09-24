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

cudaError_t launchDeviceV2(const V2KernelArgs& args, cudaStream_t stream);
cudaError_t launchDeviceV2Tma(const V2TmaKernelArgs& args, V2Direction direction, cudaStream_t stream);
cudaError_t launchBlockwiseFp8Scales(const V2QuantMatrix* matrices, const V2QuantBlock* blocks,
                                     std::uint32_t block_count, cudaStream_t stream);

}  // namespace nccl_device_v2
}  // namespace awex

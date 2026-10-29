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

#include "kernel.cuh"
#include "launch.cuh"

namespace shardstream {
namespace transport {

// Each job covers whole records of a unique source view. A single launch
// materializes every quantized view before any transport CTA can read it.
__global__ void materializeFp8Kernel(const Fragment* jobs) {
  const Fragment& job = jobs[blockIdx.x];
  quantizeToFifo(job, reinterpret_cast<std::uint8_t*>(job.staged_ptr) + job.tensor_offset,
                job.tensor_offset, job.nbytes, threadIdx.x, blockDim.x);
}

cudaError_t launchQuantize(const Fragment* jobs, std::uint32_t job_count, cudaStream_t stream) {
  if (job_count == 0) return cudaSuccess;
  materializeFp8Kernel<<<job_count, 256, 0, stream>>>(jobs);
  return cudaGetLastError();
}

cudaError_t launchReset(const KernelArgs& args, cudaStream_t stream) {
  if (args.gin_enabled == 0) return cudaSuccess;
#if SHARDSTREAM_HAS_GIN
  ginResetSignalsKernel<<<1, 256, 0, stream>>>(args);
  return cudaGetLastError();
#else
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch(const KernelArgs& args, cudaStream_t stream) {
  if (args.channel_count == 0) {
    return cudaSuccess;
  }
  transport_kernel<<<args.channel_count, kThreadsPerBlock, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace transport
}  // namespace shardstream

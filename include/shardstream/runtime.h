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
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace shardstream {
// Tensor memory remains owned by the caller and must outlive prepare/launch.
enum class ScalarType { Other, BFloat16, Float8E4M3 };
struct TensorView {
  void* pointer = nullptr;
  int device = -1;
  ScalarType dtype = ScalarType::Other;
  int64_t item_bytes = 0;
  int64_t elements = 0;
  int64_t rank = 0;
  std::array<int64_t, 2> dimensions{};
  std::array<int64_t, 2> strides{};
  bool is_cuda() const { return device >= 0; }
  int get_device() const { return device; }
  void* data_ptr() const { return pointer; }
  int64_t dim() const { return rank; }
  int64_t size(int axis) const { return dimensions.at(axis); }
  int64_t stride(int axis) const { return strides.at(axis); }
  int64_t element_size() const { return item_bytes; }
  int64_t numel() const { return elements; }
  ScalarType scalar_type() const { return dtype; }
};
using Metric = std::variant<int64_t, double, bool, std::vector<int64_t>, std::map<std::string, double>>;
using Metrics = std::map<std::string, Metric>;
std::string get_unique_id();
int64_t create(const std::string& id, int world_size, int rank, int device,
               int timeout_ms, int max_channels, int fifo_depth, int64_t step_bytes,
               int64_t chunk_bytes, int gin_fifo_depth, int64_t network_step_bytes,
               int64_t gin_chunk_bytes, int gin_context_count,
               const std::vector<int64_t>& logical_to_communicator);
void destroy(int64_t handle);
int device(int64_t handle);
Metrics launch(int64_t handle, const std::vector<TensorView>& tensors, const std::vector<int64_t>& lengths,
                const std::vector<int64_t>& tensor_offsets, const std::vector<int64_t>& tensor_row_bytes,
                const std::vector<int64_t>& tensor_row_strides, const std::vector<int64_t>& peers,
                const std::vector<int64_t>& ordinals, const std::vector<int64_t>& expected_counts,
                const std::vector<int64_t>& forward_peers, const std::vector<int64_t>& ring_ids, bool sender,
                int64_t sequence, const std::vector<std::vector<int64_t>>& quantization,
                bool prepare_only, cudaStream_t stream);

}  // namespace shardstream

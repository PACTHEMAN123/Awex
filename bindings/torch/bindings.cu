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

#include <shardstream/runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
namespace py = pybind11;
py::dict launch(int64_t handle, const py::list& tensors, const std::vector<int64_t>& lengths,
                const std::vector<int64_t>& tensor_offsets, const std::vector<int64_t>& tensor_row_bytes,
                const std::vector<int64_t>& tensor_row_strides, const std::vector<int64_t>& peers,
                const std::vector<int64_t>& ordinals, const std::vector<int64_t>& expected_counts,
                const std::vector<int64_t>& forward_peers, const std::vector<int64_t>& ring_ids, bool sender,
                int64_t sequence, const std::vector<std::vector<int64_t>>& quantization = {},
                bool prepare_only = false) {

  std::vector<shardstream::TensorView> views;
  views.reserve(tensors.size());
  int device = -1;
  for (const auto& object : tensors) {
    const auto tensor = object.cast<torch::Tensor>();
    shardstream::TensorView view;
    view.pointer = tensor.data_ptr();
    view.device = tensor.is_cuda() ? tensor.get_device() : -1;
    if (tensor.scalar_type() == at::ScalarType::BFloat16) view.dtype = shardstream::ScalarType::BFloat16;
    if (tensor.scalar_type() == at::ScalarType::Float8_e4m3fn) view.dtype = shardstream::ScalarType::Float8E4M3;
    view.item_bytes = tensor.element_size();
    view.elements = tensor.numel();
    view.dimensions = tensor.sizes().vec();
    view.strides = tensor.strides().vec();
    if (device < 0) device = view.device;
    views.push_back(std::move(view));
  }
  auto stream = at::cuda::getCurrentCUDAStream(device).stream();
  shardstream::Metrics metrics;
  {
    py::gil_scoped_release release;
    metrics = shardstream::launch(handle, views, lengths, tensor_offsets, tensor_row_bytes,
      tensor_row_strides, peers, ordinals, expected_counts, forward_peers, ring_ids,
      sender, sequence, quantization, prepare_only, stream);
  }
  return py::cast(metrics);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("unique_id_size", []() { return shardstream::get_unique_id().size(); });
  module.def("get_unique_id", []() { return py::bytes(shardstream::get_unique_id()); });
  module.def("create", [](const py::bytes& id, int world_size, int rank, int device,
     int timeout_ms, int max_channels, int fifo_depth, int64_t step_bytes,
     int64_t chunk_bytes, int gin_fifo_depth, int64_t network_step_bytes,
     int64_t gin_chunk_bytes, int gin_context_count,
     const std::vector<int64_t>& mapping) {
       const std::string value = id;
       py::gil_scoped_release release;
       return shardstream::create(value, world_size, rank, device, timeout_ms, max_channels,
         fifo_depth, step_bytes, chunk_bytes, gin_fifo_depth, network_step_bytes,
         gin_chunk_bytes, gin_context_count, mapping);
     }, py::arg("id"), py::arg("world_size"), py::arg("rank"), py::arg("device"),
     py::arg("timeout_ms"), py::arg("max_channels"), py::arg("fifo_depth"),
     py::arg("step_bytes"), py::arg("chunk_bytes"), py::arg("gin_fifo_depth") = 16,
     py::arg("network_step_bytes") = 128 * 1024, py::arg("gin_chunk_bytes") = 4 * 1024 * 1024,
     py::arg("gin_context_count") = 1, py::arg("logical_to_communicator") = std::vector<int64_t>{});
  module.def("launch", &launch, py::arg("handle"), py::arg("tensors"), py::arg("lengths"),
             py::arg("tensor_offsets"), py::arg("tensor_row_bytes"), py::arg("tensor_row_strides"),
             py::arg("peers"), py::arg("ordinals"), py::arg("expected_counts"), py::arg("forward_peers"),
             py::arg("ring_ids"), py::arg("sender"), py::arg("sequence"),
             py::arg("quantization") = std::vector<std::vector<int64_t>>{},
             py::arg("prepare_only") = false);
  module.def("prepare", [](int64_t handle, const py::list& tensors,
                const std::vector<int64_t>& lengths, const std::vector<int64_t>& tensor_offsets,
                const std::vector<int64_t>& tensor_row_bytes, const std::vector<int64_t>& tensor_row_strides,
                const std::vector<int64_t>& peers, const std::vector<int64_t>& ordinals,
                const std::vector<int64_t>& expected_counts, const std::vector<int64_t>& forward_peers,
                const std::vector<int64_t>& ring_ids, bool sender,
                const std::vector<std::vector<int64_t>>& quantization) {
      return launch(handle, tensors, lengths, tensor_offsets, tensor_row_bytes, tensor_row_strides,
                    peers, ordinals, expected_counts, forward_peers, ring_ids, sender, 0,
                    quantization, true);
    }, py::arg("handle"), py::arg("tensors"), py::arg("lengths"), py::arg("tensor_offsets"),
    py::arg("tensor_row_bytes"), py::arg("tensor_row_strides"), py::arg("peers"), py::arg("ordinals"),
    py::arg("expected_counts"), py::arg("forward_peers"), py::arg("ring_ids"), py::arg("sender"),
    py::arg("quantization") = std::vector<std::vector<int64_t>>{});
  module.def("destroy", &shardstream::destroy, py::call_guard<py::gil_scoped_release>());
}

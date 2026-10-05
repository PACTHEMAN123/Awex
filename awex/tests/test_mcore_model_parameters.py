# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import torch

from awex.converter.mcore_converter import get_mcore_model_parameters


class _FakeDTensor:
    def __init__(self, local):
        self.local = local

    def to_local(self):
        return self.local


class _FakeModel:
    def __init__(self, parameter, expert_bias):
        self.parameter = parameter
        self.expert_bias = expert_bias

    def named_parameters(self):
        return [("decoder.layers.0.self_attention.linear_qkv.weight", self.parameter)]

    def state_dict(self):
        return {"decoder.layers.0.mlp.router.expert_bias": self.expert_bias}


def test_get_mcore_model_parameters_uses_dtensor_local_payload(monkeypatch):
    import torch.distributed.tensor as tensor_module

    monkeypatch.setattr(tensor_module, "DTensor", _FakeDTensor)
    qkv_local = torch.arange(12).reshape(3, 4)
    expert_bias_local = torch.arange(2)
    model = _FakeModel(_FakeDTensor(qkv_local), _FakeDTensor(expert_bias_local))

    parameters = get_mcore_model_parameters(model)

    assert (
        parameters["decoder.layers.0.self_attention.linear_qkv.weight"] is qkv_local
    )
    assert parameters["decoder.layers.0.mlp.router.expert_bias"] is expert_bias_local


def test_get_mcore_model_parameters_prefers_megatron_fsdp_orig_param(monkeypatch):
    import torch.distributed.tensor as tensor_module

    monkeypatch.setattr(tensor_module, "DTensor", _FakeDTensor)
    qkv_tp_local = torch.arange(12).reshape(3, 4)
    qkv_dp_shard = torch.arange(4)
    parameter = _FakeDTensor(qkv_dp_shard)
    parameter.orig_param = qkv_tp_local
    model = _FakeModel(parameter, torch.arange(2))

    parameters = get_mcore_model_parameters(model)

    assert (
        parameters["decoder.layers.0.self_attention.linear_qkv.weight"]
        is qkv_tp_local
    )


def test_get_mcore_model_parameters_preserves_regular_tensors():
    qkv = torch.arange(12).reshape(3, 4)
    expert_bias = torch.arange(2)
    model = _FakeModel(qkv, expert_bias)

    parameters = get_mcore_model_parameters(model)

    assert parameters["decoder.layers.0.self_attention.linear_qkv.weight"] is qkv
    assert parameters["decoder.layers.0.mlp.router.expert_bias"] is expert_bias

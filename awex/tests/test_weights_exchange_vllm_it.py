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

import copy
import sys
from types import SimpleNamespace

import pytest

from awex.tests import weights_exchange_vllm_infer_it
from awex.tests.experimental.compare_megatron_vllm_weights_multi import (
    _should_include_name,
)
from awex.tests.weights_exchange_multi_vllm_it import (
    MultiVLLMWeightsExchangeIT,
)
from awex.tests.weights_exchange_vllm_it import (
    VLLMWeightsExchangeIT,
    vllm_inference_config,
)
from awex.util import device as device_util


def _set_distributed_env(monkeypatch, rank=0, local_rank=0, world_size=2):
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cuda")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


def test_inference_only_vllm_children_keep_node_local_rank_offsets(monkeypatch):
    launched = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,5,7")
    monkeypatch.setattr(
        weights_exchange_vllm_infer_it.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs["env"])),
    )
    args = SimpleNamespace(
        model_path="model",
        host="127.0.0.1",
        port=8000,
        num_engines=2,
        vllm_tp_size=2,
        vllm_gpu_memory_utilization=0.8,
        vllm_enable_expert_parallel=False,
    )

    groups = weights_exchange_vllm_infer_it._inference_device_groups(args)
    for engine_rank, devices in enumerate(groups):
        weights_exchange_vllm_infer_it._start_vllm_server(args, engine_rank, devices)

    assert groups == [["2", "3"], ["5", "7"]]
    assert launched[0][1]["AWEX_NODE_LOCAL_RANK_OFFSET"] == "0"
    assert launched[1][1]["AWEX_NODE_LOCAL_RANK_OFFSET"] == "2"
    assert launched[0][1]["AWEX_NODE_LOCAL_WORLD_SIZE"] == "4"
    assert launched[1][1]["AWEX_NODE_LOCAL_WORLD_SIZE"] == "4"
    assert launched[0][1]["AWEX_NODE_LOCAL_GPU_IDS"] == "2,3,5,7"
    assert launched[1][1]["AWEX_NODE_LOCAL_GPU_IDS"] == "2,3,5,7"


@pytest.mark.parametrize(
    ("name", "max_layers", "include_non_layer", "expected"),
    [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", 1, False, True),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", 1, True, False),
        ("model.embed_tokens.weight", 1, False, False),
        ("model.embed_tokens.weight", 1, True, True),
    ],
)
def test_megatron_compare_layer_filter(name, max_layers, include_non_layer, expected):
    assert _should_include_name(name, max_layers, include_non_layer) is expected


def test_select_devices_reserves_disjoint_train_and_vllm_gpus(monkeypatch):
    _set_distributed_env(monkeypatch, rank=1, local_rank=1)
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "2,3,4,5")
    config = copy.deepcopy(vllm_inference_config)
    config["tp_size"] = 2

    integration = VLLMWeightsExchangeIT(
        inference_config=config,
        comm_backend="nccl",
        train_tp_size=2,
    )

    assert integration.megatron_device == 3
    assert integration.vllm_visible_devices == [4, 5]


def test_train_tp_requires_matching_torchrun_world_size(monkeypatch):
    _set_distributed_env(monkeypatch, world_size=1)

    with pytest.raises(RuntimeError, match="Invalid Megatron parallel config"):
        VLLMWeightsExchangeIT(
            comm_backend="nccl",
            train_tp_size=2,
        )


def test_train_ep_reserves_actual_world_size_before_vllm_gpus(monkeypatch):
    _set_distributed_env(monkeypatch, rank=1, local_rank=1, world_size=2)
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "2,3,4,5")
    config = copy.deepcopy(vllm_inference_config)
    config["tp_size"] = 2

    integration = VLLMWeightsExchangeIT(
        inference_config=config,
        comm_backend="nccl",
        train_tp_size=1,
        train_ep_size=2,
        train_expert_tp_size=1,
    )

    assert integration.megatron_device == 3
    assert integration.vllm_visible_devices == [4, 5]


def test_train_parallelism_defaults_expert_tp_to_dense_tp(monkeypatch):
    _set_distributed_env(monkeypatch, world_size=2)

    with pytest.raises(RuntimeError, match="required multiple=4"):
        VLLMWeightsExchangeIT(
            comm_backend="nccl",
            train_tp_size=2,
            train_ep_size=2,
        )


def test_explicit_expert_tp_can_share_world_ranks_with_dense_tp(monkeypatch):
    _set_distributed_env(monkeypatch, world_size=2)
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "0,1,2")

    integration = VLLMWeightsExchangeIT(
        comm_backend="nccl",
        train_tp_size=2,
        train_ep_size=2,
        train_expert_tp_size=1,
    )

    assert integration.train_parallelism.required_world_size_multiple == 2
    assert integration.vllm_visible_devices == [2]


@pytest.mark.parametrize(
    "integration_class", [VLLMWeightsExchangeIT, MultiVLLMWeightsExchangeIT]
)
def test_vllm_child_uses_current_python(monkeypatch, integration_class):
    _set_distributed_env(monkeypatch, world_size=1)
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "0,1")
    config = copy.deepcopy(vllm_inference_config)
    config["tp_size"] = 1
    launched = []

    class _Process:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        "subprocess.Popen",
        lambda command, **kwargs: launched.append(command) or _Process(),
    )
    integration = integration_class(
        inference_config=config,
        comm_backend="nccl",
    )
    monkeypatch.setattr(integration, "_wait_for_health", lambda *args: None)

    integration._start_vllm_server()

    assert launched[0][0] == sys.executable

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

import pytest

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


def _set_distributed_env(
    monkeypatch, rank=0, local_rank=0, world_size=2, local_world_size=None
):
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cuda")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))


@pytest.mark.parametrize(
    ("name", "max_layers", "include_non_layer", "expected"),
    [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", 1, False, True),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", 1, True, False),
        ("model.embed_tokens.weight", 1, False, False),
        ("model.embed_tokens.weight", 1, True, True),
    ],
)
def test_megatron_compare_layer_filter(
    name, max_layers, include_non_layer, expected
):
    assert _should_include_name(name, max_layers, include_non_layer) is expected


def test_select_devices_reserves_disjoint_train_and_vllm_gpus(monkeypatch):
    _set_distributed_env(monkeypatch, rank=1, local_rank=1)
    monkeypatch.setattr(
        device_util, "visible_devices_env_value", lambda: "2,3,4,5"
    )
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
    monkeypatch.setattr(
        device_util, "visible_devices_env_value", lambda: "2,3,4,5"
    )
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
    monkeypatch.setattr(
        device_util, "visible_devices_env_value", lambda: "0,1,2"
    )

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
def test_multinode_device_selection_uses_local_world_size(
    monkeypatch, integration_class
):
    _set_distributed_env(
        monkeypatch, rank=0, local_rank=0, world_size=2, local_world_size=1
    )
    monkeypatch.setattr(
        device_util, "visible_devices_env_value", lambda: "0,1,2"
    )
    config = copy.deepcopy(vllm_inference_config)
    config["tp_size"] = 2

    integration = integration_class(
        inference_config=config,
        comm_backend="nccl",
        train_tp_size=2,
    )

    assert integration.megatron_device == 0
    assert integration.vllm_visible_devices == [1, 2]


@pytest.mark.parametrize(
    "integration_class", [VLLMWeightsExchangeIT, MultiVLLMWeightsExchangeIT]
)
def test_multinode_non_driver_does_not_reserve_vllm_devices(
    monkeypatch, integration_class
):
    _set_distributed_env(
        monkeypatch, rank=1, local_rank=0, world_size=2, local_world_size=1
    )
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "4")
    config = copy.deepcopy(vllm_inference_config)
    config["tp_size"] = 2

    integration = integration_class(
        inference_config=config,
        comm_backend="nccl",
        train_tp_size=2,
    )

    assert integration.megatron_device == 4
    assert integration.vllm_visible_devices == []


@pytest.mark.parametrize(
    "integration_class", [VLLMWeightsExchangeIT, MultiVLLMWeightsExchangeIT]
)
def test_train_only_does_not_reserve_inference_devices(
    monkeypatch, integration_class
):
    _set_distributed_env(monkeypatch, world_size=1)
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "4")
    config = copy.deepcopy(vllm_inference_config)
    config["tp_size"] = 2

    integration = integration_class(
        inference_config=config,
        comm_backend="nccl_device_v2",
        train_only=True,
    )

    assert integration.megatron_device == 4
    assert integration.vllm_visible_devices == []


@pytest.mark.parametrize(
    "integration_class", [VLLMWeightsExchangeIT, MultiVLLMWeightsExchangeIT]
)
def test_train_only_initialize_does_not_control_inference_server(
    monkeypatch, integration_class
):
    _set_distributed_env(monkeypatch, world_size=1)
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "0")
    config = copy.deepcopy(vllm_inference_config)
    config["num_engines"] = 2
    integration = integration_class(
        inference_config=config,
        comm_backend="nccl_device_v2",
        train_only=True,
    )
    calls = []
    monkeypatch.setattr(integration, "_start_meta_server", lambda: None)
    monkeypatch.setattr(integration, "_init_distributed", lambda: None)
    monkeypatch.setattr(integration, "_share_meta_server_address", lambda: None)
    monkeypatch.setattr(integration, "_init_megatron_engine", lambda: None)
    for method_name in ("_start_vllm_server", "_wait_for_health", "_awex_init"):
        monkeypatch.setattr(
            integration,
            method_name,
            lambda *args, name=method_name: calls.append((name, args)),
        )
    monkeypatch.setattr(integration, "_training_barrier", lambda: None)

    integration.initialize()

    assert calls == []


@pytest.mark.parametrize(
    "integration_class", [VLLMWeightsExchangeIT, MultiVLLMWeightsExchangeIT]
)
def test_vllm_child_uses_current_python(monkeypatch, integration_class):
    _set_distributed_env(monkeypatch, world_size=1)
    monkeypatch.setattr(
        device_util, "visible_devices_env_value", lambda: "0,1"
    )
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

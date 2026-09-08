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

import pytest

from awex.tests.weights_exchange_vllm_it import (
    VLLMWeightsExchangeIT,
    vllm_inference_config,
)
from awex.util import device as device_util


def _set_distributed_env(monkeypatch, rank=0, local_rank=0, world_size=2):
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cuda")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


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

    with pytest.raises(RuntimeError, match="WORLD_SIZE.*train TP size"):
        VLLMWeightsExchangeIT(
            comm_backend="nccl",
            train_tp_size=2,
        )

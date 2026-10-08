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

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("verl")

from awex.verl_checkpoint_engine import AwexCheckpointEngine  # noqa: E402


def test_build_topology_maps_one_leader_per_replica():
    metadata = [
        {"role": "actor", "is_master": True, "meta_server_addr": "10.0.0.1:1234"},
        {"role": "actor", "is_master": False},
        {"role": "rollout", "replica_rank": 0, "is_leader": True},
        {"role": "rollout", "replica_rank": 0, "is_leader": False},
        {"role": "rollout", "replica_rank": 1, "is_leader": True},
        {"role": "rollout", "replica_rank": 1, "is_leader": False},
    ]

    actor, rollout = AwexCheckpointEngine.build_topology(2, 4, metadata)

    assert actor["meta_server_addr"] == ["10.0.0.1:1234"] * 2
    assert actor["num_engines"] == [2, 2]
    assert rollout["engine_rank"] == [0, 0, 1, 1]
    assert rollout["num_engines"] == [2, 2, 2, 2]


def test_build_topology_rejects_duplicate_replica_leaders():
    metadata = [
        {"role": "actor", "is_master": True, "meta_server_addr": "10.0.0.1:1234"},
        {"role": "rollout", "replica_rank": 0, "is_leader": True},
        {"role": "rollout", "replica_rank": 0, "is_leader": True},
    ]

    with pytest.raises(ValueError, match="exactly one"):
        AwexCheckpointEngine.build_topology(1, 2, metadata)


@pytest.mark.parametrize("transfer_fails", [False, True])
def test_direct_receive_publishes_version_only_after_success(transfer_fails):
    calls = []

    def transfer(step):
        calls.append(("transfer", step))
        if transfer_fails:
            raise RuntimeError("transfer failed")

    async def clear_cache():
        calls.append(("clear_cache",))

    async def stamp_version(step):
        calls.append(("version", step))

    engine = AwexCheckpointEngine(256, comm_backend="nccl_device_v2")
    engine.role = "rollout"
    engine._awex_inference_engine = SimpleNamespace(update_weights=transfer)
    engine.server_adapter = SimpleNamespace(
        _has_server=True,
        server_handle=SimpleNamespace(
            clear_kv_cache=SimpleNamespace(remote=clear_cache),
            set_global_steps=SimpleNamespace(remote=stamp_version),
        ),
    )
    if transfer_fails:
        with pytest.raises(RuntimeError, match="transfer failed"):
            asyncio.run(engine.receive_weights(global_steps=7))
        assert calls == [("transfer", 7)]
    else:
        asyncio.run(engine.receive_weights(global_steps=7))
        assert calls == [("transfer", 7), ("clear_cache",), ("version", 7)]

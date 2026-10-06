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

import os
from types import SimpleNamespace

import pytest
import torch

from awex.transfer import nccl_device_v2
from awex.transfer.nccl_device_v2 import _build_recv_batch, _build_send_batch
from awex.transfer.nccl_device_v2_gin import (
    _active_rdma_endpoints,
    _configure_gin_hca_policy,
    _parse_nvidia_topology,
    _RdmaEndpoint,
    _weighted_hca_assignments,
)
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan


def _replica_operation(root: int, receiver: int) -> CommunicationOperation:
    shard = SimpleNamespace(name="weight", shape=(8,))
    return CommunicationOperation(
        send_rank=root,
        send_shard_meta=shard,
        send_offset=(0,),
        recv_rank=receiver,
        recv_shard_meta=shard,
        recv_offset=(0,),
        overlap_shape=(8,),
        train_slices=(slice(None),),
        inf_slices=(slice(None),),
    )


def test_v2_ring_broadcast_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    tensor = torch.arange(8, dtype=torch.int32)
    plan = TransferPlan(
        operations={peer: [_replica_operation(4, peer)] for peer in range(4)}
    )

    batch = _build_send_batch(
        {"weight": tensor},
        plan,
        rank=4,
        world_size=5,
        chunk_bytes=16,
        infer_instance_world_size=1,
        num_infer_engines=4,
    )

    assert batch.peers == [0, 1, 2, 3]
    assert batch.ring_ids == [-1, -1, -1, -1]


def test_v2_fixed_ring_injects_one_copy_and_relays_in_engine_order(monkeypatch):
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    tensor = torch.arange(8, dtype=torch.int32)
    send_plan = TransferPlan(
        operations={peer: [_replica_operation(4, peer)] for peer in range(4)}
    )
    send_batch = _build_send_batch(
        {"weight": tensor},
        send_plan,
        rank=4,
        world_size=5,
        chunk_bytes=16,
        infer_instance_world_size=1,
        num_infer_engines=4,
        ring_broadcast=True,
    )

    assert send_batch.peers == [0]
    assert send_batch.expected_counts == [1, 0, 0, 0, 0]
    assert send_batch.ring_ids == [4]

    recv_batch = _build_recv_batch(
        {"weight": torch.empty_like(tensor)},
        TransferPlan(operations={4: [_replica_operation(4, 1)]}),
        rank=1,
        world_size=5,
        chunk_bytes=16,
        infer_instance_world_size=1,
        num_infer_engines=4,
        ring_broadcast=True,
    )
    assert recv_batch.peers == [0]
    assert recv_batch.forward_peers == [2]
    assert recv_batch.ring_ids == [4]


def test_v2_swizzle_rotates_ring_by_root(monkeypatch):
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    tensor = torch.arange(8, dtype=torch.int32)
    send_plan = TransferPlan(
        operations={peer: [_replica_operation(5, peer)] for peer in range(4)}
    )
    send_batch = _build_send_batch(
        {"weight": tensor},
        send_plan,
        rank=5,
        world_size=6,
        chunk_bytes=16,
        infer_instance_world_size=1,
        num_infer_engines=4,
        ring_broadcast=True,
        ring_swizzle=True,
    )

    assert send_batch.peers == [1]
    assert send_batch.ring_ids == [5]

    last_batch = _build_recv_batch(
        {"weight": torch.empty_like(tensor)},
        TransferPlan(operations={5: [_replica_operation(5, 0)]}),
        rank=0,
        world_size=6,
        chunk_bytes=16,
        infer_instance_world_size=1,
        num_infer_engines=4,
        ring_broadcast=True,
        ring_swizzle=True,
    )
    assert last_batch.peers == [3]
    assert last_batch.forward_peers == [-1]


def test_v2_swizzle_switch_is_inert_when_ring_broadcast_is_off(monkeypatch):
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    tensor = torch.arange(8, dtype=torch.int32)
    batch = _build_send_batch(
        {"weight": tensor},
        TransferPlan(
            operations={peer: [_replica_operation(5, peer)] for peer in range(4)}
        ),
        rank=5,
        world_size=6,
        chunk_bytes=16,
        infer_instance_world_size=1,
        num_infer_engines=4,
        ring_broadcast=False,
        ring_swizzle=True,
    )

    assert batch.peers == [0, 1, 2, 3]
    assert batch.ring_ids == [-1, -1, -1, -1]


def test_v2_ring_rejects_partial_replica_streams(monkeypatch):
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    tensor = torch.arange(8, dtype=torch.int32)

    with pytest.raises(
        nccl_device_v2.NCCLDeviceV2UnavailableError,
        match="present in all engines",
    ):
        _build_send_batch(
            {"weight": tensor},
            TransferPlan(
                operations={peer: [_replica_operation(5, peer)] for peer in (0, 1)}
            ),
            rank=5,
            world_size=6,
            chunk_bytes=16,
            infer_instance_world_size=1,
            num_infer_engines=4,
            ring_broadcast=True,
        )


def test_active_rdma_endpoints_read_active_port_capacity(tmp_path):
    device = tmp_path / "mlx5_7"
    (device / "device" / "net" / "eth7").mkdir(parents=True)
    port = device / "ports" / "2"
    port.mkdir(parents=True)
    (port / "state").write_text("4: ACTIVE\n", encoding="ascii")
    (port / "rate").write_text("400 Gb/sec (4X NDR)\n", encoding="ascii")

    endpoints = _active_rdma_endpoints(str(tmp_path))

    assert [(item.name, item.port, item.bandwidth_gbps) for item in endpoints] == [
        ("mlx5_7", 2, 400.0)
    ]


def test_balanced_hca_policy_groups_ranks_across_devices(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setattr(
        "awex.transfer.nccl_device_v2_gin._active_rdma_endpoints",
        lambda: [
            _RdmaEndpoint(name, 1, 200.0, f"/pci/{index}")
            for index, name in enumerate(["mlx5_3", "mlx5_8", "mlx5_19", "mlx5_30"])
        ],
    )

    _configure_gin_hca_policy()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_19:1"


def test_balanced_hca_policy_applies_node_rank_offset(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("AWEX_NODE_LOCAL_RANK_OFFSET", "4")
    monkeypatch.setenv("AWEX_NODE_LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setattr(
        "awex.transfer.nccl_device_v2_gin._active_rdma_endpoints",
        lambda: [
            _RdmaEndpoint(name, 1, 200.0, f"/pci/{index}")
            for index, name in enumerate(["mlx5_3", "mlx5_8", "mlx5_19", "mlx5_30"])
        ],
    )

    _configure_gin_hca_policy()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_19:1"


def test_weighted_hca_assignments_follow_capacity_and_payload():
    endpoints = [
        _RdmaEndpoint("fast", 1, 400.0, "/pci/0"),
        _RdmaEndpoint("slow", 1, 200.0, "/pci/1"),
    ]

    uniform = _weighted_hca_assignments(endpoints, [1] * 6)
    skewed = _weighted_hca_assignments(endpoints, [800, 700, 100])

    assert [endpoint.name for endpoint in uniform].count("fast") == 4
    assert [endpoint.name for endpoint in uniform].count("slow") == 2
    assert [endpoint.name for endpoint in skewed] == ["fast", "slow", "fast"]


def test_weighted_hca_assignments_keep_ranks_on_local_numa_rails():
    endpoints = [
        _RdmaEndpoint(f"mlx5_{index}", 1, 200.0, f"/pci/{index}") for index in range(4)
    ]
    topology = [
        [3, 3, 4, 4],
        [0, 3, 4, 4],
        [3, 3, 4, 4],
        [3, 0, 4, 4],
        [4, 4, 0, 3],
        [4, 4, 3, 3],
        [4, 4, 3, 0],
        [4, 4, 3, 3],
    ]

    assignments = _weighted_hca_assignments(endpoints, [1] * 8, topology)

    assert [endpoint.name for endpoint in assignments] == [
        "mlx5_0",
        "mlx5_0",
        "mlx5_1",
        "mlx5_1",
        "mlx5_2",
        "mlx5_2",
        "mlx5_3",
        "mlx5_3",
    ]


def test_parse_nvidia_topology_maps_active_hca_columns():
    output = """\
        \x1b[4mGPU0 GPU1 NIC0 NIC1 CPU Affinity\x1b[0m
GPU0    X    NV18 PIX  SYS  0-7
GPU1    NV18 X    NODE PIX  0-7

NIC Legend:
  NIC0: mlx5_3
  NIC1: mlx5_8
"""
    endpoints = [
        _RdmaEndpoint("mlx5_3", 1, 200.0, "/pci/0"),
        _RdmaEndpoint("mlx5_8", 1, 200.0, "/pci/1"),
    ]

    distances = _parse_nvidia_topology(output, endpoints, ["0", "1"])

    assert distances == [[0, 4], [3, 0]]


def test_balanced_hca_policy_preserves_explicit_nccl_selection(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced")
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_8:1")

    _configure_gin_hca_policy()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_8:1"

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
import threading
from types import SimpleNamespace

import pytest
import torch
from shardstream import transport
from shardstream.plan import CommunicationOperation, TransferPlan
from shardstream.topology import (
    _active_rdma_endpoints,
    _configure_gin_hca_policy,
    _parse_nvidia_topology,
    _RdmaEndpoint,
    _weighted_hca_assignments,
)
from shardstream.transport import (
    _build_recv_batch,
    _build_send_batch,
    _node_major_communicator_ranks,
)


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


def test_node_major_communicator_ranks_join_split_physical_nodes():
    node_ids = [10] * 4 + [20] * 4 + [10] * 4 + [20] * 4 + [30] * 8 + [40] * 8

    mapping = _node_major_communicator_ranks(node_ids)

    assert [mapping[rank] for rank in range(0, 4)] == [0, 1, 2, 3]
    assert [mapping[rank] for rank in range(8, 12)] == [4, 5, 6, 7]
    assert [mapping[rank] for rank in range(4, 8)] == [8, 9, 10, 11]
    assert [mapping[rank] for rank in range(12, 16)] == [12, 13, 14, 15]
    assert sorted(mapping) == list(range(32))


def test_interleaved_bias_vectors_stage_and_preserve_neighbor_values(monkeypatch):
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
    backing = torch.full((16,), -1.0)
    parameters = {"gate": backing[::2], "up": backing[1::2]}
    operations = []
    for name in parameters:
        operation = _replica_operation(4, 1)
        operation.recv_shard_meta.name = name
        operations.append(operation)
    plan = TransferPlan(operations={4: operations})
    batch = _build_recv_batch(parameters, plan, 1, 5, 0)
    assert len(batch.copybacks) == 2
    for index, (destination, staging) in enumerate(batch.copybacks):
        assert staging.is_contiguous()
        staging.copy_(torch.arange(8) + 100 * (index + 1))
        destination.copy_(staging)
    assert torch.equal(backing[::2], torch.arange(8) + 100)
    assert torch.equal(backing[1::2], torch.arange(8) + 200)
    with pytest.raises(transport.TransportUnavailableError, match="strided vector"):
        _build_recv_batch(parameters, plan, 1, 5, 0, allow_staging=False)


def test_v2_ring_broadcast_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
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
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
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
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
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


@pytest.mark.parametrize("root", [8, 9, 10, 11])
def test_naive_ring_has_fixed_entry_and_groups_actual_nodes(monkeypatch, root):
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
    tensor = torch.arange(8, dtype=torch.int32)
    topology = (20, 10, 20, 10)
    targets = [0, 2, 4, 6]
    common = dict(
        world_size=24,
        chunk_bytes=16,
        infer_instance_world_size=2,
        num_infer_engines=4,
        ring_broadcast=True,
        rollout_node_ids=topology,
    )
    send = _build_send_batch(
        {"weight": tensor},
        TransferPlan(operations={p: [_replica_operation(root, p)] for p in targets}),
        rank=root,
        **common,
    )
    assert send.peers == [0]
    # Engine IDs alternate nodes; the actual chain groups them as 0,2,1,3.
    for rank, source, forward in ((0, root, 4), (4, 0, 2), (2, 4, 6), (6, 2, -1)):
        recv = _build_recv_batch(
            {"weight": torch.empty_like(tensor)},
            TransferPlan(operations={root: [_replica_operation(root, rank)]}),
            rank=rank,
            **common,
        )
        assert recv.peers == [source]
        assert recv.forward_peers == [forward]
        assert recv.ring_ids == send.ring_ids


def test_resolve_rollout_topology_uses_group_placement_once(monkeypatch):
    instance = transport.Transport.__new__(transport.Transport)
    instance._operation_lock = threading.RLock()
    instance.ring_broadcast = True
    instance.rollout_node_ids = None
    instance.world_size = 24
    instance.infer_instance_world_size = 2
    instance.num_infer_engines = 4
    instance.group = object()
    calls = []

    def gather(result, node, *, group):
        assert group is instance.group
        calls.append(node)
        result[:] = [20, 20, 10, 10, 20, 20, 10, 10] + [30] * 16

    monkeypatch.setattr(transport.dist, "all_gather_object", gather)
    instance.resolve_rollout_topology()
    instance.resolve_rollout_topology()
    assert instance.rollout_node_ids == (20, 10, 20, 10)
    assert len(calls) == 1


def test_v2_swizzle_groups_round_robin_engines_by_node(monkeypatch):
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
    monkeypatch.setenv("SHARDSTREAM_NODE_LOCAL_WORLD_SIZE", "8")
    tensor = torch.arange(8, dtype=torch.int32)
    targets = list(range(0, 16, 2))
    send_plan = TransferPlan(
        operations={peer: [_replica_operation(16, peer)] for peer in targets}
    )

    send_batch = _build_send_batch(
        {"weight": tensor},
        send_plan,
        rank=16,
        world_size=32,
        chunk_bytes=16,
        infer_instance_world_size=2,
        num_infer_engines=8,
        ring_broadcast=True,
        ring_swizzle=True,
    )
    assert send_batch.peers == [0]

    local_relay = _build_recv_batch(
        {"weight": torch.empty_like(tensor)},
        TransferPlan(operations={16: [_replica_operation(16, 0)]}),
        rank=0,
        world_size=32,
        chunk_bytes=16,
        infer_instance_world_size=2,
        num_infer_engines=8,
        ring_broadcast=True,
        ring_swizzle=True,
    )
    assert local_relay.peers == [16]
    assert local_relay.forward_peers == [4]

    cross_node_relay = _build_recv_batch(
        {"weight": torch.empty_like(tensor)},
        TransferPlan(operations={16: [_replica_operation(16, 12)]}),
        rank=12,
        world_size=32,
        chunk_bytes=16,
        infer_instance_world_size=2,
        num_infer_engines=8,
        ring_broadcast=True,
        ring_swizzle=True,
    )
    assert cross_node_relay.peers == [8]
    assert cross_node_relay.forward_peers == [2]


def test_v2_elastic_swizzle_uses_actual_nodes_instead_of_gpu_capacity(monkeypatch):
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
    monkeypatch.setenv("SHARDSTREAM_NODE_LOCAL_WORLD_SIZE", "8")
    tensor = torch.arange(8, dtype=torch.int32)
    topology = (10, 20, 10, 20)
    targets = [0, 2, 4, 6]
    send = _build_send_batch(
        {"weight": tensor},
        TransferPlan(
            operations={peer: [_replica_operation(8, peer)] for peer in targets}
        ),
        rank=8,
        world_size=24,
        chunk_bytes=16,
        infer_instance_world_size=2,
        num_infer_engines=4,
        ring_broadcast=True,
        ring_swizzle=True,
        rollout_node_ids=topology,
    )
    assert send.peers == [0]
    # Both local edges must remain LSA candidates for an intermediate cohort.
    for rank, source, forward in ((0, 8, 4), (4, 0, 2), (2, 4, 6), (6, 2, -1)):
        recv = _build_recv_batch(
            {"weight": torch.empty_like(tensor)},
            TransferPlan(operations={8: [_replica_operation(8, rank)]}),
            rank=rank,
            world_size=24,
            chunk_bytes=16,
            infer_instance_world_size=2,
            num_infer_engines=4,
            ring_broadcast=True,
            ring_swizzle=True,
            rollout_node_ids=topology,
        )
        assert recv.peers == [source]
        assert recv.forward_peers == [forward]


def test_v2_swizzle_switch_is_inert_when_ring_broadcast_is_off(monkeypatch):
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
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
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
    tensor = torch.arange(8, dtype=torch.int32)

    with pytest.raises(
        transport.TransportUnavailableError,
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
    monkeypatch.setenv("SHARDSTREAM_HCA_POLICY", "balanced")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setattr(
        "shardstream.topology._active_rdma_endpoints",
        lambda: [
            _RdmaEndpoint(name, 1, 200.0, f"/pci/{index}")
            for index, name in enumerate(["mlx5_3", "mlx5_8", "mlx5_19", "mlx5_30"])
        ],
    )

    _configure_gin_hca_policy()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_19:1"


def test_balanced_hca_policy_applies_node_rank_offset(monkeypatch):
    monkeypatch.setenv("SHARDSTREAM_HCA_POLICY", "balanced")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("SHARDSTREAM_NODE_LOCAL_RANK_OFFSET", "4")
    monkeypatch.setenv("SHARDSTREAM_NODE_LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setattr(
        "shardstream.topology._active_rdma_endpoints",
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
    monkeypatch.setenv("SHARDSTREAM_HCA_POLICY", "balanced")
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_8:1")

    _configure_gin_hca_policy()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_8:1"

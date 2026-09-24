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

import pytest

from awex.transfer.nccl_device_v2 import (
    NCCLDeviceV2Transport,
    NCCLDeviceV2UnavailableError,
    _active_rdma_endpoints,
    _configure_gin_hca_policy,
    _parse_nvidia_topology,
    _RdmaEndpoint,
    _resolve_fifo_depth,
    _resolve_gin_connections,
    _resolve_gin_context_count,
    _resolve_gin_doorbell_batch,
    _resolve_gin_reliable_doorbell,
    _resolve_network_channels_per_peer,
    _resolve_network_step_bytes,
    _weighted_hca_assignments,
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
        "awex.transfer.nccl_device_v2._active_rdma_endpoints",
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
        "awex.transfer.nccl_device_v2._active_rdma_endpoints",
        lambda: [
            _RdmaEndpoint(name, 1, 200.0, f"/pci/{index}")
            for index, name in enumerate(["mlx5_3", "mlx5_8", "mlx5_19", "mlx5_30"])
        ],
    )

    _configure_gin_hca_policy()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_19:1"


def test_balanced_hca_policy_exposes_only_local_numa_rails(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.delenv("NCCL_NETDEVS_POLICY", raising=False)
    endpoints = [
        _RdmaEndpoint(f"mlx5_{index}", 1, 200.0, f"/pci/{index}")
        for index in range(4)
    ]
    monkeypatch.setattr(
        "awex.transfer.nccl_device_v2._active_rdma_endpoints", lambda: endpoints
    )
    monkeypatch.setattr(
        "awex.transfer.nccl_device_v2._gpu_hca_topology",
        lambda _endpoints, _world_size: [
            [3, 3, 4, 4],
            [0, 3, 4, 4],
            [3, 3, 4, 4],
            [3, 0, 4, 4],
        ],
    )

    _configure_gin_hca_policy()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_0:1,mlx5_1:1"
    assert os.environ["NCCL_NETDEVS_POLICY"] == "ALL"
    assert (
        os.environ["AWEX_NCCL_DEVICE_V2_SELECTED_HCA_BANDWIDTH_GBPS"]
        == "400.0"
    )


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
        _RdmaEndpoint(f"mlx5_{index}", 1, 200.0, f"/pci/{index}")
        for index in range(4)
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


def test_fifo_depth_defaults_to_sixteen(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_FIFO_DEPTH", raising=False)

    assert _resolve_fifo_depth() == 16


def test_fifo_depth_honors_environment(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_FIFO_DEPTH", "16")

    assert _resolve_fifo_depth() == 16


@pytest.mark.parametrize("value", [0, 65])
def test_fifo_depth_rejects_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[1, 64\]"):
        _resolve_fifo_depth(value)


def test_network_step_defaults_to_nccl_cross_node_chunk(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES", raising=False)
    monkeypatch.delenv("NCCL_P2P_NET_CHUNKSIZE", raising=False)

    assert _resolve_network_step_bytes(None) == 128 * 1024


def test_network_step_honors_nccl_configuration(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES", raising=False)
    monkeypatch.setenv("NCCL_P2P_NET_CHUNKSIZE", str(256 * 1024))

    assert _resolve_network_step_bytes(None) == 256 * 1024


def test_awex_network_step_overrides_nccl_configuration(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES", str(64 * 1024))
    monkeypatch.setenv("NCCL_P2P_NET_CHUNKSIZE", str(256 * 1024))

    assert _resolve_network_step_bytes(None) == 64 * 1024


@pytest.mark.parametrize("value", [0, -16])
def test_network_step_rejects_non_positive_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match="must be positive"):
        _resolve_network_step_bytes(value)


def test_network_step_must_be_vector_aligned():
    with pytest.raises(NCCLDeviceV2UnavailableError, match="multiple of 16"):
        _resolve_network_step_bytes(127)


def test_network_channels_default_to_auto(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_CHANNELS_PER_PEER", raising=False)
    monkeypatch.delenv("NCCL_NCHANNELS_PER_NET_PEER", raising=False)

    assert _resolve_network_channels_per_peer(None) == 0


def test_network_channels_honor_nccl_configuration(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_CHANNELS_PER_PEER", raising=False)
    monkeypatch.setenv("NCCL_NCHANNELS_PER_NET_PEER", "8")

    assert _resolve_network_channels_per_peer(None) == 8


@pytest.mark.parametrize("value", [-1, 65])
def test_network_channels_reject_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[0, 64\]"):
        _resolve_network_channels_per_peer(value)


def test_gin_connections_default_to_active_rdma_devices(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS", raising=False)
    monkeypatch.delenv("NCCL_GIN_NCONNECTIONS", raising=False)
    monkeypatch.setattr(
        "awex.transfer.nccl_device_v2._detect_active_rdma_device_count", lambda: 3
    )

    assert _resolve_gin_connections(None) == 3


def test_gin_connections_fall_back_to_available_slots(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS", raising=False)
    monkeypatch.delenv("NCCL_GIN_NCONNECTIONS", raising=False)
    monkeypatch.setattr(
        "awex.transfer.nccl_device_v2._detect_active_rdma_device_count", lambda: 0
    )

    assert _resolve_gin_connections(None) == 4


def test_gin_connections_honor_nccl_configuration(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS", raising=False)
    monkeypatch.setenv("NCCL_GIN_NCONNECTIONS", "2")

    assert _resolve_gin_connections(None) == 2


def test_gin_connections_prefer_awex_override(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS", "3")
    monkeypatch.setenv("NCCL_GIN_NCONNECTIONS", "2")

    assert _resolve_gin_connections(None) == 3


@pytest.mark.parametrize("value", [-1, 5])
def test_gin_connections_reject_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[0, 4\]"):
        _resolve_gin_connections(value)


def test_gin_contexts_default_to_one_per_connection(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONTEXTS", raising=False)

    assert _resolve_gin_context_count(None) == 0


def test_transport_maps_auto_contexts_to_detected_connections(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONTEXTS", raising=False)
    monkeypatch.setenv("NCCL_GIN_NCONNECTIONS", "3")
    monkeypatch.setenv("NCCL_GIN_GDAKI_USE_RELIABLE_DB", "2")

    transport = NCCLDeviceV2Transport(None, 0, 2, gin_connections=3)

    assert transport.gin_context_count == 3


@pytest.mark.parametrize("value", [-1, 65])
def test_gin_contexts_reject_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[0, 64\]"):
        _resolve_gin_context_count(value)


def test_gin_doorbell_batch_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_DOORBELL_BATCH", raising=False)

    assert _resolve_gin_doorbell_batch(None) == 1


def test_gin_doorbell_batch_honors_environment(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_GIN_DOORBELL_BATCH", "8")

    assert _resolve_gin_doorbell_batch(None) == 8


@pytest.mark.parametrize("value", [0, 9])
def test_gin_doorbell_batch_rejects_values_beyond_fifo(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[1, 8\]"):
        _resolve_gin_doorbell_batch(value)


def test_gin_reliable_doorbell_defaults_to_fallback_mode(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_RELIABLE_DB", raising=False)
    monkeypatch.delenv("NCCL_GIN_GDAKI_USE_RELIABLE_DB", raising=False)

    assert _resolve_gin_reliable_doorbell(None) == 2


def test_gin_reliable_doorbell_honors_nccl_configuration(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_RELIABLE_DB", raising=False)
    monkeypatch.setenv("NCCL_GIN_GDAKI_USE_RELIABLE_DB", "1")

    assert _resolve_gin_reliable_doorbell(None) == 1


def test_gin_reliable_doorbell_prefers_awex_override(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_GIN_RELIABLE_DB", "0")
    monkeypatch.setenv("NCCL_GIN_GDAKI_USE_RELIABLE_DB", "1")

    assert _resolve_gin_reliable_doorbell(None) == 0


@pytest.mark.parametrize("value", [-1, 3])
def test_gin_reliable_doorbell_rejects_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[0, 2\]"):
        _resolve_gin_reliable_doorbell(value)

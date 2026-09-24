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

"""NCCL-style v2 device transport.

The fixed TransferPlan remains the source of truth. This wrapper only binds
the already ordered tensors and asks the v2 extension to split each task into
chunks, channel work batches, and FIFO steps. It is intentionally separate
from the v1 transport so the two implementations can be benchmarked side by
side while the draft is stabilized.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any


class NCCLDeviceV2UnavailableError(RuntimeError):
    """Raised when the isolated NCCL Device v2 path cannot be initialized."""


@dataclass(frozen=True, slots=True)
class _RdmaEndpoint:
    name: str
    port: int
    bandwidth_gbps: float
    pci_path: str


def _active_rdma_endpoints(
    sysfs_root: str = "/sys/class/infiniband",
) -> list[_RdmaEndpoint]:
    """Return active RDMA ports and their link capacities in stable PCI order."""

    try:
        devices = list(os.scandir(sysfs_root))
    except OSError:
        return []
    active_endpoints: list[_RdmaEndpoint] = []
    for device in devices:
        net_path = os.path.join(device.path, "device", "net")
        try:
            if not any(os.scandir(net_path)):
                continue
            ports = list(os.scandir(os.path.join(device.path, "ports")))
        except OSError:
            continue
        for port in ports:
            try:
                with open(
                    os.path.join(port.path, "state"), encoding="ascii"
                ) as state_file:
                    state = state_file.read()
            except OSError:
                continue
            if "ACTIVE" in state:
                pci_path = os.path.realpath(os.path.join(device.path, "device"))
                try:
                    port_number = int(port.name)
                except ValueError:
                    continue
                try:
                    with open(
                        os.path.join(port.path, "rate"), encoding="ascii"
                    ) as rate_file:
                        rate_match = re.search(
                            r"([0-9]+(?:\.[0-9]+)?)\s*Gb/sec", rate_file.read()
                        )
                except OSError:
                    rate_match = None
                bandwidth_gbps = float(rate_match.group(1)) if rate_match else 1.0
                active_endpoints.append(
                    _RdmaEndpoint(
                        name=device.name,
                        port=port_number,
                        bandwidth_gbps=max(1.0, bandwidth_gbps),
                        pci_path=pci_path,
                    )
                )
    return sorted(
        active_endpoints,
        key=lambda endpoint: (endpoint.pci_path, endpoint.port, endpoint.name),
    )


def _active_rdma_devices(
    sysfs_root: str = "/sys/class/infiniband",
) -> list[str]:
    """Return active RDMA device names in stable PCI order."""

    return list(
        dict.fromkeys(endpoint.name for endpoint in _active_rdma_endpoints(sysfs_root))
    )


def _weighted_hca_assignments(
    endpoints: list[_RdmaEndpoint],
    rank_payload_bytes: list[int],
    topology_distances: list[list[int]] | None = None,
) -> list[_RdmaEndpoint]:
    """Assign rank loads to HCA capacity with locality-preserving weighted bins."""

    if not endpoints or not rank_payload_bytes:
        return []
    normalized_payloads = [max(1, int(payload)) for payload in rank_payload_bytes]
    if topology_distances is not None:
        if len(topology_distances) != len(normalized_payloads) or any(
            len(distances) != len(endpoints) for distances in topology_distances
        ):
            raise NCCLDeviceV2UnavailableError(
                "GPU/HCA topology dimensions do not match local ranks and RDMA ports"
            )
        assigned_bytes = [0] * len(endpoints)
        assignments: list[_RdmaEndpoint | None] = [None] * len(normalized_payloads)

        def affinity_key(rank: int) -> tuple[int, int, int, int]:
            local_distances = sorted(
                distance
                for distance in topology_distances[rank]
                if distance < _HCA_DISTANCE_SCORES["SYS"]
            )
            best = local_distances[0] if local_distances else _HCA_DISTANCE_SCORES["SYS"]
            second = local_distances[1] if len(local_distances) > 1 else best + 1
            return (-normalized_payloads[rank], -(second - best), best, rank)

        for rank in sorted(range(len(normalized_payloads)), key=affinity_key):
            payload = normalized_payloads[rank]
            distances = topology_distances[rank]
            local_candidates = [
                index
                for index, distance in enumerate(distances)
                if distance < _HCA_DISTANCE_SCORES["SYS"]
            ]
            candidates = local_candidates or list(range(len(endpoints)))
            endpoint_index = min(
                candidates,
                key=lambda candidate: (
                    distances[candidate],
                    (assigned_bytes[candidate] + payload)
                    / endpoints[candidate].bandwidth_gbps,
                    assigned_bytes[candidate] / endpoints[candidate].bandwidth_gbps,
                    candidate,
                ),
            )
            assignments[rank] = endpoints[endpoint_index]
            assigned_bytes[endpoint_index] += payload
        return [assignment for assignment in assignments if assignment is not None]

    if len(set(normalized_payloads)) == 1:
        total_capacity = sum(endpoint.bandwidth_gbps for endpoint in endpoints)
        assignments = []
        for rank in range(len(normalized_payloads)):
            target_capacity = (
                (rank + 0.5) * total_capacity / len(normalized_payloads)
            )
            cumulative_capacity = 0.0
            for endpoint in endpoints:
                cumulative_capacity += endpoint.bandwidth_gbps
                if target_capacity <= cumulative_capacity:
                    assignments.append(endpoint)
                    break
        return assignments
    assigned_bytes = [0] * len(endpoints)
    assignments: list[_RdmaEndpoint | None] = [None] * len(normalized_payloads)
    for rank in sorted(
        range(len(normalized_payloads)),
        key=lambda candidate: (-normalized_payloads[candidate], candidate),
    ):
        payload = normalized_payloads[rank]
        endpoint_index = min(
            range(len(endpoints)),
            key=lambda candidate: (
                (assigned_bytes[candidate] + payload)
                / endpoints[candidate].bandwidth_gbps,
                assigned_bytes[candidate] / endpoints[candidate].bandwidth_gbps,
                candidate,
            ),
        )
        assignments[rank] = endpoints[endpoint_index]
        assigned_bytes[endpoint_index] += payload
    return [assignment for assignment in assignments if assignment is not None]


_HCA_DISTANCE_SCORES = {
    "PIX": 0,
    "PXB": 1,
    "PHB": 2,
    "NODE": 3,
    "SYS": 4,
}
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _parse_nvidia_topology(
    output: str, endpoints: list[_RdmaEndpoint], gpu_ids: list[str]
) -> list[list[int]] | None:
    output = _ANSI_ESCAPE_RE.sub("", output)
    lines = [line.split() for line in output.splitlines() if line.strip()]
    header = next((fields for fields in lines if fields[0] == "GPU0"), None)
    if header is None:
        return None
    nic_names: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"\s*(NIC\d+):\s+(\S+)\s*$", line)
        if match:
            nic_names[match.group(2)] = match.group(1)
    try:
        nic_columns = [header.index(nic_names[endpoint.name]) for endpoint in endpoints]
    except (KeyError, ValueError):
        return None
    rows = {fields[0]: fields for fields in lines if re.fullmatch(r"GPU\d+", fields[0])}
    distances = []
    for gpu_id in gpu_ids:
        if not gpu_id.isdigit():
            return None
        row = rows.get(f"GPU{gpu_id}")
        if row is None:
            return None
        try:
            distances.append(
                [
                    _HCA_DISTANCE_SCORES.get(row[column + 1], 5)
                    for column in nic_columns
                ]
            )
        except IndexError:
            return None
    return distances


def _node_local_gpu_ids(local_world_size: int) -> list[str]:
    configured = os.environ.get("AWEX_NODE_LOCAL_GPU_IDS")
    if configured:
        gpu_ids = [value.strip() for value in configured.split(",")]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_ids = [value.strip() for value in visible.split(",") if value.strip()]
    if len(gpu_ids) < local_world_size:
        gpu_ids = [str(rank) for rank in range(local_world_size)]
    return gpu_ids[:local_world_size]


def _gpu_hca_topology(
    endpoints: list[_RdmaEndpoint], local_world_size: int
) -> list[list[int]] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_nvidia_topology(
        result.stdout, endpoints, _node_local_gpu_ids(local_world_size)
    )


def _rank_payload_bytes_from_environment(local_world_size: int) -> list[int]:
    configured = os.environ.get("AWEX_NCCL_DEVICE_V2_RANK_PAYLOAD_BYTES", "")
    if not configured:
        return [1] * local_world_size
    try:
        payloads = [int(value.strip()) for value in configured.split(",")]
    except ValueError as exc:
        raise NCCLDeviceV2UnavailableError(
            "AWEX_NCCL_DEVICE_V2_RANK_PAYLOAD_BYTES must be comma-separated integers"
        ) from exc
    if len(payloads) != local_world_size or any(payload < 0 for payload in payloads):
        raise NCCLDeviceV2UnavailableError(
            "AWEX_NCCL_DEVICE_V2_RANK_PAYLOAD_BYTES must contain one non-negative "
            "value per local rank"
        )
    return payloads


def _configure_gin_hca_policy() -> None:
    """Spread local rank load across active RDMA capacity before NCCL starts."""

    policy = os.environ.get(
        "AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced"
    ).strip().lower()
    if policy == "topology" or "NCCL_IB_HCA" in os.environ:
        return
    if policy != "balanced":
        raise NCCLDeviceV2UnavailableError(
            "AWEX_NCCL_DEVICE_V2_HCA_POLICY must be topology or balanced"
        )
    try:
        local_rank = int(os.environ["LOCAL_RANK"]) + int(
            os.environ.get("AWEX_NODE_LOCAL_RANK_OFFSET", "0")
        )
        local_world_size = int(
            os.environ.get(
                "AWEX_NODE_LOCAL_WORLD_SIZE",
                os.environ.get("LOCAL_WORLD_SIZE", ""),
            )
        )
    except (KeyError, ValueError):
        return
    endpoints = _active_rdma_endpoints()
    if not endpoints or local_world_size <= 0 or not 0 <= local_rank < local_world_size:
        return
    topology = _gpu_hca_topology(endpoints, local_world_size)
    assignments = _weighted_hca_assignments(
        endpoints,
        _rank_payload_bytes_from_environment(local_world_size),
        topology,
    )
    selected_endpoints = [assignments[local_rank]]
    if topology is not None:
        local_endpoints = [
            endpoint
            for endpoint, distance in zip(endpoints, topology[local_rank])
            if distance < _HCA_DISTANCE_SCORES["SYS"]
        ]
        if local_endpoints:
            selected_endpoints = local_endpoints
    os.environ["NCCL_IB_HCA"] = "=" + ",".join(
        f"{endpoint.name}:{endpoint.port}" for endpoint in selected_endpoints
    )
    if len(selected_endpoints) > 1:
        os.environ.setdefault("NCCL_NETDEVS_POLICY", "ALL")
    os.environ["AWEX_NCCL_DEVICE_V2_SELECTED_HCA_BANDWIDTH_GBPS"] = str(
        sum(endpoint.bandwidth_gbps for endpoint in selected_endpoints)
    )


def _preload_configured_nccl() -> None:
    """Load the configured NCCL before torch can load another SONAME match."""

    library_dir = os.environ.get("AWEX_NCCL_LIB", "")
    if not library_dir:
        return
    candidates = (
        os.path.join(library_dir, "libnccl.so.2"),
        os.path.join(library_dir, "libnccl.so"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                raise NCCLDeviceV2UnavailableError(
                    f"Failed to preload NCCL from {candidate}: {exc}"
                ) from exc
            return


_configure_gin_hca_policy()
_preload_configured_nccl()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from awex import logging  # noqa: E402
from awex.transfer.tensor_layout import (  # noqa: E402
    StaticTensorLayout,
    slice_layout_fragments,
)
from awex.transfer.transfer_plan import (  # noqa: E402
    CommunicationOperation,
    TransferPlan,
    slice_tensor,
)
from awex.util import device as device_util  # noqa: E402

logger = logging.getLogger(__name__)

_extension_lock = threading.Lock()
_extension: Any | None = None


@dataclass
class _V2Batch:
    tensors: list[torch.Tensor]
    offsets: list[int]
    lengths: list[int]
    tensor_offsets: list[int]
    tensor_row_bytes: list[int]
    tensor_row_strides: list[int]
    peers: list[int]
    ordinals: list[int]
    region_indices: list[int]
    region_bytes: list[int]
    expected_counts: list[int]
    copybacks: list[tuple[torch.Tensor, torch.Tensor]]


def _candidate_include_paths() -> list[str]:
    values = []
    configured = os.environ.get("AWEX_NCCL_INCLUDE", "")
    if configured:
        values.extend(path for path in configured.split(os.pathsep) if path)
    values.extend(
        path
        for path in (
            "/usr/local/cuda/include",
            "/usr/local/nccl/include",
            "/opt/nccl/include",
        )
        if os.path.isdir(path)
    )
    return list(dict.fromkeys(values))


def _candidate_library_paths() -> list[str]:
    values = []
    configured = os.environ.get("AWEX_NCCL_LIB", "")
    if configured:
        values.extend(path for path in configured.split(os.pathsep) if path)
    values.extend(
        path
        for path in (
            "/usr/local/cuda/lib64",
            "/usr/local/nccl/lib",
            "/opt/nccl/lib",
        )
        if os.path.isdir(path)
    )
    return list(dict.fromkeys(values))


def _operation_groups(
    plan: TransferPlan, rank: int, world_size: int
) -> list[tuple[int, list[CommunicationOperation]]]:
    peers = sorted(plan.operations)
    if any(peer < 0 or peer >= world_size for peer in peers):
        raise NCCLDeviceV2UnavailableError(
            f"v2 plan contains an invalid peer for world_size={world_size}: {peers}"
        )
    if rank in peers:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 does not support self-transfer operations"
        )
    return [(peer, list(plan.operations[peer])) for peer in peers]


def _resolve_chunk_bytes(chunk_bytes: int | None) -> int:
    if chunk_bytes is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_CHUNK_BYTES")
        try:
            chunk_bytes = 4 * 1024 * 1024 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_CHUNK_BYTES must be an integer"
            ) from exc
    chunk_bytes = int(chunk_bytes)
    if chunk_bytes < 0:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 chunk_bytes must be non-negative"
        )
    if chunk_bytes and chunk_bytes % 16 != 0:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 chunk_bytes must be a multiple of 16"
        )
    return chunk_bytes


def _resolve_network_step_bytes(network_step_bytes: int | None) -> int:
    if network_step_bytes is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES")
        if configured is None:
            configured = os.environ.get("NCCL_P2P_NET_CHUNKSIZE")
        try:
            network_step_bytes = 128 * 1024 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES must be an integer"
            ) from exc
    network_step_bytes = int(network_step_bytes)
    if network_step_bytes <= 0:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 network_step_bytes must be positive"
        )
    if network_step_bytes % 16 != 0:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 network_step_bytes must be a multiple of 16"
        )
    return network_step_bytes


def _resolve_fifo_depth(fifo_depth: int | None = None) -> int:
    if fifo_depth is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_FIFO_DEPTH")
        try:
            fifo_depth = 16 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_FIFO_DEPTH must be an integer"
            ) from exc
    fifo_depth = int(fifo_depth)
    if fifo_depth < 1 or fifo_depth > 64:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 FIFO depth must be in [1, 64]"
        )
    return fifo_depth


def _resolve_network_channels_per_peer(
    network_channels_per_peer: int | None,
) -> int:
    if network_channels_per_peer is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_NET_CHANNELS_PER_PEER")
        if configured is None:
            configured = os.environ.get("NCCL_NCHANNELS_PER_NET_PEER")
        try:
            network_channels_per_peer = 0 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_NET_CHANNELS_PER_PEER must be an integer"
            ) from exc
    network_channels_per_peer = int(network_channels_per_peer)
    if network_channels_per_peer < 0 or network_channels_per_peer > 64:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 network_channels_per_peer must be in [0, 64]"
        )
    return network_channels_per_peer


def _detect_active_rdma_device_count(
    sysfs_root: str = "/sys/class/infiniband",
) -> int:
    """Count active RDMA devices backed by a visible network interface."""

    return len(_active_rdma_devices(sysfs_root))


def _resolve_gin_connections(gin_connections: int | None) -> int:
    if gin_connections is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS")
        if configured is None:
            configured = os.environ.get("NCCL_GIN_NCONNECTIONS")
        try:
            gin_connections = (
                min(4, _detect_active_rdma_device_count()) or 4
                if configured is None
                else int(configured)
            )
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS must be an integer"
            ) from exc
    gin_connections = int(gin_connections)
    if gin_connections < 0 or gin_connections > 4:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 GIN connections must be in [0, 4]"
        )
    return gin_connections


def _resolve_gin_context_count(gin_context_count: int | None) -> int:
    if gin_context_count is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_GIN_CONTEXTS")
        try:
            gin_context_count = 0 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_GIN_CONTEXTS must be an integer"
            ) from exc
    gin_context_count = int(gin_context_count)
    if gin_context_count < 0 or gin_context_count > 64:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 GIN contexts must be in [0, 64]"
        )
    return gin_context_count


def _resolve_gin_doorbell_batch(gin_doorbell_batch: int | None) -> int:
    if gin_doorbell_batch is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_GIN_DOORBELL_BATCH")
        try:
            gin_doorbell_batch = 1 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_GIN_DOORBELL_BATCH must be an integer"
            ) from exc
    gin_doorbell_batch = int(gin_doorbell_batch)
    if gin_doorbell_batch < 1 or gin_doorbell_batch > 8:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 GIN doorbell batch must be in [1, 8]"
        )
    return gin_doorbell_batch


def _resolve_gin_reliable_doorbell(gin_reliable_doorbell: int | None) -> int:
    if gin_reliable_doorbell is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_GIN_RELIABLE_DB")
        if configured is None:
            configured = os.environ.get("NCCL_GIN_GDAKI_USE_RELIABLE_DB")
        try:
            gin_reliable_doorbell = 2 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_GIN_RELIABLE_DB must be an integer"
            ) from exc
    gin_reliable_doorbell = int(gin_reliable_doorbell)
    if gin_reliable_doorbell < 0 or gin_reliable_doorbell > 2:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 GIN reliable doorbell mode must be in [0, 2]"
        )
    return gin_reliable_doorbell


def _sequence_from_step(step_id: int) -> int:
    sequence = int(step_id) + 2
    if sequence <= 0:
        raise NCCLDeviceV2UnavailableError(
            f"nccl_device_v2 step_id must be at least -1, got {step_id}."
        )
    return sequence


def _ensure_cuda_tensor(tensor: torch.Tensor, description: str) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise NCCLDeviceV2UnavailableError(
            f"nccl_device_v2 only supports CUDA tensors ({description})"
        )


def _tensor_copy_layout(tensor: torch.Tensor, name: str) -> tuple[int, int]:
    element_size = int(tensor.element_size())
    total_bytes = int(tensor.numel()) * element_size
    if tensor.is_contiguous() or tensor.dim() < 2:
        return total_bytes, total_bytes
    if int(tensor.stride(-1)) != 1:
        raise NCCLDeviceV2UnavailableError(
            "v2 only supports views contiguous in their innermost dimension: "
            f"parameter={name}, shape={tuple(tensor.shape)}, "
            f"stride={tuple(tensor.stride())}"
        )
    for dimension in range(tensor.dim() - 2):
        expected = int(tensor.shape[dimension + 1]) * int(tensor.stride(dimension + 1))
        if int(tensor.stride(dimension)) != expected:
            raise NCCLDeviceV2UnavailableError(
                "v2 tensor view cannot be represented by one row stride: "
                f"parameter={name}, shape={tuple(tensor.shape)}, "
                f"stride={tuple(tensor.stride())}"
            )
    row_bytes = int(tensor.shape[-1]) * element_size
    row_stride = int(tensor.stride(-2)) * element_size
    if row_bytes <= 0 or row_stride < row_bytes:
        raise NCCLDeviceV2UnavailableError(
            f"v2 tensor row stride is invalid: parameter={name}"
        )
    return row_bytes, row_stride


def _append_tensor_range(
    *,
    tensors: list[torch.Tensor],
    tensor_offsets: list[int],
    tensor_row_bytes: list[int],
    tensor_row_strides: list[int],
    offsets: list[int],
    lengths: list[int],
    peers: list[int],
    ordinals: list[int],
    region_indices: list[int],
    tensor: torch.Tensor,
    tensor_offset: int,
    nbytes: int,
    row_bytes: int,
    row_stride: int,
    peer: int,
    peer_offset: int,
    ordinal: int,
    region_index: int,
) -> int:
    # Preserve one descriptor per physical TransferPlan span. C++ concatenates
    # these descriptors into a virtual peer stream before selecting channels
    # and lowering transport chunks, so chunk boundaries may cross tensors.
    tensors.append(tensor)
    tensor_offsets.append(tensor_offset)
    tensor_row_bytes.append(row_bytes)
    tensor_row_strides.append(row_stride)
    offsets.append(peer_offset)
    lengths.append(nbytes)
    peers.append(peer)
    ordinals.append(ordinal)
    region_indices.append(region_index)
    return ordinal + 1


def _build_send_batch(
    parameters: dict,
    plan: TransferPlan,
    rank: int,
    world_size: int,
    chunk_bytes: int,
    allow_staging: bool = True,
) -> _V2Batch:
    _resolve_chunk_bytes(chunk_bytes)
    tensors: list[torch.Tensor] = []
    offsets: list[int] = []
    lengths: list[int] = []
    tensor_offsets: list[int] = []
    tensor_row_bytes: list[int] = []
    tensor_row_strides: list[int] = []
    peers: list[int] = []
    ordinals: list[int] = []
    region_indices: list[int] = []
    region_bytes = [0] * world_size
    expected_counts = [0] * world_size
    context = {}
    for peer, operations in _operation_groups(plan, rank, world_size):
        peer_offset = 0
        ordinal = 0
        for op in operations:
            parameter = parameters[op.send_shard_meta.name]
            if isinstance(parameter, StaticTensorLayout):
                if (
                    op.send_tensor_span_numels
                    and parameter.span_numels != op.send_tensor_span_numels
                ):
                    raise NCCLDeviceV2UnavailableError(
                        "Compiled source layout does not match transfer plan for "
                        f"{op.send_shard_meta.name}"
                    )
                fragments = parameter.slice(op.train_slices)
            else:
                if allow_staging:
                    tensor = slice_tensor(parameter, op, True, slice_context=context)
                else:
                    tensor = parameter[op.train_slices]
                if allow_staging and not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                fragments = [tensor]
            for fragment in fragments:
                _ensure_cuda_tensor(fragment, op.send_shard_meta.name)
                length = int(fragment.numel()) * int(fragment.element_size())
                row_bytes, row_stride = _tensor_copy_layout(
                    fragment, op.send_shard_meta.name
                )
                ordinal = _append_tensor_range(
                    tensors=tensors,
                    tensor_offsets=tensor_offsets,
                    tensor_row_bytes=tensor_row_bytes,
                    tensor_row_strides=tensor_row_strides,
                    offsets=offsets,
                    lengths=lengths,
                    peers=peers,
                    ordinals=ordinals,
                    region_indices=region_indices,
                    tensor=fragment,
                    tensor_offset=0,
                    nbytes=length,
                    row_bytes=row_bytes,
                    row_stride=row_stride,
                    peer=peer,
                    peer_offset=peer_offset,
                    ordinal=ordinal,
                    region_index=rank,
                )
                peer_offset += length
        expected_counts[peer] = ordinal
        region_bytes[rank] = max(region_bytes[rank], peer_offset)
    return _V2Batch(
        tensors=tensors,
        offsets=offsets,
        lengths=lengths,
        tensor_offsets=tensor_offsets,
        tensor_row_bytes=tensor_row_bytes,
        tensor_row_strides=tensor_row_strides,
        peers=peers,
        ordinals=ordinals,
        region_indices=region_indices,
        region_bytes=region_bytes,
        expected_counts=expected_counts,
        copybacks=[],
    )


def _build_recv_batch(
    parameters: dict,
    plan: TransferPlan,
    rank: int,
    world_size: int,
    chunk_bytes: int,
    allow_staging: bool = True,
) -> _V2Batch:
    _resolve_chunk_bytes(chunk_bytes)
    tensors: list[torch.Tensor] = []
    offsets: list[int] = []
    lengths: list[int] = []
    tensor_offsets: list[int] = []
    tensor_row_bytes: list[int] = []
    tensor_row_strides: list[int] = []
    peers: list[int] = []
    ordinals: list[int] = []
    region_indices: list[int] = []
    region_bytes = [0] * world_size
    expected_counts = [0] * world_size
    copybacks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for peer, operations in _operation_groups(plan, rank, world_size):
        peer_offset = 0
        ordinal = 0
        for op in operations:
            parameter = parameters[op.recv_shard_meta.name]
            view = parameter[op.inf_slices]
            target = view
            try:
                row_bytes, row_stride = _tensor_copy_layout(
                    target, op.recv_shard_meta.name
                )
            except NCCLDeviceV2UnavailableError:
                if not allow_staging:
                    raise
                target = torch.empty_like(view, memory_format=torch.contiguous_format)
                copybacks.append((view, target))
                row_bytes, row_stride = _tensor_copy_layout(
                    target, op.recv_shard_meta.name
                )
            _ensure_cuda_tensor(target, op.recv_shard_meta.name)
            fragment_numels = [int(target.numel())]
            if op.send_tensor_span_numels:
                layout_fragments = slice_layout_fragments(
                    op.send_shard_meta.shape,
                    op.train_slices,
                    op.send_tensor_span_numels,
                )
                fragment_numels = [fragment[2] for fragment in layout_fragments]
                if sum(fragment_numels) != target.numel():
                    raise NCCLDeviceV2UnavailableError(
                        "Source layout slice does not match receive tensor for "
                        f"{op.recv_shard_meta.name}"
                    )
            target_offset = 0
            element_size = int(target.element_size())
            for fragment_numel in fragment_numels:
                length = int(fragment_numel) * element_size
                ordinal = _append_tensor_range(
                    tensors=tensors,
                    tensor_offsets=tensor_offsets,
                    tensor_row_bytes=tensor_row_bytes,
                    tensor_row_strides=tensor_row_strides,
                    offsets=offsets,
                    lengths=lengths,
                    peers=peers,
                    ordinals=ordinals,
                    region_indices=region_indices,
                    tensor=target,
                    tensor_offset=target_offset,
                    nbytes=length,
                    row_bytes=row_bytes,
                    row_stride=row_stride,
                    peer=peer,
                    peer_offset=peer_offset,
                    ordinal=ordinal,
                    region_index=peer,
                )
                target_offset += length
                peer_offset += length
        expected_counts[peer] = ordinal
        region_bytes[peer] = max(region_bytes[peer], peer_offset)
    return _V2Batch(
        tensors=tensors,
        offsets=offsets,
        lengths=lengths,
        tensor_offsets=tensor_offsets,
        tensor_row_bytes=tensor_row_bytes,
        tensor_row_strides=tensor_row_strides,
        peers=peers,
        ordinals=ordinals,
        region_indices=region_indices,
        region_bytes=region_bytes,
        expected_counts=expected_counts,
        copybacks=copybacks,
    )


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise NCCLDeviceV2UnavailableError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise NCCLDeviceV2UnavailableError(f"{name} must be >= {minimum}")
    return parsed


def _load_extension() -> Any:
    """Build/load the v2 CUDA extension on first use."""

    global _extension
    if _extension is not None:
        return _extension
    with _extension_lock:
        if _extension is not None:
            return _extension
        if device_util.get_device_type() != "cuda" or not torch.cuda.is_available():
            raise NCCLDeviceV2UnavailableError(
                "nccl_device_v2 requires a CUDA runtime and cannot run on "
                "non-CUDA devices."
            )

        module_name = os.environ.get("AWEX_NCCL_DEVICE_V2_EXTENSION", "")
        if module_name:
            try:
                _extension = __import__(module_name, fromlist=["*"])
                return _extension
            except Exception as exc:
                raise NCCLDeviceV2UnavailableError(
                    "Failed to import AWEX_NCCL_DEVICE_V2_EXTENSION="
                    f"{module_name!r}: {exc}"
                ) from exc

        include_paths = _candidate_include_paths()
        header_candidates = [
            candidate
            for path in include_paths
            for candidate in (
                os.path.join(path, "nccl_device.h"),
                os.path.join(path, "nccl_device", "core.h"),
            )
        ]
        if not any(os.path.exists(path) for path in header_candidates):
            raise NCCLDeviceV2UnavailableError(
                "NCCL Device API headers were not found. Set AWEX_NCCL_INCLUDE to "
                "the NCCL include directory containing nccl_device.h."
            )

        try:
            from torch.utils.cpp_extension import load
        except Exception as exc:
            raise NCCLDeviceV2UnavailableError(
                "torch.utils.cpp_extension is required to build nccl_device_v2."
            ) from exc

        source = os.path.join(
            os.path.dirname(__file__),
            "nccl_device_v2",
            "nccl_device_v2_ext.cu",
        )
        kernel_source = os.path.join(
            os.path.dirname(__file__),
            "nccl_device_v2",
            "device_v2_launch.cu",
        )
        if not os.path.exists(source):
            raise NCCLDeviceV2UnavailableError(f"Missing CUDA source: {source}")
        if not os.path.exists(kernel_source):
            raise NCCLDeviceV2UnavailableError(f"Missing CUDA source: {kernel_source}")

        library_paths = _candidate_library_paths()
        try:
            _extension = load(
                name="awex_nccl_device_ext_v2",
                sources=[source, kernel_source],
                extra_include_paths=include_paths,
                extra_cuda_cflags=[
                    "-O3",
                    "-DNCCL_OS_LINUX",
                    "--expt-extended-lambda",
                    "--expt-relaxed-constexpr",
                    "-Xptxas=-maxrregcount=96",
                ],
                extra_ldflags=[
                    *(f"-L{path}" for path in library_paths),
                    *(
                        f"-Wl,--disable-new-dtags,-rpath,{path}"
                        for path in library_paths
                    ),
                    "-lnccl",
                    "-ldl",
                ],
                with_cuda=True,
                verbose=os.environ.get("AWEX_NCCL_DEVICE_VERBOSE_BUILD", "0") == "1",
            )
        except Exception as exc:
            raise NCCLDeviceV2UnavailableError(
                "Failed to build the NCCL Device v2 extension. Check CUDA/NCCL "
                "versions and AWEX_NCCL_INCLUDE/AWEX_NCCL_LIB."
            ) from exc
        return _extension


class NCCLDeviceV2Transport:
    """NCCL-style channel/work/FIFO transport for a fixed TransferPlan."""

    def __init__(
        self,
        group: Any,
        rank: int,
        world_size: int,
        timeout_ms: int | None = None,
        chunk_bytes: int | None = None,
        infer_instance_world_size: int = 0,
        num_infer_engines: int = 1,
        network_step_bytes: int | None = None,
        network_channels_per_peer: int | None = None,
        gin_connections: int | None = None,
        gin_context_count: int | None = None,
        gin_doorbell_batch: int | None = None,
        gin_reliable_doorbell: int | None = None,
    ):
        if world_size < 2 or world_size > 256:
            raise NCCLDeviceV2UnavailableError(
                f"nccl_device_v2 supports world_size in [2, 256], got {world_size}."
            )
        self.group = group
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.timeout_ms = int(
            timeout_ms or os.environ.get("AWEX_NCCL_DEVICE_TIMEOUT_MS", "120000")
        )
        self.chunk_bytes = _resolve_chunk_bytes(chunk_bytes)
        self.max_channels = _env_int("AWEX_NCCL_DEVICE_V2_MAX_CHANNELS", 64, minimum=1)
        if self.max_channels > 64:
            raise NCCLDeviceV2UnavailableError(
                "nccl_device_v2 max_channels must be at most 64"
            )
        self.fifo_depth = _resolve_fifo_depth()
        self.step_bytes = _env_int(
            "AWEX_NCCL_DEVICE_V2_STEP_BYTES", 512 * 1024, minimum=1
        )
        self.network_step_bytes = _resolve_network_step_bytes(network_step_bytes)
        self.requested_network_channels_per_peer = (
            _resolve_network_channels_per_peer(network_channels_per_peer)
        )
        self.gin_connections = _resolve_gin_connections(gin_connections)
        self.gin_context_count = _resolve_gin_context_count(gin_context_count)
        if self.gin_context_count == 0:
            # NCCL 2.30.4 does not round a one-context request up to the
            # negotiated connection count. Keep every connection addressable.
            self.gin_context_count = self.gin_connections or 1
        self.gin_doorbell_batch = _resolve_gin_doorbell_batch(gin_doorbell_batch)
        self.gin_reliable_doorbell = _resolve_gin_reliable_doorbell(
            gin_reliable_doorbell
        )
        if self.gin_connections:
            os.environ["NCCL_GIN_NCONNECTIONS"] = str(self.gin_connections)
        else:
            os.environ.pop("NCCL_GIN_NCONNECTIONS", None)
        os.environ["NCCL_GIN_GDAKI_USE_RELIABLE_DB"] = str(
            self.gin_reliable_doorbell
        )
        if self.chunk_bytes and self.chunk_bytes < max(
            self.step_bytes, self.network_step_bytes
        ):
            raise NCCLDeviceV2UnavailableError(
                "nccl_device_v2 chunk_bytes must be at least both local and network "
                "step_bytes"
            )
        self.infer_instance_world_size = int(infer_instance_world_size)
        self.num_infer_engines = int(num_infer_engines)
        self._extension = None
        self._handle: int | None = None
        self._initialized = False
        self._logged_batch_shape = False
        self._prepared_send = None
        self._prepared_recv = None
        logger.info(
            "Configured nccl_device_v2 rank=%s chunk_bytes=%s max_channels=%s "
            "fifo_depth=%s step_bytes=%s network_step_bytes=%s "
            "requested_network_channels_per_peer=%s gin_connections=%s "
            "gin_context_count=%s gin_doorbell_batch=%s "
            "gin_reliable_doorbell=%s hca_policy=%s selected_hca=%s",
            self.rank,
            self.chunk_bytes,
            self.max_channels,
            self.fifo_depth,
            self.step_bytes,
            self.network_step_bytes,
            self.requested_network_channels_per_peer,
            self.gin_connections,
            self.gin_context_count,
            self.gin_doorbell_batch,
            self.gin_reliable_doorbell,
            os.environ.get("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced"),
            os.environ.get("NCCL_IB_HCA", "topology"),
        )

    def _ensure_initialized(self) -> float:
        if self._initialized:
            return 0.0
        start_time = time.perf_counter()
        self._extension = _load_extension()
        device = torch.device(device_util.get_torch_device())
        unique_id_size = int(self._extension.unique_id_size())
        unique_id_tensor = torch.empty(unique_id_size, dtype=torch.uint8, device=device)
        if self.rank == 0:
            unique_id = self._extension.get_unique_id()
            if len(unique_id) != unique_id_size:
                raise NCCLDeviceV2UnavailableError(
                    "NCCL unique-id size returned by the v2 extension is inconsistent."
                )
            unique_id_tensor.copy_(
                torch.tensor(list(unique_id), dtype=torch.uint8, device=device)
            )
        dist.broadcast(unique_id_tensor, src=0, group=self.group)
        unique_id = bytes(unique_id_tensor.cpu().tolist())
        self._handle = int(
            self._extension.create(
                unique_id,
                self.world_size,
                self.rank,
                int(device.index or 0),
                self.timeout_ms,
                self.max_channels,
                self.fifo_depth,
                self.step_bytes,
                self.network_step_bytes,
                self.chunk_bytes,
                self.requested_network_channels_per_peer,
                self.gin_context_count,
                self.gin_doorbell_batch,
            )
        )
        self._initialized = True
        logger.info(
            "Initialized nccl_device_v2 rank=%s world_size=%s window_config="
            "channels:%s fifo:%s step_bytes:%s network_step_bytes:%s "
            "requested_network_channels_per_peer:%s gin_connections:%s "
            "gin_context_count:%s gin_doorbell_batch:%s "
            "gin_reliable_doorbell:%s",
            self.rank,
            self.world_size,
            self.max_channels,
            self.fifo_depth,
            self.step_bytes,
            self.network_step_bytes,
            self.requested_network_channels_per_peer,
            self.gin_connections,
            self.gin_context_count,
            self.gin_doorbell_batch,
            self.gin_reliable_doorbell,
        )
        return (time.perf_counter() - start_time) * 1000.0

    def _run(self, batch: _V2Batch, sender: bool, sequence: int) -> dict[str, float]:
        run_start = time.perf_counter()
        if not self._logged_batch_shape:
            logger.info(
                "Lowered nccl_device_v2 plan rank=%s sender=%s spans=%s "
                "payload_bytes=%s chunk_bytes=%s expected_counts=%s",
                self.rank,
                sender,
                len(batch.tensors),
                sum(batch.lengths),
                self.chunk_bytes,
                batch.expected_counts,
            )
            self._logged_batch_shape = True
        init_time_ms = self._ensure_initialized()
        assert self._extension is not None and self._handle is not None
        extension_metrics = dict(
            self._extension.launch(
                self._handle,
                batch.tensors,
                batch.lengths,
                batch.tensor_offsets,
                batch.tensor_row_bytes,
                batch.tensor_row_strides,
                batch.peers,
                batch.ordinals,
                batch.expected_counts,
                bool(sender),
                int(sequence),
            )
        )
        extension_metrics["hca_policy"] = os.environ.get(
            "AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced"
        )
        extension_metrics["selected_hca"] = os.environ.get(
            "NCCL_IB_HCA", "topology"
        )
        selected_hca_bandwidth = os.environ.get(
            "AWEX_NCCL_DEVICE_V2_SELECTED_HCA_BANDWIDTH_GBPS"
        )
        if selected_hca_bandwidth is not None:
            extension_metrics["selected_hca_bandwidth_gbps"] = float(
                selected_hca_bandwidth
            )
        copyback_start = time.perf_counter()
        if batch.copybacks:
            with torch.no_grad():
                for destination, staging in batch.copybacks:
                    destination.copy_(staging)
            torch.cuda.current_stream().synchronize()
        copyback_time_ms = (time.perf_counter() - copyback_start) * 1000.0
        extension_metrics.update(
            {
                "payload_bytes": float(sum(batch.lengths)),
                "transport_init_time_ms": init_time_ms,
                "python_copyback_time_ms": copyback_time_ms,
                "transport_total_time_ms": (time.perf_counter() - run_start) * 1000.0,
                "gin_reliable_doorbell_mode": self.gin_reliable_doorbell,
            }
        )
        extension_metrics["reader_copyback_total_time_ms"] = (
            extension_metrics.get("reader_copyback_time_ms", 0.0) + copyback_time_ms
        )
        return extension_metrics

    def send(
        self, parameters: dict, plan: TransferPlan, step_id: int
    ) -> dict[str, float]:
        build_batch_time_ms = 0.0
        prepared = self._prepared_send
        if prepared is not None and prepared[0] is parameters and prepared[1] is plan:
            batch = prepared[2]
        else:
            build_start = time.perf_counter()
            batch = _build_send_batch(
                parameters,
                plan,
                self.rank,
                self.world_size,
                self.chunk_bytes,
            )
            build_batch_time_ms = (time.perf_counter() - build_start) * 1000.0
        metrics = self._run(batch, sender=True, sequence=_sequence_from_step(step_id))
        metrics["build_batch_time_ms"] = build_batch_time_ms
        return metrics

    def recv(
        self, parameters: dict, plan: TransferPlan, step_id: int
    ) -> dict[str, float]:
        build_batch_time_ms = 0.0
        prepared = self._prepared_recv
        if prepared is not None and prepared[0] is parameters and prepared[1] is plan:
            batch = prepared[2]
        else:
            build_start = time.perf_counter()
            batch = _build_recv_batch(
                parameters,
                plan,
                self.rank,
                self.world_size,
                self.chunk_bytes,
            )
            build_batch_time_ms = (time.perf_counter() - build_start) * 1000.0
        metrics = self._run(batch, sender=False, sequence=_sequence_from_step(step_id))
        metrics["build_batch_time_ms"] = build_batch_time_ms
        return metrics

    def prepare_send(
        self,
        parameters: dict,
        plan: TransferPlan,
        allow_staging: bool = True,
    ) -> None:
        batch = _build_send_batch(
            parameters,
            plan,
            self.rank,
            self.world_size,
            self.chunk_bytes,
            allow_staging=allow_staging,
        )
        self._prepared_send = (parameters, plan, batch)

    def prepare_recv(
        self,
        parameters: dict,
        plan: TransferPlan,
        allow_staging: bool = True,
    ) -> None:
        batch = _build_recv_batch(
            parameters,
            plan,
            self.rank,
            self.world_size,
            self.chunk_bytes,
            allow_staging=allow_staging,
        )
        self._prepared_recv = (parameters, plan, batch)

    def close(self) -> None:
        try:
            if self._handle is not None and self._extension is not None:
                self._extension.destroy(self._handle)
        finally:
            self._handle = None
            self._initialized = False
            self._prepared_send = None
            self._prepared_recv = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

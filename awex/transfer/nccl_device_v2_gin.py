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

"""Host configuration for the NCCL Device v2 GIN transport."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass


class NCCLDeviceV2UnavailableError(RuntimeError):
    """Raised when the NCCL Device v2 transport cannot be initialized."""


@dataclass(frozen=True, slots=True)
class _RdmaEndpoint:
    name: str
    port: int
    bandwidth_gbps: float
    pci_path: str


_HCA_DISTANCE_SCORES = {
    "PIX": 0,
    "PXB": 1,
    "PHB": 2,
    "NODE": 3,
    "SYS": 4,
}
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _active_rdma_endpoints(
    sysfs_root: str = "/sys/class/infiniband",
) -> list[_RdmaEndpoint]:
    """Return active RDMA ports and their link capacities in PCI order."""

    try:
        devices = list(os.scandir(sysfs_root))
    except OSError:
        return []
    endpoints = []
    for device in devices:
        try:
            if not any(os.scandir(os.path.join(device.path, "device", "net"))):
                continue
            ports = list(os.scandir(os.path.join(device.path, "ports")))
        except OSError:
            continue
        for port in ports:
            try:
                with open(
                    os.path.join(port.path, "state"), encoding="ascii"
                ) as state_file:
                    if "ACTIVE" not in state_file.read():
                        continue
                port_number = int(port.name)
            except (OSError, ValueError):
                continue
            try:
                with open(
                    os.path.join(port.path, "rate"), encoding="ascii"
                ) as rate_file:
                    rate = re.search(
                        r"([0-9]+(?:\.[0-9]+)?)\s*Gb/sec", rate_file.read()
                    )
            except OSError:
                rate = None
            endpoints.append(
                _RdmaEndpoint(
                    name=device.name,
                    port=port_number,
                    bandwidth_gbps=max(1.0, float(rate.group(1)) if rate else 1.0),
                    pci_path=os.path.realpath(os.path.join(device.path, "device")),
                )
            )
    return sorted(
        endpoints,
        key=lambda endpoint: (endpoint.pci_path, endpoint.port, endpoint.name),
    )


def _active_rdma_devices(
    sysfs_root: str = "/sys/class/infiniband",
) -> list[str]:
    return list(
        dict.fromkeys(endpoint.name for endpoint in _active_rdma_endpoints(sysfs_root))
    )


def _weighted_hca_assignments(
    endpoints: list[_RdmaEndpoint],
    rank_payload_bytes: list[int],
    topology_distances: list[list[int]] | None = None,
) -> list[_RdmaEndpoint]:
    """Assign rank loads to HCA capacity while preserving PCI locality."""

    if not endpoints or not rank_payload_bytes:
        return []
    payloads = [max(1, int(payload)) for payload in rank_payload_bytes]
    if topology_distances is not None:
        if len(topology_distances) != len(payloads) or any(
            len(distances) != len(endpoints) for distances in topology_distances
        ):
            raise NCCLDeviceV2UnavailableError(
                "GPU/HCA topology dimensions do not match local ranks and RDMA ports"
            )
        assigned_bytes = [0] * len(endpoints)
        assignments: list[_RdmaEndpoint | None] = [None] * len(payloads)

        def rank_key(rank: int) -> tuple[int, int, int, int]:
            local = sorted(
                distance
                for distance in topology_distances[rank]
                if distance < _HCA_DISTANCE_SCORES["SYS"]
            )
            best = local[0] if local else _HCA_DISTANCE_SCORES["SYS"]
            second = local[1] if len(local) > 1 else best + 1
            return (-payloads[rank], -(second - best), best, rank)

        for rank in sorted(range(len(payloads)), key=rank_key):
            distances = topology_distances[rank]
            candidates = [
                index
                for index, distance in enumerate(distances)
                if distance < _HCA_DISTANCE_SCORES["SYS"]
            ] or list(range(len(endpoints)))
            endpoint_index = min(
                candidates,
                key=lambda candidate: (
                    distances[candidate],
                    (assigned_bytes[candidate] + payloads[rank])
                    / endpoints[candidate].bandwidth_gbps,
                    assigned_bytes[candidate] / endpoints[candidate].bandwidth_gbps,
                    candidate,
                ),
            )
            assignments[rank] = endpoints[endpoint_index]
            assigned_bytes[endpoint_index] += payloads[rank]
        return [assignment for assignment in assignments if assignment is not None]

    if len(set(payloads)) == 1:
        total_capacity = sum(endpoint.bandwidth_gbps for endpoint in endpoints)
        assignments = []
        for rank in range(len(payloads)):
            target = (rank + 0.5) * total_capacity / len(payloads)
            cumulative = 0.0
            for endpoint in endpoints:
                cumulative += endpoint.bandwidth_gbps
                if target <= cumulative:
                    assignments.append(endpoint)
                    break
        return assignments

    assigned_bytes = [0] * len(endpoints)
    assignments: list[_RdmaEndpoint | None] = [None] * len(payloads)
    for rank in sorted(range(len(payloads)), key=lambda item: (-payloads[item], item)):
        endpoint_index = min(
            range(len(endpoints)),
            key=lambda candidate: (
                (assigned_bytes[candidate] + payloads[rank])
                / endpoints[candidate].bandwidth_gbps,
                assigned_bytes[candidate] / endpoints[candidate].bandwidth_gbps,
                candidate,
            ),
        )
        assignments[rank] = endpoints[endpoint_index]
        assigned_bytes[endpoint_index] += payloads[rank]
    return [assignment for assignment in assignments if assignment is not None]


def _parse_nvidia_topology(
    output: str, endpoints: list[_RdmaEndpoint], gpu_ids: list[str]
) -> list[list[int]] | None:
    output = _ANSI_ESCAPE_RE.sub("", output)
    lines = [line.split() for line in output.splitlines() if line.strip()]
    header = next((fields for fields in lines if fields[0] == "GPU0"), None)
    if header is None:
        return None
    nic_names = {}
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
                [_HCA_DISTANCE_SCORES.get(row[column + 1], 5) for column in nic_columns]
            )
        except IndexError:
            return None
    return distances


def _node_local_gpu_ids(local_world_size: int) -> list[str]:
    configured = os.environ.get("AWEX_NODE_LOCAL_GPU_IDS")
    visible = configured or os.environ.get("CUDA_VISIBLE_DEVICES", "")
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
    """Bind each rank to one NUMA-local HCA before NCCL initialization."""

    policy = (
        os.environ.get("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced").strip().lower()
    )
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
                "AWEX_NODE_LOCAL_WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "")
            )
        )
    except (KeyError, ValueError):
        return
    endpoints = _active_rdma_endpoints()
    if not endpoints or local_world_size <= 0 or not 0 <= local_rank < local_world_size:
        return
    assignments = _weighted_hca_assignments(
        endpoints,
        _rank_payload_bytes_from_environment(local_world_size),
        _gpu_hca_topology(endpoints, local_world_size),
    )
    endpoint = assignments[local_rank]
    os.environ["NCCL_IB_HCA"] = f"={endpoint.name}:{endpoint.port}"
    os.environ["AWEX_NCCL_DEVICE_V2_SELECTED_HCA_BANDWIDTH_GBPS"] = str(
        endpoint.bandwidth_gbps
    )


def _detect_active_rdma_device_count(
    sysfs_root: str = "/sys/class/infiniband",
) -> int:
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


def _resolve_gin_fifo_depth(gin_fifo_depth: int | None = None) -> int:
    if gin_fifo_depth is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_V2_GIN_FIFO_DEPTH")
        if configured is None:
            # The old branch-wide name is now a GIN-only compatibility alias.
            configured = os.environ.get("AWEX_NCCL_DEVICE_V2_FIFO_DEPTH")
        try:
            gin_fifo_depth = 16 if configured is None else int(configured)
        except ValueError as exc:
            raise NCCLDeviceV2UnavailableError(
                "AWEX_NCCL_DEVICE_V2_GIN_FIFO_DEPTH must be an integer"
            ) from exc
    gin_fifo_depth = int(gin_fifo_depth)
    if gin_fifo_depth < 1 or gin_fifo_depth > 64:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 GIN FIFO depth must be in [1, 64]"
        )
    return gin_fifo_depth


def _gin_chunk_bytes(network_step_bytes: int) -> int:
    """Keep GIN chunking independent from the public LSA chunk setting."""

    return max(4 * 1024 * 1024, int(network_step_bytes))


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

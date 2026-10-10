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

"""Bind fixed plans to the CUDA transport, with reusable prepared batches."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from functools import wraps
from typing import Any

from shardstream.topology import (
    TransportUnavailableError,
    _configure_gin_hca_policy,
    _gin_chunk_bytes,
    _resolve_gin_connections,
    _resolve_gin_fifo_depth,
    _resolve_gin_reliable_doorbell,
)


def _preload_configured_nccl() -> None:
    """Load the configured NCCL before torch can load another SONAME match."""

    library_dir = os.environ.get("SHARDSTREAM_NCCL_LIB", "")
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
                raise TransportUnavailableError(
                    f"Failed to preload NCCL from {candidate}: {exc}"
                ) from exc
            return


_configure_gin_hca_policy()
_preload_configured_nccl()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from shardstream import logging  # noqa: E402
from shardstream._utils import device as device_util  # noqa: E402
from shardstream.layout import (  # noqa: E402
    StaticTensorLayout,
    slice_layout_fragments,
)
from shardstream.membership import (  # noqa: E402
    Participant,
    RolloutMembership,
)
from shardstream.plan import (  # noqa: E402
    CommunicationOperation,
    TransferPlan,
    slice_tensor,
)

logger = logging.getLogger(__name__)

_extension_lock = threading.Lock()
_extension: Any | None = None


@dataclass
class _Batch:
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
    forward_peers: list[int]
    ring_ids: list[int]
    quantization: list[list[int]] = dataclass_field(default_factory=list)
    scale_tensors: list[torch.Tensor] = dataclass_field(default_factory=list)


_NO_PEER = -1
_NO_RING = -1


class _RingOrderStrategy(Enum):
    FIXED = "fixed"
    ROOT_SWIZZLE = "root_swizzle"


def _node_major_communicator_ranks(node_ids: list[int]) -> list[int]:
    """Map logical ranks to node-contiguous communicator ranks."""

    ranks_by_node: dict[int, list[int]] = {}
    for logical_rank, node_id in enumerate(node_ids):
        ranks_by_node.setdefault(int(node_id), []).append(logical_rank)
    logical_to_communicator = [0] * len(node_ids)
    communicator_rank = 0
    for logical_ranks in ranks_by_node.values():
        for logical_rank in logical_ranks:
            logical_to_communicator[logical_rank] = communicator_rank
            communicator_rank += 1
    return logical_to_communicator


def _local_node_id() -> int:
    hostname = socket.gethostname().encode("utf-8")
    return int.from_bytes(hashlib.sha256(hostname).digest()[:8], "little", signed=True)


def _operation_groups(
    plan: TransferPlan, rank: int, world_size: int
) -> list[tuple[int, list[CommunicationOperation]]]:
    peers = sorted(plan.operations)
    if any(peer < 0 or peer >= world_size for peer in peers):
        raise TransportUnavailableError(
            f"Transport plan contains an invalid peer for world_size={world_size}: {peers}"
        )
    if rank in peers:
        raise TransportUnavailableError(
            "transport does not support self-transfer operations"
        )
    return [(peer, list(plan.operations[peer])) for peer in peers]


def _resolve_chunk_bytes(chunk_bytes: int | None) -> int:
    if chunk_bytes is None:
        configured = os.environ.get("SHARDSTREAM_CHUNK_BYTES")
        try:
            chunk_bytes = 4 * 1024 * 1024 if configured is None else int(configured)
        except ValueError as exc:
            raise TransportUnavailableError(
                "SHARDSTREAM_CHUNK_BYTES must be an integer"
            ) from exc
    chunk_bytes = int(chunk_bytes)
    if chunk_bytes < 0:
        raise TransportUnavailableError("transport chunk_bytes must be non-negative")
    if chunk_bytes and chunk_bytes % 16 != 0:
        raise TransportUnavailableError(
            "transport chunk_bytes must be a multiple of 16"
        )
    return chunk_bytes


def _resolve_network_step_bytes(network_step_bytes: int | None) -> int:
    if network_step_bytes is None:
        configured = os.environ.get("SHARDSTREAM_NET_STEP_BYTES")
        if configured is None:
            configured = os.environ.get("NCCL_P2P_NET_CHUNKSIZE")
        try:
            network_step_bytes = 128 * 1024 if configured is None else int(configured)
        except ValueError as exc:
            raise TransportUnavailableError(
                "SHARDSTREAM_NET_STEP_BYTES must be an integer"
            ) from exc
    network_step_bytes = int(network_step_bytes)
    if network_step_bytes <= 0:
        raise TransportUnavailableError("transport network_step_bytes must be positive")
    if network_step_bytes % 16 != 0:
        raise TransportUnavailableError(
            "transport network_step_bytes must be a multiple of 16"
        )
    return network_step_bytes


def _resolve_bool(name: str, value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    configured = os.environ.get(name, "0").strip().lower()
    if configured in {"0", "false", "no", "off"}:
        return False
    if configured in {"1", "true", "yes", "on"}:
        return True
    raise TransportUnavailableError(
        f"{name} must be one of 0/1, false/true, no/yes, or off/on"
    )


def _ring_order(
    root: int,
    peers: list[int],
    strategy: _RingOrderStrategy,
    *,
    infer_instance_world_size: int = 0,
    num_infer_engines: int = 1,
    rollout_node_ids: tuple[Any, ...] | None = None,
) -> list[int]:
    ordered = sorted(peers)
    if len(ordered) < 2:
        return ordered

    if rollout_node_ids is not None:
        if len(rollout_node_ids) != num_infer_engines or infer_instance_world_size <= 0:
            raise TransportUnavailableError("Invalid explicit rollout topology")
        groups: dict[Any, list[int]] = {}
        for peer in ordered:
            groups.setdefault(
                rollout_node_ids[peer // infer_instance_world_size], []
            ).append(peer)
        node_groups = list(groups.values())
        if strategy is _RingOrderStrategy.FIXED:
            return [peer for group in node_groups for peer in group]
        node_offset = int(root) % len(node_groups)
        member_offset = int(root) // len(node_groups)
        swizzled = []
        for i in range(len(node_groups)):
            group = node_groups[(node_offset + i) % len(node_groups)]
            offset = member_offset % len(group)
            swizzled.extend(group[offset:] + group[:offset])
        return swizzled

    try:
        local_world_size = int(os.environ.get("SHARDSTREAM_NODE_LOCAL_WORLD_SIZE", "0"))
    except ValueError:
        local_world_size = 0
    instance_world_size = int(infer_instance_world_size)
    engine_count = int(num_infer_engines)
    engines_per_node = (
        local_world_size // instance_world_size
        if instance_world_size > 0 and local_world_size % instance_world_size == 0
        else 0
    )
    node_count = (
        (engine_count + engines_per_node - 1) // engines_per_node
        if engines_per_node > 0
        else 0
    )
    if 1 < node_count <= engine_count:
        # veRL assigns engine groups round-robin across rollout nodes. Grouping
        # equal node slots keeps most relay hops on the NCCL LSA transport.
        node_groups: list[list[int]] = [[] for _ in range(node_count)]
        for peer in ordered:
            engine = peer // instance_world_size
            node_groups[engine % node_count].append(peer)
        if all(node_groups):
            if strategy is _RingOrderStrategy.FIXED:
                return [peer for group in node_groups for peer in group]
            node_offset = int(root) % node_count
            member_offset = (int(root) // node_count) % max(
                len(group) for group in node_groups
            )
            swizzled: list[int] = []
            for node_index in range(node_count):
                group = node_groups[(node_offset + node_index) % node_count]
                offset = member_offset % len(group)
                swizzled.extend(group[offset:] + group[:offset])
            return swizzled

    if strategy is _RingOrderStrategy.FIXED:
        return ordered
    offset = int(root) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _ring_id(root: int, logical_peer: int, instance_world_size: int) -> int:
    return int(root) * int(instance_world_size) + int(logical_peer)


def _reindex_batch_streams(batch: _Batch) -> None:
    next_ordinal = [0] * len(batch.expected_counts)
    for index, peer in enumerate(batch.peers):
        batch.ordinals[index] = next_ordinal[peer]
        next_ordinal[peer] += 1
    batch.expected_counts = next_ordinal


def _apply_send_ring_routes(
    batch: _Batch,
    *,
    root: int,
    infer_instance_world_size: int,
    num_infer_engines: int,
    swizzle: bool,
    rollout_node_ids: tuple[Any, ...] | None = None,
) -> None:
    instance_world_size = int(infer_instance_world_size)
    engine_count = int(num_infer_engines)
    if engine_count < 2 or instance_world_size <= 0:
        return
    infer_world_size = engine_count * instance_world_size
    if infer_world_size > len(batch.expected_counts):
        raise TransportUnavailableError(
            "transport ring inference topology exceeds transfer world size"
        )

    indices_by_peer: dict[int, list[int]] = {}
    for index, peer in enumerate(batch.peers):
        indices_by_peer.setdefault(peer, []).append(index)

    suppressed: set[int] = set()
    ring_by_index: dict[int, int] = {}
    for logical_peer in range(instance_world_size):
        targets = [
            engine * instance_world_size + logical_peer
            for engine in range(engine_count)
        ]
        present_targets = [target for target in targets if target in indices_by_peer]
        if not present_targets:
            continue
        if present_targets != targets:
            raise TransportUnavailableError(
                "transport ring broadcast requires each active logical "
                f"rollout peer to be present in all engines: root={root}, "
                f"logical_peer={logical_peer}, present={present_targets}, "
                f"expected={targets}"
            )

        def signature(index: int) -> tuple[int, ...]:
            tensor = batch.tensors[index]
            return (
                int(tensor.data_ptr()),
                int(batch.offsets[index]),
                int(batch.lengths[index]),
                int(batch.tensor_offsets[index]),
                int(batch.tensor_row_bytes[index]),
                int(batch.tensor_row_strides[index]),
                *(batch.quantization[index][:2] if batch.quantization else []),
            )

        strategy = (
            _RingOrderStrategy.ROOT_SWIZZLE if swizzle else _RingOrderStrategy.FIXED
        )
        order = _ring_order(
            root,
            targets,
            strategy,
            infer_instance_world_size=instance_world_size,
            num_infer_engines=engine_count,
            rollout_node_ids=rollout_node_ids,
        )
        canonical = [signature(index) for index in indices_by_peer[order[0]]]
        if not canonical or any(
            [signature(index) for index in indices_by_peer[target]] != canonical
            for target in order[1:]
        ):
            raise TransportUnavailableError(
                "transport ring broadcast requires identical replicated "
                f"streams: root={root}, logical_peer={logical_peer}"
            )
        route_id = _ring_id(root, logical_peer, instance_world_size)
        for index in indices_by_peer[order[0]]:
            ring_by_index[index] = route_id
        suppressed.update(order[1:])

    if not suppressed:
        return
    keep = [index for index, peer in enumerate(batch.peers) if peer not in suppressed]
    for field in (
        "tensors",
        "offsets",
        "lengths",
        "tensor_offsets",
        "tensor_row_bytes",
        "tensor_row_strides",
        "peers",
        "ordinals",
        "region_indices",
        "forward_peers",
        "ring_ids",
    ):
        values = getattr(batch, field)
        setattr(batch, field, [values[index] for index in keep])
    if batch.quantization:
        batch.quantization = [batch.quantization[index] for index in keep]
    batch.ring_ids = [ring_by_index.get(index, _NO_RING) for index in keep]
    _reindex_batch_streams(batch)


def _apply_recv_ring_routes(
    batch: _Batch,
    *,
    rank: int,
    infer_instance_world_size: int,
    num_infer_engines: int,
    swizzle: bool,
    rollout_node_ids: tuple[Any, ...] | None = None,
) -> None:
    instance_world_size = int(infer_instance_world_size)
    engine_count = int(num_infer_engines)
    if (
        engine_count < 2
        or instance_world_size <= 0
        or rank >= engine_count * instance_world_size
    ):
        return
    logical_peer = int(rank) % instance_world_size
    targets = [
        engine * instance_world_size + logical_peer for engine in range(engine_count)
    ]
    strategy = _RingOrderStrategy.ROOT_SWIZZLE if swizzle else _RingOrderStrategy.FIXED
    for index, root in enumerate(list(batch.peers)):
        order = _ring_order(
            root,
            targets,
            strategy,
            infer_instance_world_size=instance_world_size,
            num_infer_engines=engine_count,
            rollout_node_ids=rollout_node_ids,
        )
        position = order.index(rank)
        batch.peers[index] = root if position == 0 else order[position - 1]
        batch.forward_peers[index] = (
            order[position + 1] if position + 1 < len(order) else _NO_PEER
        )
        batch.ring_ids[index] = _ring_id(root, logical_peer, instance_world_size)
    _reindex_batch_streams(batch)


def _sequence_from_step(step_id: int) -> int:
    sequence = int(step_id) + 2
    if sequence <= 0:
        raise TransportUnavailableError(
            f"transport step_id must be at least -1, got {step_id}."
        )
    return sequence


def _ensure_cuda_tensor(tensor: torch.Tensor, description: str) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise TransportUnavailableError(
            f"transport only supports CUDA tensors ({description})"
        )


def _tensor_copy_layout(tensor: torch.Tensor, name: str) -> tuple[int, int]:
    element_size = int(tensor.element_size())
    total_bytes = int(tensor.numel()) * element_size
    if tensor.is_contiguous():
        return total_bytes, total_bytes
    if tensor.dim() < 2:
        raise TransportUnavailableError(
            f"Transport requires staging for a strided vector: parameter={name}, "
            f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
        )
    if int(tensor.stride(-1)) != 1:
        raise TransportUnavailableError(
            "Transport only supports views contiguous in their innermost dimension: "
            f"parameter={name}, shape={tuple(tensor.shape)}, "
            f"stride={tuple(tensor.stride())}"
        )
    for dimension in range(tensor.dim() - 2):
        expected = int(tensor.shape[dimension + 1]) * int(tensor.stride(dimension + 1))
        if int(tensor.stride(dimension)) != expected:
            raise TransportUnavailableError(
                "Transport tensor view cannot be represented by one row stride: "
                f"parameter={name}, shape={tuple(tensor.shape)}, "
                f"stride={tuple(tensor.stride())}"
            )
    row_bytes = int(tensor.shape[-1]) * element_size
    row_stride = int(tensor.stride(-2)) * element_size
    if row_bytes <= 0 or row_stride < row_bytes:
        raise TransportUnavailableError(
            f"Transport tensor row stride is invalid: parameter={name}"
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


def _fp8_wire_bytes(tensor, block_shape, dtype) -> int:
    br, bc = block_shape
    if (
        tensor.dtype != dtype
        or tensor.ndim != 2
        or tensor.stride(1) != 1
        or tensor.data_ptr() % 16
        or (tensor.stride(0) * tensor.element_size()) % 16
        or tensor.shape[0] % br
        or tensor.shape[1] % bc
    ):
        raise TransportUnavailableError(
            f"FP8 transfer requires block-aligned 2D {dtype} matrices: "
            f"shape={tuple(tensor.shape)}, block={block_shape}, dtype={tensor.dtype}"
        )
    return tensor.numel() // (br * bc) * (br * bc + 16)


def _build_send_batch(
    parameters: dict,
    plan: TransferPlan,
    rank: int,
    world_size: int,
    chunk_bytes: int,
    allow_staging: bool = True,
    infer_instance_world_size: int = 0,
    num_infer_engines: int = 1,
    ring_broadcast: bool = False,
    ring_swizzle: bool = False,
    fp8_block_shape: tuple[int, int] | None = None,
    rollout_node_ids: tuple[Any, ...] | None = None,
) -> _Batch:
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
    quantization = []
    for peer, operations in _operation_groups(plan, rank, world_size):
        peer_offset = 0
        ordinal = 0
        for op in operations:
            if (
                str(getattr(op.recv_shard_meta, "dtype", None)).removeprefix("torch.")
                == "float8_e4m3fn"
                and not fp8_block_shape
            ):
                raise TransportUnavailableError(
                    "BF16-to-FP8 transfer requires the blockwise feature on every rank"
                )
            parameter = parameters[op.send_shard_meta.name]
            if isinstance(parameter, StaticTensorLayout):
                if (
                    op.send_tensor_span_numels
                    and parameter.span_numels != op.send_tensor_span_numels
                ):
                    raise TransportUnavailableError(
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
                if (
                    fp8_block_shape
                    and isinstance(parameter, StaticTensorLayout)
                    and str(op.recv_shard_meta.dtype).removeprefix("torch.")
                    == "float8_e4m3fn"
                ):
                    fragment = fragment.view(-1, op.overlap_shape[-1])
                _ensure_cuda_tensor(fragment, op.send_shard_meta.name)
                length = int(fragment.numel()) * int(fragment.element_size())
                row_bytes, row_stride = _tensor_copy_layout(
                    fragment, op.send_shard_meta.name
                )
                q = [0, 0, 0, 0]
                if (
                    fp8_block_shape
                    and str(op.recv_shard_meta.dtype).removeprefix("torch.")
                    == "float8_e4m3fn"
                ):
                    length = _fp8_wire_bytes(fragment, fp8_block_shape, torch.bfloat16)
                    row_bytes = int(fragment.shape[1]) * 2
                    row_stride = int(fragment.stride(0)) * 2
                    q = [*fp8_block_shape, 0, 0]
                quantization.append(q)
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
    batch = _Batch(
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
        forward_peers=[_NO_PEER] * len(tensors),
        ring_ids=[_NO_RING] * len(tensors),
        quantization=quantization if fp8_block_shape else [],
    )
    if ring_broadcast:
        _apply_send_ring_routes(
            batch,
            root=rank,
            infer_instance_world_size=infer_instance_world_size,
            num_infer_engines=num_infer_engines,
            swizzle=ring_swizzle,
            rollout_node_ids=rollout_node_ids,
        )
    return batch


def _build_recv_batch(
    parameters: dict,
    plan: TransferPlan,
    rank: int,
    world_size: int,
    chunk_bytes: int,
    allow_staging: bool = True,
    infer_instance_world_size: int = 0,
    num_infer_engines: int = 1,
    ring_broadcast: bool = False,
    ring_swizzle: bool = False,
    fp8_block_shape: tuple[int, int] | None = None,
    rollout_node_ids: tuple[Any, ...] | None = None,
) -> _Batch:
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
    quantization = []
    scale_tensors = []
    for peer, operations in _operation_groups(plan, rank, world_size):
        peer_offset = 0
        ordinal = 0
        for op in operations:
            parameter = parameters[op.recv_shard_meta.name]
            if parameter.dtype == torch.float8_e4m3fn and not fp8_block_shape:
                raise TransportUnavailableError(
                    "Native FP8 parameters require the blockwise transfer feature"
                )
            view = parameter[op.inf_slices]
            target = view
            try:
                row_bytes, row_stride = _tensor_copy_layout(
                    target, op.recv_shard_meta.name
                )
            except TransportUnavailableError:
                if not allow_staging:
                    raise
                target = torch.empty_like(view, memory_format=torch.contiguous_format)
                copybacks.append((view, target))
                row_bytes, row_stride = _tensor_copy_layout(
                    target, op.recv_shard_meta.name
                )
            _ensure_cuda_tensor(target, op.recv_shard_meta.name)
            if fp8_block_shape and target.dtype == torch.float8_e4m3fn:
                _fp8_wire_bytes(target, fp8_block_shape, torch.float8_e4m3fn)
                row_bytes = int(target.shape[1])
                row_stride = int(target.stride(0))
            fragment_numels = [int(target.numel())]
            if op.send_tensor_span_numels:
                layout_fragments = slice_layout_fragments(
                    op.send_shard_meta.shape,
                    op.train_slices,
                    op.send_tensor_span_numels,
                )
                fragment_numels = [fragment[2] for fragment in layout_fragments]
                if sum(fragment_numels) != target.numel():
                    raise TransportUnavailableError(
                        "Source layout slice does not match receive tensor for "
                        f"{op.recv_shard_meta.name}"
                    )
            target_offset = 0
            element_size = int(target.element_size())
            for fragment_numel in fragment_numels:
                fragment_target = target
                fragment_offset = target_offset
                length = int(fragment_numel) * element_size
                q = [0, 0, 0, 0]
                if fp8_block_shape and target.dtype == torch.float8_e4m3fn:
                    br, bc = fp8_block_shape
                    if (
                        target.ndim != 2
                        or fragment_numel % target.shape[1]
                        or target_offset % row_bytes
                    ):
                        raise TransportUnavailableError(
                            "FP8 source spans must contain complete matrix rows"
                        )
                    row_start = target_offset // row_bytes
                    row_count = fragment_numel // target.shape[1]
                    fragment_target = target.narrow(0, row_start, row_count)
                    length = _fp8_wire_bytes(
                        fragment_target, fp8_block_shape, torch.float8_e4m3fn
                    )
                    scale = parameters[op.recv_shard_meta.name + "_scale_inv"]
                    scale_slices = []
                    for dim, block in enumerate((br, bc)):
                        start, stop, step = op.inf_slices[dim].indices(
                            parameter.shape[dim]
                        )
                        if step != 1 or start % block or stop % block:
                            raise TransportUnavailableError(
                                "FP8 resharding must align to quantization blocks"
                            )
                        scale_slices.append(slice(start // block, stop // block))
                    scale = scale[tuple(scale_slices)].narrow(
                        0, row_start // br, row_count // br
                    )
                    if scale.dtype != torch.float32 or scale.stride(-1) != 1:
                        raise TransportUnavailableError(
                            "FP8 scales require row-major FP32 storage"
                        )
                    scale_tensors.append(scale)
                    q = [br, bc, int(scale.data_ptr()), int(scale.stride(0)) * 4]
                    fragment_offset = 0
                quantization.append(q)
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
                    tensor=fragment_target,
                    tensor_offset=fragment_offset,
                    nbytes=length,
                    row_bytes=row_bytes,
                    row_stride=row_stride,
                    peer=peer,
                    peer_offset=peer_offset,
                    ordinal=ordinal,
                    region_index=peer,
                )
                target_offset += int(fragment_numel) * element_size
                peer_offset += length
        expected_counts[peer] = ordinal
        region_bytes[peer] = max(region_bytes[peer], peer_offset)
    batch = _Batch(
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
        forward_peers=[_NO_PEER] * len(tensors),
        ring_ids=[_NO_RING] * len(tensors),
        quantization=quantization if fp8_block_shape else [],
        scale_tensors=scale_tensors,
    )
    if ring_broadcast:
        _apply_recv_ring_routes(
            batch,
            rank=rank,
            infer_instance_world_size=infer_instance_world_size,
            num_infer_engines=num_infer_engines,
            swizzle=ring_swizzle,
            rollout_node_ids=rollout_node_ids,
        )
    return batch


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TransportUnavailableError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise TransportUnavailableError(f"{name} must be >= {minimum}")
    return parsed


def _load_extension() -> Any:
    """Load the installed CMake extension once; compilation is an explicit step."""
    global _extension
    if _extension is not None:
        return _extension
    with _extension_lock:
        if _extension is None:
            try:
                from shardstream import _native

                _extension = _native
            except ImportError as exc:
                raise TransportUnavailableError(
                    "Build/install shardstream with SHARDSTREAM_BUILD_PYTHON=ON "
                    "before creating a CUDA transport."
                ) from exc
    return _extension


def _serialized_transport_call(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._operation_lock:
            return method(self, *args, **kwargs)

    return call


class Transport:
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
        gin_connections: int | None = None,
        gin_reliable_doorbell: int | None = None,
        ring_broadcast: bool | None = None,
        ring_swizzle: bool | None = None,
        membership_epoch: int = 0,
        rollout_node_ids: tuple[Any, ...] | None = None,
    ):
        self._operation_lock = threading.RLock()
        if membership_epoch < 0:
            raise TransportUnavailableError("membership_epoch must be nonnegative")
        self.membership_epoch = int(membership_epoch)
        if rollout_node_ids is not None and len(rollout_node_ids) != num_infer_engines:
            raise TransportUnavailableError(
                "Explicit rollout topology must cover every engine"
            )
        self.rollout_node_ids = (
            tuple(rollout_node_ids) if rollout_node_ids is not None else None
        )
        self._membership_binding = None
        self._device_prepared_binding = None
        self._reconfiguration_failed = False
        if world_size < 2 or world_size > 256:
            raise TransportUnavailableError(
                f"transport supports world_size in [2, 256], got {world_size}."
            )
        self.group = group
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.timeout_ms = int(
            timeout_ms or os.environ.get("SHARDSTREAM_TIMEOUT_MS", "120000")
        )
        self.chunk_bytes = _resolve_chunk_bytes(chunk_bytes)
        self.max_channels = _env_int("SHARDSTREAM_MAX_CHANNELS", 64, minimum=1)
        if self.max_channels > 64:
            raise TransportUnavailableError("transport max_channels must be at most 64")
        # Preserve main's LSA protocol. Remote peers use independent GIN
        # settings below and cannot retune local FIFO/chunk behavior.
        self.fifo_depth = 8
        self.step_bytes = _env_int("SHARDSTREAM_STEP_BYTES", 512 * 1024, minimum=1)
        self.network_step_bytes = _resolve_network_step_bytes(network_step_bytes)
        self.gin_fifo_depth = _resolve_gin_fifo_depth()
        self.gin_chunk_bytes = _gin_chunk_bytes(self.network_step_bytes)
        self.gin_connections = _resolve_gin_connections(gin_connections)
        # Keep two independent device queues per physical GIN connection. The
        # context index still round-robins connections, while the second queue
        # avoids serializing every ring lane on one HCA doorbell stream.
        self.gin_context_count = (
            min(self.max_channels, 2 * self.gin_connections)
            if self.gin_connections
            else 1
        )
        self.gin_reliable_doorbell = _resolve_gin_reliable_doorbell(
            gin_reliable_doorbell
        )
        if self.gin_connections:
            os.environ["NCCL_GIN_NCONNECTIONS"] = str(self.gin_connections)
        else:
            os.environ.pop("NCCL_GIN_NCONNECTIONS", None)
        os.environ["NCCL_GIN_GDAKI_USE_RELIABLE_DB"] = str(self.gin_reliable_doorbell)
        if self.chunk_bytes and self.chunk_bytes < self.step_bytes:
            raise TransportUnavailableError(
                "transport chunk_bytes must be at least the LSA step_bytes"
            )
        self.infer_instance_world_size = int(infer_instance_world_size)
        self.num_infer_engines = int(num_infer_engines)
        self.ring_broadcast = _resolve_bool(
            "SHARDSTREAM_RING_BROADCAST", ring_broadcast
        )
        self.ring_swizzle = _resolve_bool("SHARDSTREAM_RING_SWIZZLE", ring_swizzle)
        self.fp8_block_shape = None
        if _resolve_bool("SHARDSTREAM_FP8_BLOCKWISE", None):
            br = _env_int("SHARDSTREAM_FP8_BLOCK_ROWS", 128, 1)
            bc = _env_int("SHARDSTREAM_FP8_BLOCK_COLS", 128, 1)
            if br not in (64, 128) or bc not in (64, 128):
                raise TransportUnavailableError(
                    "FP8 block dimensions must be 64 or 128"
                )
            self.fp8_block_shape = (br, bc)
        self._extension = None
        self._handle: int | None = None
        self._initialized = False
        self._logged_batch_shape = False
        self._prepared_send = None
        self._prepared_recv = None
        self._logical_to_communicator = list(range(self.world_size))
        logger.info(
            "Configured transport rank=%s chunk_bytes=%s max_channels=%s "
            "fifo_depth=%s step_bytes=%s gin_fifo_depth=%s "
            "network_step_bytes=%s gin_chunk_bytes=%s "
            "gin_connections=%s gin_context_count=%s "
            "gin_reliable_doorbell=%s ring_broadcast=%s ring_swizzle=%s "
            "hca_policy=%s selected_hca=%s",
            self.rank,
            self.chunk_bytes,
            self.max_channels,
            self.fifo_depth,
            self.step_bytes,
            self.gin_fifo_depth,
            self.network_step_bytes,
            self.gin_chunk_bytes,
            self.gin_connections,
            self.gin_context_count,
            self.gin_reliable_doorbell,
            self.ring_broadcast,
            self.ring_swizzle,
            os.environ.get("SHARDSTREAM_HCA_POLICY", "balanced"),
            os.environ.get("NCCL_IB_HCA", "topology"),
        )

    @_serialized_transport_call
    def resolve_rollout_topology(self) -> None:
        """Collect actual placement before building rings, once per membership.

        All transfer ranks call this after setting their CUDA device. Explicit
        membership topology already supplies this information. Node groups use
        their first engine's order; node identity values never determine order.
        """
        if not self.ring_broadcast or self.rollout_node_ids is not None:
            return
        nodes = [None] * self.world_size
        dist.all_gather_object(nodes, _local_node_id(), group=self.group)
        engine_nodes = []
        for engine in range(self.num_infer_engines):
            members = nodes[
                engine * self.infer_instance_world_size : (engine + 1)
                * self.infer_instance_world_size
            ]
            if not members or len(set(members)) != 1:
                raise TransportUnavailableError(
                    "Each rollout TP instance must reside on one node"
                )
            engine_nodes.append(members[0])
        self.rollout_node_ids = tuple(engine_nodes)

    def _ensure_initialized(self) -> float:
        if self._reconfiguration_failed:
            raise TransportUnavailableError(
                "A failed membership rebuild must be recovered explicitly"
            )
        if self._initialized:
            return 0.0
        start_time = time.perf_counter()
        self._extension = _load_extension()
        device = torch.device(device_util.get_torch_device())
        feature_sums = torch.tensor(
            [
                int(self.ring_broadcast),
                int(self.ring_swizzle),
                int(self.fp8_block_shape is not None),
            ],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(feature_sums, op=dist.ReduceOp.SUM, group=self.group)
        feature_sums = [int(value) for value in feature_sums.cpu().tolist()]
        if any(value not in (0, self.world_size) for value in feature_sums):
            raise TransportUnavailableError(
                "transport ring feature flags must match on every rank: "
                f"ring_broadcast_sum={feature_sums[0]}, "
                f"ring_swizzle_sum={feature_sums[1]}, world_size={self.world_size}"
            )
        if self.fp8_block_shape:
            block_shape = torch.tensor(
                self.fp8_block_shape, dtype=torch.int32, device=device
            )
            shapes = [torch.empty_like(block_shape) for _ in range(self.world_size)]
            dist.all_gather(shapes, block_shape, group=self.group)
            if any(not torch.equal(block_shape, other) for other in shapes):
                raise TransportUnavailableError(
                    "FP8 block shape must match on every rank"
                )
        local_node_id = torch.tensor(
            [
                _local_node_id(),
                self.membership_epoch,
                self.infer_instance_world_size,
                self.num_infer_engines,
            ],
            dtype=torch.int64,
            device=device,
        )
        gathered_node_ids = [
            torch.empty_like(local_node_id) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_node_ids, local_node_id, group=self.group)
        identities = [value.cpu().tolist() for value in gathered_node_ids]
        if any(value[1:] != identities[0][1:] for value in identities):
            raise TransportUnavailableError(
                "Membership epoch and inference geometry must match on every rank"
            )
        node_ids = [int(value[0]) for value in identities]
        self._logical_to_communicator = _node_major_communicator_ranks(node_ids)
        communicator_rank = self._logical_to_communicator[self.rank]
        unique_id_size = int(self._extension.unique_id_size())
        unique_id_tensor = torch.empty(unique_id_size, dtype=torch.uint8, device=device)
        if self.rank == 0:
            unique_id = self._extension.get_unique_id()
            if len(unique_id) != unique_id_size:
                raise TransportUnavailableError(
                    "NCCL unique-id size returned by the ShardStream extension is inconsistent."
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
                self.chunk_bytes,
                self.gin_fifo_depth,
                self.network_step_bytes,
                self.gin_chunk_bytes,
                self.gin_context_count,
                self._logical_to_communicator,
            )
        )
        self._initialized = True
        logger.info(
            "Initialized transport rank=%s world_size=%s window_config="
            "channels:%s lsa_fifo:%s lsa_step_bytes:%s gin_fifo:%s "
            "network_step_bytes:%s "
            "gin_connections:%s gin_context_count:%s "
            "gin_reliable_doorbell:%s communicator_rank:%s node_count:%s",
            self.rank,
            self.world_size,
            self.max_channels,
            self.fifo_depth,
            self.step_bytes,
            self.gin_fifo_depth,
            self.network_step_bytes,
            self.gin_connections,
            self.gin_context_count,
            self.gin_reliable_doorbell,
            communicator_rank,
            len(set(node_ids)),
        )
        return (time.perf_counter() - start_time) * 1000.0

    def _run(self, batch: _Batch, sender: bool, sequence: int) -> dict[str, float]:
        run_start = time.perf_counter()
        if not self._logged_batch_shape:
            if os.environ.get("SHARDSTREAM_PROFILE", "0") == "1":
                routes = sorted(
                    set(zip(batch.ring_ids, batch.peers, batch.forward_peers))
                )
                print(
                    "SHARDSTREAM_RING_ROUTES "
                    + json.dumps(
                        {
                            "rank": self.rank,
                            "sender": sender,
                            "rollout_node_ids": self.rollout_node_ids,
                            "routes": routes,
                        }
                    ),
                    flush=True,
                )
            strided_spans = [
                (length, row_bytes)
                for length, row_bytes, row_stride in zip(
                    batch.lengths, batch.tensor_row_bytes, batch.tensor_row_strides
                )
                if row_bytes != row_stride
            ]
            logger.info(
                "Lowered transport plan rank=%s sender=%s spans=%s "
                "payload_bytes=%s chunk_bytes=%s expected_counts=%s "
                "strided_span_count=%s strided_payload_bytes=%s strided_row_bytes=%s",
                self.rank,
                sender,
                len(batch.tensors),
                sum(batch.lengths),
                self.chunk_bytes,
                batch.expected_counts,
                len(strided_spans),
                sum(length for length, _ in strided_spans),
                sorted({row_bytes for _, row_bytes in strided_spans}),
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
                batch.forward_peers,
                batch.ring_ids,
                bool(sender),
                int(sequence),
                *([batch.quantization] if self.fp8_block_shape else []),
            )
        )
        extension_metrics["hca_policy"] = os.environ.get(
            "SHARDSTREAM_HCA_POLICY", "balanced"
        )
        extension_metrics["fp8_blockwise"] = self.fp8_block_shape is not None
        extension_metrics["fp8_quantized_span_count"] = sum(
            bool(q[0]) for q in batch.quantization
        )
        extension_metrics["fp8_wire_bytes"] = sum(
            length for length, q in zip(batch.lengths, batch.quantization) if q[0]
        )
        extension_metrics["selected_hca"] = os.environ.get("NCCL_IB_HCA", "topology")
        extension_metrics["ring_order_strategy"] = (
            _RingOrderStrategy.ROOT_SWIZZLE.value
            if self.ring_broadcast and self.ring_swizzle
            else _RingOrderStrategy.FIXED.value
            if self.ring_broadcast
            else "disabled"
        )
        selected_hca_bandwidth = os.environ.get(
            "SHARDSTREAM_SELECTED_HCA_BANDWIDTH_GBPS"
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

    @_serialized_transport_call
    def send(
        self, parameters: dict, plan: TransferPlan, step_id: int
    ) -> dict[str, float]:
        self._check_membership_binding(parameters, plan, True)
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
                infer_instance_world_size=self.infer_instance_world_size,
                num_infer_engines=self.num_infer_engines,
                ring_broadcast=self.ring_broadcast,
                ring_swizzle=self.ring_swizzle,
                fp8_block_shape=self.fp8_block_shape,
                rollout_node_ids=self.rollout_node_ids,
            )
            build_batch_time_ms = (time.perf_counter() - build_start) * 1000.0
        metrics = self._run(batch, sender=True, sequence=_sequence_from_step(step_id))
        metrics["build_batch_time_ms"] = build_batch_time_ms
        metrics["membership_epoch"] = self.membership_epoch
        return metrics

    @_serialized_transport_call
    def recv(
        self, parameters: dict, plan: TransferPlan, step_id: int
    ) -> dict[str, float]:
        self._check_membership_binding(parameters, plan, False)
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
                infer_instance_world_size=self.infer_instance_world_size,
                num_infer_engines=self.num_infer_engines,
                ring_broadcast=self.ring_broadcast,
                ring_swizzle=self.ring_swizzle,
                fp8_block_shape=self.fp8_block_shape,
                rollout_node_ids=self.rollout_node_ids,
            )
            build_batch_time_ms = (time.perf_counter() - build_start) * 1000.0
        metrics = self._run(batch, sender=False, sequence=_sequence_from_step(step_id))
        metrics["build_batch_time_ms"] = build_batch_time_ms
        metrics["membership_epoch"] = self.membership_epoch
        return metrics

    @_serialized_transport_call
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
            infer_instance_world_size=self.infer_instance_world_size,
            num_infer_engines=self.num_infer_engines,
            ring_broadcast=self.ring_broadcast,
            ring_swizzle=self.ring_swizzle,
            fp8_block_shape=self.fp8_block_shape,
            rollout_node_ids=self.rollout_node_ids,
        )
        self._prepared_send = (parameters, plan, batch)
        self._device_prepared_binding = None

    @_serialized_transport_call
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
            infer_instance_world_size=self.infer_instance_world_size,
            num_infer_engines=self.num_infer_engines,
            ring_broadcast=self.ring_broadcast,
            ring_swizzle=self.ring_swizzle,
            fp8_block_shape=self.fp8_block_shape,
            rollout_node_ids=self.rollout_node_ids,
        )
        self._prepared_recv = (parameters, plan, batch)
        self._device_prepared_binding = None

    def _check_membership_binding(self, parameters, plan, sender):
        if self._reconfiguration_failed:
            raise TransportUnavailableError(
                "Membership rebuild failed; cannot publish weights"
            )
        if self._membership_binding is not None:
            params, bound_plan, bound_sender = self._membership_binding
            if (
                params is not parameters
                or bound_plan is not plan
                or bound_sender != sender
            ):
                raise TransportUnavailableError(
                    "Stale parameters, plan or role after membership change"
                )

    @_serialized_transport_call
    def initialize_prepared_plan(
        self,
        parameters: dict[str, torch.Tensor],
        plan: TransferPlan,
        sender: bool,
    ) -> dict[str, Any]:
        """Collectively build communicator/FIFO/cache before the next update.

        Call on every new-epoch rank, after prepare_send/prepare_recv and before
        acknowledging membership preparation. No weights are read or written,
        no quantization/forwarding kernel runs, and no weight version advances.
        """
        self._check_membership_binding(parameters, plan, sender)
        prepared = self._prepared_send if sender else self._prepared_recv
        if prepared is None or prepared[0] is not parameters or prepared[1] is not plan:
            raise TransportUnavailableError(
                "Prepare the exact host plan before initializing its device cache"
            )
        started = time.perf_counter()
        try:
            init_time_ms = self._ensure_initialized()
            batch = prepared[2]
            metrics = dict(
                self._extension.prepare(
                    self._handle,
                    batch.tensors,
                    batch.lengths,
                    batch.tensor_offsets,
                    batch.tensor_row_bytes,
                    batch.tensor_row_strides,
                    batch.peers,
                    batch.ordinals,
                    batch.expected_counts,
                    batch.forward_peers,
                    batch.ring_ids,
                    bool(sender),
                    *([batch.quantization] if self.fp8_block_shape else []),
                )
            )
            if not metrics.get("device_plan_ready") or metrics.get(
                "kernel_launched", True
            ):
                raise TransportUnavailableError(
                    "Device prepare did not establish a cache without launching a kernel"
                )
            self._device_prepared_binding = (parameters, plan, bool(sender))
            metrics.update(
                transport_init_time_ms=init_time_ms,
                membership_epoch=self.membership_epoch,
                preparation_total_time_ms=(time.perf_counter() - started) * 1000,
            )
            return metrics
        except Exception:
            self._reconfiguration_failed = True
            self.close()
            raise

    def is_device_plan_ready(
        self, parameters: dict, plan: TransferPlan, sender: bool
    ) -> bool:
        """Readiness for these exact tensors, role and plan in this epoch."""
        with self._operation_lock:
            binding = self._device_prepared_binding
            return bool(
                not self._reconfiguration_failed
                and self._initialized
                and binding is not None
                and binding[0] is parameters
                and binding[1] is plan
                and binding[2] == sender
            )

    @_serialized_transport_call
    def reconfigure(
        self,
        group: Any,
        membership: RolloutMembership,
        participant: Participant,
        parameters: dict[str, torch.Tensor],
        plan: TransferPlan,
        rollout_node_ids: tuple[Any, ...] | None = None,
    ) -> None:
        """Bind an expanded epoch without replacing model tensors or the kernel.

        All old participants must first finish their previous publication. The
        application creates a new process group containing old and new ranks,
        calls this method on surviving ranks, and creates fresh transports on
        new ranks. Process groups are borrowed and remain caller-owned.
        """
        rank = membership.rank(participant)
        if rollout_node_ids is not None and len(rollout_node_ids) != len(
            membership.engine_ids
        ):
            raise TransportUnavailableError(
                "Explicit rollout topology must cover every engine"
            )
        sender = participant[0] == "training"
        training_size = (
            self.world_size - self.infer_instance_world_size * self.num_infer_engines
        )
        if membership.epoch != self.membership_epoch + 1:
            raise TransportUnavailableError(
                "Membership epochs must advance exactly once"
            )
        if (
            membership.training_world_size != training_size
            or membership.inference_tp_size != self.infer_instance_world_size
        ):
            raise TransportUnavailableError(
                "Changing training size or inference TP is not supported"
            )
        if len(membership.engine_ids) <= self.num_infer_engines:
            raise TransportUnavailableError(
                "Dynamic reconfiguration only supports adding engines"
            )
        # Custom transfer groups are independent of the training default group;
        # dist.get_rank(group) may map the default rank instead of this PG rank.
        if group.rank() != rank or group.size() != membership.world_size:
            raise TransportUnavailableError(
                "New process group does not match the membership"
            )
        for peer, operations in plan.operations.items():
            if not 0 <= peer < membership.world_size:
                raise TransportUnavailableError(
                    "Plan has a peer outside the new membership"
                )
            for op in operations:
                if (op.send_rank if sender else op.recv_rank) != rank:
                    raise TransportUnavailableError(
                        "Plan uses the previous epoch's local rank"
                    )
                if (
                    (op.recv_rank if sender else op.send_rank) != peer
                    or not membership.inference_world_size
                    <= op.send_rank
                    < membership.world_size
                    or not 0 <= op.recv_rank < membership.inference_world_size
                ):
                    raise TransportUnavailableError(
                        "Plan peers do not match the new training/rollout partition"
                    )
        # Release the old FIFO and registrations before preparing the new plan.
        self._reconfiguration_failed = True
        try:
            self.close()
            self.group = group
            self.rank = rank
            self.world_size = membership.world_size
            self.infer_instance_world_size = membership.inference_tp_size
            self.num_infer_engines = len(membership.engine_ids)
            self.rollout_node_ids = (
                tuple(rollout_node_ids) if rollout_node_ids is not None else None
            )
            self._logical_to_communicator = list(range(self.world_size))
            self._logged_batch_shape = False
            if sender:
                self.prepare_send(parameters, plan, allow_staging=False)
            else:
                self.prepare_recv(parameters, plan, allow_staging=False)
            self.membership_epoch = membership.epoch
            self._membership_binding = (parameters, plan, sender)
            self._reconfiguration_failed = False
        except Exception:
            self.close()
            raise

    @_serialized_transport_call
    def close(self) -> None:
        try:
            if self._handle is not None and self._extension is not None:
                self._extension.destroy(self._handle)
        finally:
            self._handle = None
            self._initialized = False
            self._prepared_send = None
            self._prepared_recv = None
            self._device_prepared_binding = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

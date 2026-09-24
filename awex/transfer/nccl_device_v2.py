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
import threading
import time
from dataclasses import dataclass
from typing import Any


class NCCLDeviceV2UnavailableError(RuntimeError):
    """Raised when the isolated NCCL Device v2 path cannot be initialized."""


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


_preload_configured_nccl()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from awex import logging  # noqa: E402
from awex.transfer.tensor_layout import (  # noqa: E402
    BlockwiseFp8Layout,
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
    tensor_lengths: list[int]
    tensor_offsets: list[int]
    tensor_row_bytes: list[int]
    tensor_row_strides: list[int]
    wire_dtypes: list[int]
    wire_element_bytes: list[int]
    quant_scale_tensors: list[torch.Tensor]
    quant_modes: list[int]
    quant_rows: list[int]
    quant_cols: list[int]
    quant_row_offsets: list[int]
    quant_col_offsets: list[int]
    quant_scale_row_strides: list[int]
    quant_block_rows: list[int]
    quant_block_cols: list[int]
    quant_group_ids: list[int]
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


def _blockwise_quant_group_ids(
    operations: list[CommunicationOperation],
) -> list[int]:
    """Pair logical destination weights with their scale tensors.

    A physical source layout can split one weight operation into several task
    descriptors. Keeping the group on the operation makes that fragmentation
    explicit instead of asking the CUDA lowering code to infer matrix bounds
    from descriptor adjacency.
    """

    suffix = "_scale_inv"
    weights: dict[str, list[int]] = {}
    scales: dict[str, list[int]] = {}
    for index, op in enumerate(operations):
        name = op.recv_shard_meta.name
        if name.endswith(suffix):
            scales.setdefault(name[: -len(suffix)], []).append(index)
        else:
            weights.setdefault(name, []).append(index)

    group_ids = [-1] * len(operations)
    group_id = 0
    for name in sorted(weights.keys() & scales.keys()):
        for weight_index, scale_index in zip(weights[name], scales[name]):
            group_ids[weight_index] = group_id
            group_ids[scale_index] = group_id
            group_id += 1
    return group_ids


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


_V2_DTYPE_CODES = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}
for _dtype_name, _dtype_code in (
    ("float8_e4m3fn", 4),
    ("float8_e5m2", 5),
):
    _dtype = getattr(torch, _dtype_name, None)
    if _dtype is not None:
        _V2_DTYPE_CODES[_dtype] = _dtype_code


def _normalize_dtype(dtype: Any, description: str) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""), None)
    if not isinstance(dtype, torch.dtype):
        raise NCCLDeviceV2UnavailableError(
            f"nccl_device_v2 has an invalid dtype for {description}: {dtype!r}"
        )
    return dtype


def _dtype_descriptor(dtype: Any, description: str) -> tuple[torch.dtype, int, int]:
    dtype = _normalize_dtype(dtype, description)
    return dtype, _V2_DTYPE_CODES.get(dtype, 0), int(dtype.itemsize)


def _wire_dtype(
    operation: CommunicationOperation, fallback: torch.dtype
) -> torch.dtype:
    dtype = getattr(operation.recv_shard_meta, "dtype", None)
    if dtype is None:
        return fallback
    return _normalize_dtype(dtype, operation.recv_shard_meta.name)


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


def _quant_matrix_copy_layout(tensor: torch.Tensor, name: str) -> tuple[int, int]:
    if tensor.dim() != 2 or int(tensor.stride(1)) != 1:
        raise NCCLDeviceV2UnavailableError(
            "v2 block-wise FP8 source must be a row-major 2-D view: "
            f"parameter={name}, shape={tuple(tensor.shape)}, "
            f"stride={tuple(tensor.stride())}"
        )
    row_bytes = int(tensor.shape[1]) * int(tensor.element_size())
    row_stride = int(tensor.stride(0)) * int(tensor.element_size())
    if row_stride < row_bytes:
        raise NCCLDeviceV2UnavailableError(
            f"v2 block-wise FP8 row stride is invalid: parameter={name}"
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
    tensor_lengths: list[int],
    wire_dtypes: list[int],
    wire_element_bytes: list[int],
    quant_scale_tensors: list[torch.Tensor],
    quant_modes: list[int],
    quant_rows: list[int],
    quant_cols: list[int],
    quant_row_offsets: list[int],
    quant_col_offsets: list[int],
    quant_scale_row_strides: list[int],
    quant_block_rows: list[int],
    quant_block_cols: list[int],
    quant_group_ids: list[int],
    peers: list[int],
    ordinals: list[int],
    region_indices: list[int],
    tensor: torch.Tensor,
    wire_dtype: torch.dtype,
    tensor_offset: int,
    nbytes: int,
    row_bytes: int,
    row_stride: int,
    peer: int,
    peer_offset: int,
    ordinal: int,
    region_index: int,
    quant_scale_tensor: torch.Tensor | None = None,
    quant_rows_value: int = 0,
    quant_cols_value: int = 0,
    quant_row_offset: int = 0,
    quant_col_offset: int = 0,
    quant_scale_row_stride: int = 0,
    quant_block_shape: tuple[int, int] = (0, 0),
    quant_group_id: int = -1,
) -> int:
    # Preserve one descriptor per physical TransferPlan span. C++ concatenates
    # these descriptors into a virtual peer stream before selecting channels
    # and lowering transport chunks, so chunk boundaries may cross tensors.
    tensor_dtype, tensor_dtype_code, tensor_element_bytes = _dtype_descriptor(
        tensor.dtype, "source tensor"
    )
    wire_dtype, wire_dtype_code, wire_item_bytes = _dtype_descriptor(
        wire_dtype, "wire format"
    )
    if tensor_dtype != wire_dtype and (tensor_dtype_code == 0 or wire_dtype_code == 0):
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 streaming cast does not support "
            f"{tensor_dtype} -> {wire_dtype}"
        )
    if nbytes % wire_item_bytes:
        raise NCCLDeviceV2UnavailableError(
            "nccl_device_v2 wire range must contain complete elements"
        )
    if quant_scale_tensor is not None:
        descriptor = {
            "shape": tuple(int(dim) for dim in tensor.shape),
            "tensor_offset": int(tensor_offset),
            "nbytes": int(nbytes),
            "row_bytes": int(row_bytes),
            "row_stride": int(row_stride),
            "quant_rows": int(quant_rows_value),
            "quant_cols": int(quant_cols_value),
            "row_offset": int(quant_row_offset),
            "col_offset": int(quant_col_offset),
            "block_shape": tuple(int(dim) for dim in quant_block_shape),
        }
        valid = (
            tensor.dim() == 2
            and quant_rows_value > 0
            and quant_cols_value > 0
            and tensor_offset == 0
            and row_bytes == quant_cols_value * tensor_element_bytes
            and nbytes == quant_rows_value * quant_cols_value * wire_item_bytes
            and tuple(quant_block_shape) == (128, 128)
            and quant_row_offset % 128 == 0
            and quant_col_offset % 128 == 0
            and quant_scale_row_stride > 0
        )
        if not valid:
            raise NCCLDeviceV2UnavailableError(
                "Invalid nccl_device_v2 block-wise FP8 descriptor: "
                f"{descriptor}"
            )

    tensors.append(tensor)
    tensor_offsets.append(tensor_offset)
    tensor_row_bytes.append(row_bytes)
    tensor_row_strides.append(row_stride)
    offsets.append(peer_offset)
    lengths.append(nbytes)
    tensor_lengths.append(nbytes // wire_item_bytes * tensor_element_bytes)
    wire_dtypes.append(wire_dtype_code)
    wire_element_bytes.append(wire_item_bytes)
    quant_scale_tensors.append(
        tensor if quant_scale_tensor is None else quant_scale_tensor
    )
    quant_modes.append(0 if quant_scale_tensor is None else 1)
    quant_rows.append(int(quant_rows_value))
    quant_cols.append(int(quant_cols_value))
    quant_row_offsets.append(int(quant_row_offset))
    quant_col_offsets.append(int(quant_col_offset))
    quant_scale_row_strides.append(int(quant_scale_row_stride))
    quant_block_rows.append(int(quant_block_shape[0]))
    quant_block_cols.append(int(quant_block_shape[1]))
    quant_group_ids.append(int(quant_group_id))
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
    tensor_lengths: list[int] = []
    tensor_offsets: list[int] = []
    tensor_row_bytes: list[int] = []
    tensor_row_strides: list[int] = []
    wire_dtypes: list[int] = []
    wire_element_bytes: list[int] = []
    quant_scale_tensors: list[torch.Tensor] = []
    quant_modes: list[int] = []
    quant_rows: list[int] = []
    quant_cols: list[int] = []
    quant_row_offsets: list[int] = []
    quant_col_offsets: list[int] = []
    quant_scale_row_strides: list[int] = []
    quant_block_rows: list[int] = []
    quant_block_cols: list[int] = []
    quant_group_ids: list[int] = []
    peers: list[int] = []
    ordinals: list[int] = []
    region_indices: list[int] = []
    region_bytes = [0] * world_size
    expected_counts = [0] * world_size
    context = {}
    for peer, operations in _operation_groups(plan, rank, world_size):
        peer_offset = 0
        ordinal = 0
        operation_group_ids = _blockwise_quant_group_ids(operations)
        for operation_index, op in enumerate(operations):
            parameter = parameters[op.send_shard_meta.name]
            blockwise_state = None
            source_fragments = None
            if isinstance(parameter, BlockwiseFp8Layout):
                if parameter.kind == "weight":
                    blockwise_state = parameter.state
                    source_layout = blockwise_state.source
                    if (
                        op.send_tensor_span_numels
                        and isinstance(source_layout, StaticTensorLayout)
                        and source_layout.span_numels != op.send_tensor_span_numels
                    ):
                        raise NCCLDeviceV2UnavailableError(
                            "Compiled block-wise source layout does not match "
                            f"the transfer plan for {op.send_shard_meta.name}"
                        )
                    source_fragments = parameter.source_fragments(op.train_slices)
                    fragments = [fragment.tensor for fragment in source_fragments]
                else:
                    blockwise_state = parameter.state
                    tensor = parameter.scale_view(op.train_slices)
                    fragments = [tensor]
            elif isinstance(parameter, StaticTensorLayout):
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
            for fragment_number, fragment in enumerate(fragments):
                _ensure_cuda_tensor(fragment, op.send_shard_meta.name)
                target_dtype = _wire_dtype(op, fragment.dtype)
                if blockwise_state is not None and source_fragments is not None:
                    if target_dtype != torch.float8_e4m3fn:
                        raise NCCLDeviceV2UnavailableError(
                            "Block-wise FP8 weight requires an E4M3 receiver: "
                            f"{op.recv_shard_meta.name} uses {target_dtype}"
                        )
                _, _, wire_item_bytes = _dtype_descriptor(
                    target_dtype, op.recv_shard_meta.name
                )
                length = int(fragment.numel()) * wire_item_bytes
                row_bytes, row_stride = _tensor_copy_layout(
                    fragment, op.send_shard_meta.name
                )
                quant_fragment = None
                if source_fragments is not None:
                    quant_fragment = source_fragments[fragment_number]
                    row_bytes, row_stride = _quant_matrix_copy_layout(
                        fragment, op.send_shard_meta.name
                    )
                ordinal = _append_tensor_range(
                    tensors=tensors,
                    tensor_offsets=tensor_offsets,
                    tensor_row_bytes=tensor_row_bytes,
                    tensor_row_strides=tensor_row_strides,
                    offsets=offsets,
                    lengths=lengths,
                    tensor_lengths=tensor_lengths,
                    wire_dtypes=wire_dtypes,
                    wire_element_bytes=wire_element_bytes,
                    quant_scale_tensors=quant_scale_tensors,
                    quant_modes=quant_modes,
                    quant_rows=quant_rows,
                    quant_cols=quant_cols,
                    quant_row_offsets=quant_row_offsets,
                    quant_col_offsets=quant_col_offsets,
                    quant_scale_row_strides=quant_scale_row_strides,
                    quant_block_rows=quant_block_rows,
                    quant_block_cols=quant_block_cols,
                    quant_group_ids=quant_group_ids,
                    peers=peers,
                    ordinals=ordinals,
                    region_indices=region_indices,
                    tensor=fragment,
                    wire_dtype=target_dtype,
                    tensor_offset=0,
                    nbytes=length,
                    row_bytes=row_bytes,
                    row_stride=row_stride,
                    peer=peer,
                    peer_offset=peer_offset,
                    ordinal=ordinal,
                    region_index=rank,
                    quant_scale_tensor=(
                        blockwise_state.scale if quant_fragment is not None else None
                    ),
                    quant_rows_value=(
                        int(fragment.shape[0]) if quant_fragment is not None else 0
                    ),
                    quant_cols_value=(
                        int(fragment.shape[1]) if quant_fragment is not None else 0
                    ),
                    quant_row_offset=(
                        quant_fragment.row_offset if quant_fragment is not None else 0
                    ),
                    quant_col_offset=(
                        quant_fragment.col_offset if quant_fragment is not None else 0
                    ),
                    quant_scale_row_stride=(
                        int(blockwise_state.scale.stride(0))
                        if quant_fragment is not None
                        else 0
                    ),
                    quant_block_shape=(
                        blockwise_state.block_shape
                        if quant_fragment is not None
                        else (0, 0)
                    ),
                    quant_group_id=(
                        operation_group_ids[operation_index]
                        if isinstance(parameter, BlockwiseFp8Layout)
                        else -1
                    ),
                )
                peer_offset += length
        expected_counts[peer] = ordinal
        region_bytes[rank] = max(region_bytes[rank], peer_offset)
    return _V2Batch(
        tensors=tensors,
        offsets=offsets,
        lengths=lengths,
        tensor_lengths=tensor_lengths,
        tensor_offsets=tensor_offsets,
        tensor_row_bytes=tensor_row_bytes,
        tensor_row_strides=tensor_row_strides,
        wire_dtypes=wire_dtypes,
        wire_element_bytes=wire_element_bytes,
        quant_scale_tensors=quant_scale_tensors,
        quant_modes=quant_modes,
        quant_rows=quant_rows,
        quant_cols=quant_cols,
        quant_row_offsets=quant_row_offsets,
        quant_col_offsets=quant_col_offsets,
        quant_scale_row_strides=quant_scale_row_strides,
        quant_block_rows=quant_block_rows,
        quant_block_cols=quant_block_cols,
        quant_group_ids=quant_group_ids,
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
    tensor_lengths: list[int] = []
    tensor_offsets: list[int] = []
    tensor_row_bytes: list[int] = []
    tensor_row_strides: list[int] = []
    wire_dtypes: list[int] = []
    wire_element_bytes: list[int] = []
    quant_scale_tensors: list[torch.Tensor] = []
    quant_modes: list[int] = []
    quant_rows: list[int] = []
    quant_cols: list[int] = []
    quant_row_offsets: list[int] = []
    quant_col_offsets: list[int] = []
    quant_scale_row_strides: list[int] = []
    quant_block_rows: list[int] = []
    quant_block_cols: list[int] = []
    quant_group_ids: list[int] = []
    peers: list[int] = []
    ordinals: list[int] = []
    region_indices: list[int] = []
    region_bytes = [0] * world_size
    expected_counts = [0] * world_size
    copybacks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for peer, operations in _operation_groups(plan, rank, world_size):
        peer_offset = 0
        ordinal = 0
        operation_group_ids = _blockwise_quant_group_ids(operations)
        for operation_index, op in enumerate(operations):
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
            target_wire_dtype = _wire_dtype(op, target.dtype)
            if target.dtype != target_wire_dtype:
                raise NCCLDeviceV2UnavailableError(
                    "nccl_device_v2 receive tensor dtype does not match the wire "
                    f"format for {op.recv_shard_meta.name}: "
                    f"tensor={target.dtype}, wire={target_wire_dtype}"
                )
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
                    tensor_lengths=tensor_lengths,
                    wire_dtypes=wire_dtypes,
                    wire_element_bytes=wire_element_bytes,
                    quant_scale_tensors=quant_scale_tensors,
                    quant_modes=quant_modes,
                    quant_rows=quant_rows,
                    quant_cols=quant_cols,
                    quant_row_offsets=quant_row_offsets,
                    quant_col_offsets=quant_col_offsets,
                    quant_scale_row_strides=quant_scale_row_strides,
                    quant_block_rows=quant_block_rows,
                    quant_block_cols=quant_block_cols,
                    quant_group_ids=quant_group_ids,
                    peers=peers,
                    ordinals=ordinals,
                    region_indices=region_indices,
                    tensor=target,
                    wire_dtype=target_wire_dtype,
                    tensor_offset=target_offset,
                    nbytes=length,
                    row_bytes=row_bytes,
                    row_stride=row_stride,
                    peer=peer,
                    peer_offset=peer_offset,
                    ordinal=ordinal,
                    region_index=peer,
                    quant_group_id=operation_group_ids[operation_index],
                )
                target_offset += length
                peer_offset += length
        expected_counts[peer] = ordinal
        region_bytes[peer] = max(region_bytes[peer], peer_offset)
    return _V2Batch(
        tensors=tensors,
        offsets=offsets,
        lengths=lengths,
        tensor_lengths=tensor_lengths,
        tensor_offsets=tensor_offsets,
        tensor_row_bytes=tensor_row_bytes,
        tensor_row_strides=tensor_row_strides,
        wire_dtypes=wire_dtypes,
        wire_element_bytes=wire_element_bytes,
        quant_scale_tensors=quant_scale_tensors,
        quant_modes=quant_modes,
        quant_rows=quant_rows,
        quant_cols=quant_cols,
        quant_row_offsets=quant_row_offsets,
        quant_col_offsets=quant_col_offsets,
        quant_scale_row_strides=quant_scale_row_strides,
        quant_block_rows=quant_block_rows,
        quant_block_cols=quant_block_cols,
        quant_group_ids=quant_group_ids,
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
                extra_cuda_cflags=["-O3"],
                extra_ldflags=[
                    *(f"-L{path}" for path in library_paths),
                    *(
                        f"-Wl,--disable-new-dtags,-rpath,{path}"
                        for path in library_paths
                    ),
                    "-lnccl",
                    "-lcuda",
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
        self.fifo_depth = 8
        self.step_bytes = _env_int(
            "AWEX_NCCL_DEVICE_V2_STEP_BYTES", 512 * 1024, minimum=1
        )
        if self.chunk_bytes and self.chunk_bytes < self.step_bytes:
            raise NCCLDeviceV2UnavailableError(
                "nccl_device_v2 chunk_bytes must be at least step_bytes"
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
            "fifo_depth=%s step_bytes=%s",
            self.rank,
            self.chunk_bytes,
            self.max_channels,
            self.fifo_depth,
            self.step_bytes,
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
                self.chunk_bytes,
            )
        )
        self._initialized = True
        logger.info(
            "Initialized nccl_device_v2 rank=%s world_size=%s window_config="
            "channels:%s fifo:%s step_bytes:%s",
            self.rank,
            self.world_size,
            self.max_channels,
            self.fifo_depth,
            self.step_bytes,
        )
        return (time.perf_counter() - start_time) * 1000.0

    def _run(self, batch: _V2Batch, sender: bool, sequence: int) -> dict[str, float]:
        run_start = time.perf_counter()
        if not self._logged_batch_shape:
            logger.info(
                "Lowered nccl_device_v2 plan rank=%s sender=%s spans=%s "
                "tensor_bytes=%s wire_bytes=%s chunk_bytes=%s expected_counts=%s",
                self.rank,
                sender,
                len(batch.tensors),
                sum(batch.tensor_lengths),
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
                batch.wire_dtypes,
                batch.wire_element_bytes,
                batch.quant_scale_tensors,
                batch.quant_modes,
                batch.quant_rows,
                batch.quant_cols,
                batch.quant_row_offsets,
                batch.quant_col_offsets,
                batch.quant_scale_row_strides,
                batch.quant_block_rows,
                batch.quant_block_cols,
                batch.quant_group_ids,
                batch.peers,
                batch.ordinals,
                batch.expected_counts,
                bool(sender),
                int(sequence),
            )
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
                "tensor_bytes": float(sum(batch.tensor_lengths)),
                "wire_compression_ratio": (
                    float(sum(batch.tensor_lengths)) / float(sum(batch.lengths))
                    if batch.lengths and sum(batch.lengths)
                    else 1.0
                ),
                "transport_init_time_ms": init_time_ms,
                "python_copyback_time_ms": copyback_time_ms,
                "transport_total_time_ms": (time.perf_counter() - run_start) * 1000.0,
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

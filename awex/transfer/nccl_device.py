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

"""Experimental NCCL Device API transport.

The transport deliberately keeps the existing TransferPlan as the source of
truth.  Python only binds tensor pointers to a compact task list; the data
movement itself is performed by ``awex_nccl_device_ext`` in a CUDA kernel.
The extension is loaded lazily so importing Awex or using the legacy NCCL
backend does not require a CUDA toolkit or a recent NCCL installation.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from awex import logging
from awex.transfer.tensor_layout import (
    StaticTensorLayout,
    slice_layout_fragments,
)
from awex.transfer.transfer_plan import (
    CommunicationOperation,
    TransferPlan,
    build_transfer_chunks,
    slice_tensor,
)
from awex.util import device as device_util

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
_CHUNK_ALIGNMENT = 16


class NCCLDeviceUnavailableError(RuntimeError):
    """Raised when the optional NCCL Device API path cannot be initialized."""


@dataclass
class _DeviceBatch:
    tensors: List[torch.Tensor]
    offsets: List[int]
    lengths: List[int]
    peers: List[int]
    ordinals: List[int]
    region_indices: List[int]
    region_bytes: List[int]
    expected_counts: List[int]
    copybacks: List[Tuple[torch.Tensor, torch.Tensor]]


_extension_lock = threading.Lock()
_extension: Optional[Any] = None


def _candidate_include_paths() -> List[str]:
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
    result = []
    for path in values:
        if path not in result:
            result.append(path)
    return result


def _candidate_library_paths() -> List[str]:
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
    result = []
    for path in values:
        if path not in result:
            result.append(path)
    return result


def _load_extension() -> Any:
    """Build/load the optional CUDA extension on first use."""

    global _extension
    if _extension is not None:
        return _extension
    with _extension_lock:
        if _extension is not None:
            return _extension
        if device_util.get_device_type() != "cuda" or not torch.cuda.is_available():
            raise NCCLDeviceUnavailableError(
                "nccl_device requires a CUDA runtime; use comm_backend=nccl "
                "on non-CUDA devices."
            )

        module_name = os.environ.get("AWEX_NCCL_DEVICE_EXTENSION", "")
        if module_name:
            try:
                _extension = __import__(module_name, fromlist=["*"])
                return _extension
            except Exception as exc:
                raise NCCLDeviceUnavailableError(
                    f"Failed to import AWEX_NCCL_DEVICE_EXTENSION={module_name!r}: {exc}"
                ) from exc

        include_paths = _candidate_include_paths()
        header_candidates = [
            candidate
            for path in include_paths
            for candidate in (
                Path(path) / "nccl_device.h",
                Path(path) / "nccl_device" / "core.h",
            )
        ]
        if not any(path.exists() for path in header_candidates):
            raise NCCLDeviceUnavailableError(
                "NCCL Device API headers were not found. Set AWEX_NCCL_INCLUDE to "
                "the NCCL include directory containing nccl_device.h."
            )

        try:
            from torch.utils.cpp_extension import load
        except Exception as exc:
            raise NCCLDeviceUnavailableError(
                "torch.utils.cpp_extension is required to build nccl_device."
            ) from exc

        source = Path(__file__).with_name("nccl_device_ext.cu")
        if not source.exists():
            raise NCCLDeviceUnavailableError(f"Missing CUDA source: {source}")

        extra_ldflags = [
            *(f"-L{path}" for path in _candidate_library_paths()),
            "-lnccl",
        ]
        try:
            _extension = load(
                name="awex_nccl_device_ext_v4",
                sources=[str(source)],
                extra_include_paths=include_paths,
                extra_cuda_cflags=["-O3"],
                extra_ldflags=extra_ldflags,
                with_cuda=True,
                verbose=os.environ.get("AWEX_NCCL_DEVICE_VERBOSE_BUILD", "0") == "1",
            )
        except Exception as exc:
            raise NCCLDeviceUnavailableError(
                "Failed to build the NCCL Device API extension. Check CUDA/NCCL "
                "versions and AWEX_NCCL_INCLUDE/AWEX_NCCL_LIB."
            ) from exc
        return _extension


def _ensure_cuda_tensor(tensor: torch.Tensor, description: str) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise NCCLDeviceUnavailableError(
            f"nccl_device only supports CUDA tensors ({description})."
        )
    if not tensor.is_contiguous():
        raise NCCLDeviceUnavailableError(
            f"nccl_device received a non-contiguous tensor ({description}); "
            "the operation must be staged before submission."
        )


def _operation_groups(
    plan: TransferPlan, rank: int, world_size: int
) -> List[Tuple[int, List[CommunicationOperation]]]:
    peers = sorted(plan.operations)
    if any(peer < 0 or peer >= world_size for peer in peers):
        raise NCCLDeviceUnavailableError(
            f"nccl_device plan contains an invalid peer for world_size={world_size}: {peers}"
        )
    if rank in peers:
        raise NCCLDeviceUnavailableError(
            "nccl_device does not support self-transfer operations"
        )
    return [(peer, list(plan.operations[peer])) for peer in peers]


def _resolve_chunk_bytes(chunk_bytes: Optional[int]) -> int:
    if chunk_bytes is None:
        configured = os.environ.get("AWEX_NCCL_DEVICE_CHUNK_BYTES")
        try:
            chunk_bytes = (
                _DEFAULT_CHUNK_BYTES if configured is None else int(configured)
            )
        except ValueError as exc:
            raise NCCLDeviceUnavailableError(
                "AWEX_NCCL_DEVICE_CHUNK_BYTES must be an integer"
            ) from exc
    chunk_bytes = int(chunk_bytes)
    if chunk_bytes < 0:
        raise NCCLDeviceUnavailableError("nccl_device chunk_bytes must be non-negative")
    if chunk_bytes and chunk_bytes % _CHUNK_ALIGNMENT != 0:
        raise NCCLDeviceUnavailableError(
            f"nccl_device chunk_bytes must be a multiple of {_CHUNK_ALIGNMENT}"
        )
    return chunk_bytes


def _split_contiguous_tensor(
    tensor: torch.Tensor, chunk_bytes: int
) -> List[Tuple[torch.Tensor, int, int]]:
    """Return tensor views paired with their byte offset and length."""

    element_size = int(tensor.element_size())
    total_bytes = int(tensor.numel()) * element_size
    flat = tensor.reshape(-1)
    result = []
    for chunk in build_transfer_chunks(total_bytes, chunk_bytes):
        if chunk.byte_offset % element_size or chunk.nbytes % element_size:
            raise NCCLDeviceUnavailableError(
                "nccl_device chunk boundaries must align to tensor elements"
            )
        result.append(
            (
                flat.narrow(
                    0,
                    chunk.byte_offset // element_size,
                    chunk.nbytes // element_size,
                ),
                chunk.byte_offset,
                chunk.nbytes,
            )
        )
    return result


def _sequence_from_step(step_id: int) -> int:
    sequence = int(step_id) + 2
    if sequence <= 0:
        raise NCCLDeviceUnavailableError(
            f"nccl_device step_id must be at least -1, got {step_id}."
        )
    return sequence


def _build_send_batch(
    parameters: dict,
    plan: TransferPlan,
    rank: int,
    world_size: int,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    allow_staging: bool = True,
) -> _DeviceBatch:
    chunk_bytes = _resolve_chunk_bytes(chunk_bytes)
    tensors: List[torch.Tensor] = []
    offsets: List[int] = []
    lengths: List[int] = []
    peers: List[int] = []
    ordinals: List[int] = []
    region_indices: List[int] = []
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
                    raise NCCLDeviceUnavailableError(
                        "Compiled source layout does not match transfer plan for "
                        f"{op.send_shard_meta.name}: layout={parameter.span_numels} "
                        f"plan={op.send_tensor_span_numels}"
                    )
                fragments = parameter.slice(op.train_slices)
            else:
                if allow_staging:
                    tensor = slice_tensor(parameter, op, True, slice_context=context)
                else:
                    tensor = parameter[op.train_slices]
                    if not tensor.is_contiguous():
                        raise NCCLDeviceUnavailableError(
                            "Compiled nccl_device plan would require a sender "
                            "staging copy: "
                            f"parameter={op.send_shard_meta.name}, "
                            f"shape={tuple(parameter.shape)}, "
                            f"slices={op.train_slices}. Adjust train/inference "
                            "sharding alignment or lower this slice to direct spans."
                        )
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                fragments = [tensor]

            for fragment in fragments:
                _ensure_cuda_tensor(fragment, op.send_shard_meta.name)
                length = int(fragment.numel()) * int(fragment.element_size())
                for chunk, chunk_offset, chunk_length in _split_contiguous_tensor(
                    fragment, chunk_bytes
                ):
                    tensors.append(chunk)
                    offsets.append(peer_offset + chunk_offset)
                    lengths.append(chunk_length)
                    peers.append(peer)
                    ordinals.append(ordinal)
                    region_indices.append(rank)
                    ordinal += 1
                peer_offset += length
        expected_counts[peer] = ordinal
        region_bytes[rank] = max(region_bytes[rank], peer_offset)
    return _DeviceBatch(
        tensors,
        offsets,
        lengths,
        peers,
        ordinals,
        region_indices,
        region_bytes,
        expected_counts,
        [],
    )


def _build_recv_batch(
    parameters: dict,
    plan: TransferPlan,
    rank: int,
    world_size: int,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    allow_staging: bool = True,
) -> _DeviceBatch:
    chunk_bytes = _resolve_chunk_bytes(chunk_bytes)
    tensors: List[torch.Tensor] = []
    offsets: List[int] = []
    lengths: List[int] = []
    peers: List[int] = []
    ordinals: List[int] = []
    region_indices: List[int] = []
    region_bytes = [0] * world_size
    expected_counts = [0] * world_size
    copybacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for peer, operations in _operation_groups(plan, rank, world_size):
        peer_offset = 0
        ordinal = 0
        for op in operations:
            parameter = parameters[op.recv_shard_meta.name]
            view = parameter[op.inf_slices]
            target = view
            if not view.is_contiguous():
                if not allow_staging:
                    raise NCCLDeviceUnavailableError(
                        "Compiled nccl_device plan would require a receiver staging "
                        "copy: "
                        f"parameter={op.recv_shard_meta.name}, "
                        f"shape={tuple(parameter.shape)}, "
                        f"slices={op.inf_slices}. Adjust train/inference sharding "
                        "alignment or lower this slice to direct spans."
                    )
                target = torch.empty_like(view, memory_format=torch.contiguous_format)
                copybacks.append((view, target))
            _ensure_cuda_tensor(target, op.recv_shard_meta.name)
            targets = [target]
            if op.send_tensor_span_numels:
                layout_fragments = slice_layout_fragments(
                    op.send_shard_meta.shape,
                    op.train_slices,
                    op.send_tensor_span_numels,
                )
                fragment_numels = [fragment[2] for fragment in layout_fragments]
                if sum(fragment_numels) != target.numel():
                    raise NCCLDeviceUnavailableError(
                        "Source layout slice does not match receive tensor for "
                        f"{op.recv_shard_meta.name}: source={sum(fragment_numels)} "
                        f"target={target.numel()}"
                    )
                flat_target = target.reshape(-1)
                target_offset = 0
                targets = []
                for fragment_numel in fragment_numels:
                    targets.append(flat_target.narrow(0, target_offset, fragment_numel))
                    target_offset += fragment_numel

            for fragment in targets:
                length = int(fragment.numel()) * int(fragment.element_size())
                for chunk, chunk_offset, chunk_length in _split_contiguous_tensor(
                    fragment, chunk_bytes
                ):
                    tensors.append(chunk)
                    offsets.append(peer_offset + chunk_offset)
                    lengths.append(chunk_length)
                    peers.append(peer)
                    ordinals.append(ordinal)
                    region_indices.append(peer)
                    ordinal += 1
                peer_offset += length
        expected_counts[peer] = ordinal
        region_bytes[peer] = max(region_bytes[peer], peer_offset)
    return _DeviceBatch(
        tensors,
        offsets,
        lengths,
        peers,
        ordinals,
        region_indices,
        region_bytes,
        expected_counts,
        copybacks,
    )


class NCCLDeviceTransport:
    """Multi-peer LSA transport backed by a custom CUDA task kernel."""

    def __init__(
        self,
        group: Any,
        rank: int,
        world_size: int,
        timeout_ms: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
    ):
        if world_size < 2 or world_size > 256:
            raise NCCLDeviceUnavailableError(
                f"nccl_device supports world_size in [2, 256], got {world_size}."
            )
        self.group = group
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.timeout_ms = int(
            timeout_ms or os.environ.get("AWEX_NCCL_DEVICE_TIMEOUT_MS", "120000")
        )
        self.chunk_bytes = _resolve_chunk_bytes(chunk_bytes)
        self._extension = None
        self._handle: Optional[int] = None
        self._initialized = False
        self._region_sizes: Optional[List[int]] = None
        self._logged_batch_shape = False
        self._prepared_send = None
        self._prepared_recv = None
        logger.info(
            "Configured nccl_device transport rank=%s chunk_bytes=%s",
            self.rank,
            self.chunk_bytes,
        )

    def _ensure_initialized(self, total_bytes: int) -> float:
        if self._initialized:
            return 0.0
        start_time = time.perf_counter()
        self._extension = _load_extension()
        device = torch.device(device_util.get_torch_device())
        max_bytes = int(total_bytes)

        unique_id_size = int(self._extension.unique_id_size())
        unique_id_tensor = torch.empty(unique_id_size, dtype=torch.uint8, device=device)
        if self.rank == 0:
            unique_id = self._extension.get_unique_id()
            if len(unique_id) != unique_id_size:
                raise NCCLDeviceUnavailableError(
                    "NCCL unique-id size returned by the extension is inconsistent."
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
                max_bytes,
                int(device.index or 0),
                self.timeout_ms,
            )
        )
        self._initialized = True
        logger.info(
            "Initialized nccl_device transport rank=%s world_size=%s window_bytes=%s",
            self.rank,
            self.world_size,
            max_bytes,
        )
        return (time.perf_counter() - start_time) * 1000.0

    def _run(
        self, batch: _DeviceBatch, sender: bool, sequence: int
    ) -> Dict[str, float]:
        run_start = time.perf_counter()
        if not self._logged_batch_shape:
            logger.info(
                "Lowered nccl_device plan rank=%s sender=%s tasks=%s "
                "payload_bytes=%s chunk_bytes=%s expected_counts=%s",
                self.rank,
                sender,
                len(batch.tensors),
                sum(batch.lengths),
                self.chunk_bytes,
                batch.expected_counts,
            )
            self._logged_batch_shape = True
        device = torch.device(device_util.get_torch_device())
        region_metadata_start = time.perf_counter()
        if self._region_sizes is None:
            region_sizes = torch.tensor(
                batch.region_bytes, dtype=torch.int64, device=device
            )
            dist.all_reduce(region_sizes, op=dist.ReduceOp.MAX, group=self.group)
            self._region_sizes = [int(value) for value in region_sizes.cpu().tolist()]
        elif any(
            requested > allocated
            for requested, allocated in zip(batch.region_bytes, self._region_sizes)
        ):
            raise NCCLDeviceUnavailableError(
                "nccl_device transfer plan grew after its symmetric window was created"
            )
        region_sizes_list = self._region_sizes
        region_offsets = []
        total_bytes = 0
        for region_size in region_sizes_list:
            region_offsets.append(total_bytes)
            total_bytes += region_size
        absolute_offsets = [
            region_offsets[region] + offset
            for region, offset in zip(batch.region_indices, batch.offsets)
        ]
        region_metadata_time_ms = (
            time.perf_counter() - region_metadata_start
        ) * 1000.0
        transport_init_time_ms = self._ensure_initialized(total_bytes)
        assert self._extension is not None and self._handle is not None
        extension_metrics = dict(
            self._extension.launch(
                self._handle,
                batch.tensors,
                absolute_offsets,
                batch.lengths,
                batch.peers,
                batch.ordinals,
                batch.expected_counts,
                bool(sender),
                int(sequence),
            )
        )
        python_copyback_start = time.perf_counter()
        if batch.copybacks:
            with torch.no_grad():
                for destination, staging in batch.copybacks:
                    destination.copy_(staging)
            torch.cuda.current_stream().synchronize()
        python_copyback_time_ms = (
            time.perf_counter() - python_copyback_start
        ) * 1000.0
        extension_metrics.update(
            {
                "payload_bytes": float(sum(batch.lengths)),
                "region_metadata_time_ms": region_metadata_time_ms,
                "transport_init_time_ms": transport_init_time_ms,
                "python_copyback_time_ms": python_copyback_time_ms,
                "transport_total_time_ms": (
                    time.perf_counter() - run_start
                )
                * 1000.0,
            }
        )
        extension_metrics["reader_copyback_total_time_ms"] = (
            extension_metrics.get("reader_copyback_time_ms", 0.0)
            + python_copyback_time_ms
        )
        return extension_metrics

    def send(
        self, parameters: dict, plan: TransferPlan, step_id: int
    ) -> Dict[str, float]:
        build_batch_time_ms = 0.0
        prepared = self._prepared_send
        if prepared is not None and prepared[0] is parameters and prepared[1] is plan:
            batch = prepared[2]
        else:
            build_start = time.perf_counter()
            batch = _build_send_batch(
                parameters, plan, self.rank, self.world_size, self.chunk_bytes
            )
            build_batch_time_ms = (time.perf_counter() - build_start) * 1000.0
        metrics = self._run(
            batch, sender=True, sequence=_sequence_from_step(step_id)
        )
        metrics["build_batch_time_ms"] = build_batch_time_ms
        return metrics

    def recv(
        self, parameters: dict, plan: TransferPlan, step_id: int
    ) -> Dict[str, float]:
        build_batch_time_ms = 0.0
        prepared = self._prepared_recv
        if prepared is not None and prepared[0] is parameters and prepared[1] is plan:
            batch = prepared[2]
        else:
            build_start = time.perf_counter()
            batch = _build_recv_batch(
                parameters, plan, self.rank, self.world_size, self.chunk_bytes
            )
            build_batch_time_ms = (time.perf_counter() - build_start) * 1000.0
        metrics = self._run(
            batch, sender=False, sequence=_sequence_from_step(step_id)
        )
        metrics["build_batch_time_ms"] = build_batch_time_ms
        return metrics

    def prepare_send(
        self,
        parameters: dict,
        plan: TransferPlan,
        allow_staging: bool = True,
    ) -> None:
        """Bind stable source tensors to device tasks once during initialization."""

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
        """Bind stable destination tensors to device tasks once during initialization."""
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
            self._region_sizes = None
            self._prepared_send = None
            self._prepared_recv = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

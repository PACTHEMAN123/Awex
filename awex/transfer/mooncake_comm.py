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

"""Mooncake data transfers, with AWEX metadata used only for rendezvous."""

import os
import threading
from collections.abc import Callable

import torch

from awex.meta.meta_server import MetaServerClient
from awex.transfer.transfer_plan import TransferPlan, slice_tensor
from awex.util import device as device_util
from awex.util.common import get_ip_address


def _op_signature(op):
    return (
        op.send_shard_meta.name,
        op.recv_shard_meta.name,
        tuple(op.send_offset),
        tuple(op.recv_offset),
        tuple(op.overlap_shape),
        tuple((s.start, s.stop, s.step) for s in op.train_slices),
        tuple((s.start, s.stop, s.step) for s in op.inf_slices),
    )


class MooncakeTransport:
    """Keep exported buffers alive until every reader acknowledges completion.

    A failed exchange must abort the job. In particular, a metadata timeout does
    not prove that remote DMA has stopped: retain registered buffers on failure
    and reject subsequent exchanges instead of allowing their storage to be reused.
    """

    registration_policy = "persistent_snapshots"

    def __init__(self, meta_client: MetaServerClient, rank: int, timeout: float):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as exc:
            raise ImportError(
                "Mooncake requires the optional dependency: pip install 'awex[mooncake]'"
            ) from exc
        self.protocol = os.environ.get("AWEX_MOONCAKE_PROTOCOL", "rdma")
        if self.protocol not in ("rdma", "tcp"):
            raise ValueError("AWEX_MOONCAKE_PROTOCOL must be rdma or tcp")
        if device_util.get_device_type() not in ("cpu", "cuda"):
            raise ValueError("Mooncake weight exchange supports CPU and CUDA devices")
        self.engine = TransferEngine()
        host = os.environ.get("AWEX_MOONCAKE_HOST") or get_ip_address()
        self._check(
            self.engine.initialize(
                host,
                "P2PHANDSHAKE",
                self.protocol,
                os.environ.get("AWEX_MOONCAKE_DEVICE", ""),
            ),
            "initialize",
        )
        self.endpoint = f"{host}:{self.engine.get_rpc_port()}"
        self.meta_client = meta_client
        self.rank = rank
        self.timeout = timeout
        self.sequence = 0
        self.failed = False
        self.buffers = {}
        self._snapshots = {}
        self._staging = {}
        self._state_lock = threading.Lock()
        self._active = False
        self._closed = False
        self._send_layout = None

    @staticmethod
    def _check(result, operation):
        if result != 0:
            raise RuntimeError(f"Mooncake {operation} failed with code {result}")

    def _begin(self, step_id):
        with self._state_lock:
            if self.failed or self._closed:
                raise RuntimeError(
                    "Previous Mooncake exchange failed or closed; restart the job"
                )
            if self._active:
                raise RuntimeError("A Mooncake exchange is already active")
            self._active = True
            # An invocation counter prevents stale descriptors/acknowledgments
            # when multiple publications use the same training step.
            key = f"mooncake_{step_id}_{self.sequence}"
            self.sequence += 1
            return key

    def _end(self, *, failed=False):
        with self._state_lock:
            self.failed = self.failed or failed
            self._active = False

    def close(self):
        """Release idle registrations; a failed publication requires job teardown.

        A timeout cannot establish that peer DMA has stopped. Failed or active
        transports therefore retain their allocations instead of recycling them.
        """
        with self._state_lock:
            if self.failed or self._active:
                raise RuntimeError("Cannot close an active or failed Mooncake exchange")
            self._closed = True
            self._release()
            self._snapshots.clear()
            self._staging.clear()

    def _register(self, tensor):
        address = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        if size:
            self._check(self.engine.register_memory(address, size), "register_memory")
            self.buffers[address] = tensor
        return address, size

    def _release(self):
        for address in list(self.buffers):
            self._check(self.engine.unregister_memory(address), "unregister_memory")
            del self.buffers[address]

    @torch.no_grad()
    def send(
        self,
        parameters: dict[str, torch.Tensor],
        plan: TransferPlan,
        step_id: int,
        *,
        offload: bool = False,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        """Snapshot and consume converted parameters, then serve all peer reads."""
        prefix = self._begin(step_id)
        published = False
        try:
            descriptors = {}
            slices = {}
            layout = []
            for peer, operations in sorted(plan.operations.items()):
                for op in operations:
                    source = slice_tensor(parameters[op.send_shard_meta.name], op, True)
                    device = (
                        torch.device("cpu")
                        if offload or self.protocol == "tcp"
                        else source.device
                    )
                    layout.append(
                        (
                            peer,
                            _op_signature(op),
                            tuple(source.shape),
                            source.dtype,
                            device,
                        )
                    )
            layout = tuple(layout)
            if self._send_layout is not None and layout != self._send_layout:
                raise ValueError("Mooncake publication layout changed; restart the job")
            self._send_layout = layout
            for peer, operations in sorted(plan.operations.items()):
                entries = []
                for op in operations:
                    cache_key = (
                        op.send_shard_meta.name,
                        tuple((s.start, s.stop, s.step) for s in op.train_slices),
                    )
                    if cache_key not in slices:
                        source = slice_tensor(
                            parameters[op.send_shard_meta.name], op, True
                        )
                        target_device = (
                            "cpu"
                            if offload or self.protocol == "tcp"
                            else source.device
                        )
                        tensor = self._snapshots.get(cache_key)
                        if tensor is None:
                            # Own a stable, compact snapshot independently of
                            # converter/model allocations, which may be replaced.
                            tensor = torch.empty(
                                source.shape, dtype=source.dtype, device=target_device
                            )
                            self._register(tensor)
                            self._snapshots[cache_key] = tensor
                        tensor.copy_(source)
                        del source
                        slices[cache_key] = (
                            tensor.data_ptr(),
                            tensor.numel() * tensor.element_size(),
                            tuple(tensor.shape),
                            str(tensor.dtype),
                        )
                    entries.append((_op_signature(op), *slices[cache_key]))
                descriptors[peer] = entries
            device_util.synchronize()
            # Drop converted tensors before the colocate callback offloads model
            # weights. Registered snapshots remain owned by this transport.
            parameters.clear()
            if on_ready is not None:
                on_ready()
            for peer, entries in descriptors.items():
                # A failed HTTP response may still have published the addresses.
                published = True
                self.meta_client.put_object(
                    f"{prefix}_{self.rank}_{peer}_buffers", (self.endpoint, entries)
                )
            for peer in descriptors:
                key = f"{prefix}_{self.rank}_{peer}"
                error = self.meta_client.get_object(f"{key}_done", timeout=self.timeout)
                if error:
                    raise RuntimeError(f"Mooncake reader {peer} failed: {error}")
            # Every peer has completed its read. Keep registered snapshots for
            # the next publication; only now is overwriting them safe.
            for peer in descriptors:
                key = f"{prefix}_{self.rank}_{peer}"
                self.meta_client.delete_if_exists(f"{key}_buffers")
                self.meta_client.delete_if_exists(f"{key}_done")
        except Exception:
            self._end(failed=True)
            if not published:
                self._release()
                self._snapshots.clear()
            raise
        else:
            self._end()

    @torch.no_grad()
    def receive(
        self, parameters: dict[str, torch.Tensor], plan: TransferPlan, step_id: int
    ) -> None:
        prefix = self._begin(step_id)
        try:
            for peer, operations in sorted(plan.operations.items()):
                key = f"{prefix}_{peer}_{self.rank}"
                endpoint, entries = self.meta_client.get_object(
                    f"{key}_buffers", timeout=self.timeout
                )
                try:
                    if len(entries) != len(operations):
                        raise ValueError(
                            "Mooncake transfer plan operation count mismatch"
                        )
                    for op, entry in zip(operations, entries, strict=False):
                        signature, address, size, shape, dtype = entry
                        target = slice_tensor(
                            parameters[op.recv_shard_meta.name], op, False
                        )
                        if (
                            signature != _op_signature(op)
                            or tuple(target.shape) != shape
                            or str(target.dtype) != dtype
                            or target.numel() * target.element_size() != size
                        ):
                            raise ValueError("Mooncake transfer plan tensor mismatch")
                        if not size:
                            continue
                        # Staging supports strided slices and transposed converter
                        # views; TCP uses host buffers on both sides.
                        device = (
                            torch.device("cpu")
                            if self.protocol == "tcp"
                            else target.device
                        )
                        cache_key = (target.dtype, device)
                        storage = self._staging.get(cache_key)
                        count = target.numel()
                        if storage is None or storage.numel() < count:
                            # The preceding synchronous read and target copy
                            # have finished, so an undersized buffer can retire.
                            if storage is not None:
                                old_address = storage.data_ptr()
                                self._check(
                                    self.engine.unregister_memory(old_address),
                                    "unregister_memory",
                                )
                                del self.buffers[old_address]
                                del self._staging[cache_key]
                            storage = torch.empty(
                                count, dtype=target.dtype, device=device
                            )
                            self._register(storage)
                            self._staging[cache_key] = storage
                        buffer = storage[:count].view(shape)
                        local_address = buffer.data_ptr()
                        device_util.synchronize()
                        self._check(
                            self.engine.transfer_sync_read(
                                endpoint, local_address, address, size
                            ),
                            "transfer_sync_read",
                        )
                        target.copy_(buffer)
                        # Finish the copy before reusing persistent staging.
                        device_util.synchronize()
                    self.meta_client.put_object(f"{key}_done", "")
                except Exception as exc:
                    self.meta_client.put_object(f"{key}_done", str(exc))
                    raise
        except Exception:
            self._end(failed=True)
            raise
        else:
            self._end()

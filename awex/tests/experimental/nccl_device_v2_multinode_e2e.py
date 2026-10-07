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

"""Two-rank point-to-point or four-rank ring payload test for v2 GIN."""

# ruff: noqa: E402, I001

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("AWEX_NCCL_INCLUDE", "/usr/local/cuda/include")
os.environ.setdefault("AWEX_NCCL_LIB", "/usr/local/cuda/targets/x86_64-linux/lib")

# The transport has no model-stack dependency, but awex.__init__ eagerly imports
# optional reader/model modules. Keep this standalone hardware test runnable in
# the minimal NCCL containers by exposing the source tree as the package root.
_awex_package = types.ModuleType("awex")
_awex_package.__path__ = [str(Path(__file__).resolve().parents[2])]
sys.modules.setdefault("awex", _awex_package)

from awex.transfer.nccl_device_v2 import NCCLDeviceV2Transport
from awex.transfer.transfer_plan import (
    CommunicationOperation,
    TransferPlan,
)
import torch
import torch.distributed as dist


_WEIGHT_SPECS = {
    "model.dense.weight": ((1024, 1024), torch.float32),
    "model.expert.weight": ((2048, 1024), torch.float16),
    "model.norm.weight": ((4096,), torch.float32),
}
_STEP_VALUES = ((1.25, -2.5, 7.0), (3.5, 0.75, -4.0))


def _make_plan(rank: int, writer: int, readers: list[int]) -> TransferPlan:
    operations = {}
    for reader in readers:
        if rank not in (writer, reader):
            continue
        peer_ops = []
        for name, (shape, _) in _WEIGHT_SPECS.items():
            shard = SimpleNamespace(name=name, shape=shape)
            slices = tuple(slice(None) for _ in shape)
            peer_ops.append(
                CommunicationOperation(
                    send_rank=writer,
                    send_shard_meta=shard,
                    send_offset=tuple(0 for _ in shape),
                    recv_rank=reader,
                    recv_shard_meta=shard,
                    recv_offset=tuple(0 for _ in shape),
                    overlap_shape=shape,
                    train_slices=slices,
                    inf_slices=slices,
                )
            )
        operations[reader if rank == writer else writer] = peer_ops
    return TransferPlan(operations=operations)


def _set_values(parameters: dict[str, torch.Tensor], values: tuple[float, ...]) -> None:
    with torch.no_grad():
        for tensor, value in zip(parameters.values(), values):
            # Position-dependent values catch misaddressed or prematurely reused
            # FIFO slices; constant fills cannot distinguish two equal slices.
            pattern = (torch.arange(tensor.numel(), device=tensor.device) % 97).reshape(
                tensor.shape
            )
            tensor.copy_(pattern.to(torch.float32) / 128 + value)
    torch.cuda.synchronize()


def _verify_values(
    parameters: dict[str, torch.Tensor], values: tuple[float, ...]
) -> None:
    for (name, tensor), value in zip(parameters.items(), values):
        pattern = (torch.arange(tensor.numel(), device=tensor.device) % 97).reshape(
            tensor.shape
        )
        expected = (pattern.to(torch.float32) / 128 + value).to(tensor.dtype)
        if not torch.equal(tensor, expected):
            difference = float((tensor.float() - expected.float()).abs().max().item())
            raise AssertionError(
                f"weight mismatch for {name}: max_abs_diff={difference}"
            )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4):
        raise RuntimeError("nccl_device_v2_multinode_e2e requires two or four ranks")
    ring = world_size == 4
    writer = world_size - 1 if ring else 0
    readers = list(range(world_size - 1)) if ring else [1]
    if ring:
        _WEIGHT_SPECS["model.dense.weight"] = ((4096, 1024), torch.float32)
        _WEIGHT_SPECS["model.expert.weight"] = ((8192, 1024), torch.float16)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    parameters = {
        name: torch.zeros(shape, dtype=dtype, device="cuda")
        for name, (shape, dtype) in _WEIGHT_SPECS.items()
    }
    plan = _make_plan(rank, writer, readers)
    transport = NCCLDeviceV2Transport(
        dist.group.WORLD,
        rank,
        world_size,
        timeout_ms=120_000,
        chunk_bytes=4 * 1024 * 1024,
        infer_instance_world_size=1 if ring else 0,
        num_infer_engines=len(readers),
        ring_broadcast=ring,
        ring_swizzle=ring,
    )

    try:
        if rank == writer:
            transport.prepare_send(parameters, plan, allow_staging=False)
        else:
            transport.prepare_recv(parameters, plan, allow_staging=False)

        launch_metrics = []
        updates = 10 if ring else 2
        for index in range(updates):
            step_id = index - 1
            values = tuple(value + index / 8 for value in _STEP_VALUES[index % 2])
            if rank == writer:
                _set_values(parameters, values)
            else:
                _set_values(parameters, (0.0, 0.0, 0.0))
            dist.barrier(device_ids=[local_rank])
            if rank == writer:
                metrics = transport.send(parameters, plan, step_id)
            else:
                metrics = transport.recv(parameters, plan, step_id)
            launch_metrics.append(metrics)
            dist.barrier(device_ids=[local_rank])
            if rank in readers:
                _verify_values(parameters, values)

        metrics = launch_metrics[-1]
        expected_peers = 2 if ring and metrics.get("ring_relay_count", 0) else 1
        if (
            metrics["lsa_peer_count"] != 0
            or metrics["gin_peer_count"] != expected_peers
        ):
            raise AssertionError(f"expected the GIN path, got {metrics}")
        if not metrics["gin_enabled"] or metrics["gin_context_count"] <= 0:
            raise AssertionError(f"GIN was not initialized: {metrics}")
        if metrics["gin_connection_count"] <= 0:
            raise AssertionError(f"GIN has no network connections: {metrics}")
        if metrics["network_step_bytes"] != transport.network_step_bytes:
            raise AssertionError(f"expected NCCL network step size: {metrics}")
        if metrics["fifo_depth"] != 8 or metrics["gin_fifo_depth"] != 16:
            raise AssertionError(
                f"LSA and GIN FIFO settings were not isolated: {metrics}"
            )
        if metrics["channel_count"] != metrics["network_channels_per_peer"]:
            raise AssertionError(f"GIN did not use its network channels: {metrics}")
        if launch_metrics[0]["plan_cache_hit"]:
            raise AssertionError(f"first update unexpectedly hit cache: {metrics}")
        if not launch_metrics[1]["plan_cache_hit"]:
            raise AssertionError(f"second update missed plan cache: {metrics}")
        if metrics["build_batch_time_ms"] != 0.0:
            raise AssertionError(f"prepared update rebuilt its batch: {metrics}")

        dist.barrier(device_ids=[local_rank])
        print(
            "AWEX_V2_MULTINODE_E2E_PASS "
            + json.dumps(
                {
                    "rank": rank,
                    "payload_bytes": metrics["payload_bytes"],
                    "gin_type": metrics["gin_type"],
                    "gin_context_count": metrics["gin_context_count"],
                    "gin_connection_count": metrics["gin_connection_count"],
                    "channel_count": metrics["channel_count"],
                    "network_channels_per_peer": metrics["network_channels_per_peer"],
                    "network_step_bytes": metrics["network_step_bytes"],
                    "registered_window_bytes": metrics["registered_window_bytes"],
                    "kernel_transfer_time_ms": metrics["kernel_transfer_time_ms"],
                    "plan_cache_hit": metrics["plan_cache_hit"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        transport.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

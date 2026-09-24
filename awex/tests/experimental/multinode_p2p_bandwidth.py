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

"""Measure aggregate GPU-to-GPU bandwidth between two equal-size nodes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("nccl", "nccl_device_v2"), required=True
    )
    parser.add_argument("--tensor-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    return parser.parse_args()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _pair_for_rank(rank: int, ranks_per_node: int) -> tuple[int, bool]:
    if rank < ranks_per_node:
        return rank + ranks_per_node, True
    return rank - ranks_per_node, False


def _make_plan(rank: int, peer: int, sender: bool, tensor_bytes: int):
    from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan

    writer_rank = rank if sender else peer
    reader_rank = peer if sender else rank
    shard = SimpleNamespace(name="payload", shape=(tensor_bytes,))
    operation = CommunicationOperation(
        send_rank=writer_rank,
        send_shard_meta=shard,
        send_offset=(0,),
        recv_rank=reader_rank,
        recv_shard_meta=shard,
        recv_offset=(0,),
        overlap_shape=(tensor_bytes,),
        train_slices=(slice(None),),
        inf_slices=(slice(None),),
    )
    return TransferPlan(operations={peer: [operation]})


def _run_device_v2(args, torch, dist, rank: int, peer: int, sender: bool):
    from awex.transfer.nccl_device_v2 import NCCLDeviceV2Transport

    tensor = torch.empty(args.tensor_bytes, dtype=torch.uint8, device="cuda")
    tensor.fill_((rank + 1) & 0xFF if sender else 0)
    plan = _make_plan(rank, peer, sender, args.tensor_bytes)
    parameters = {"payload": tensor}
    transport = NCCLDeviceV2Transport(
        dist.group.WORLD,
        rank,
        dist.get_world_size(),
        timeout_ms=args.timeout_ms,
    )
    if sender:
        transport.prepare_send(parameters, plan, allow_staging=False)
    else:
        transport.prepare_recv(parameters, plan, allow_staging=False)

    def run(sequence: int) -> tuple[float, dict]:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        metrics = (
            transport.send(parameters, plan, sequence)
            if sender
            else transport.recv(parameters, plan, sequence)
        )
        return float(metrics["kernel_transfer_time_ms"]), metrics

    return tensor, run, transport.close


def _run_nccl(args, torch, dist, rank: int, peer: int, sender: bool):
    tensor = torch.empty(args.tensor_bytes, dtype=torch.uint8, device="cuda")
    tensor.fill_((rank + 1) & 0xFF if sender else 0)

    def run(_sequence: int) -> tuple[float, dict]:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation = (
            dist.P2POp(dist.isend, tensor, peer)
            if sender
            else dist.P2POp(dist.irecv, tensor, peer)
        )
        requests = dist.batch_isend_irecv([operation])
        for request in requests:
            request.wait()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms, {}

    return tensor, run, lambda: None


def main() -> None:
    args = _parse_args()
    if args.tensor_bytes <= 0 or args.warmups < 0 or args.iterations <= 0:
        raise ValueError("tensor bytes and iterations must be positive")

    # Device v2 must preload the configured NCCL before importing torch.
    if args.backend == "nccl_device_v2":
        from awex.transfer import nccl_device_v2 as _device_v2  # noqa: F401

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 * local_world_size:
        raise RuntimeError(
            "multinode_p2p_bandwidth requires two nodes with equal process counts"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    peer, sender = _pair_for_rank(rank, local_world_size)
    factory = _run_device_v2 if args.backend == "nccl_device_v2" else _run_nccl
    tensor, run, close = factory(args, torch, dist, rank, peer, sender)

    timings = []
    last_metrics = {}
    try:
        for sequence in range(args.warmups + args.iterations):
            elapsed_ms, last_metrics = run(sequence)
            critical_path = torch.tensor([elapsed_ms], dtype=torch.float64, device="cuda")
            dist.all_reduce(critical_path, op=dist.ReduceOp.MAX)
            if sequence >= args.warmups:
                timings.append(float(critical_path.item()))

        dist.barrier(device_ids=[local_rank])
        if not sender:
            expected = ((peer + 1) & 0xFF)
            samples = tensor[:: max(1, args.tensor_bytes // 1024)]
            if not torch.all(samples == expected).item():
                raise AssertionError(
                    f"rank {rank} received corrupt data from rank {peer}"
                )
        dist.barrier(device_ids=[local_rank])

        if rank == 0:
            payload_bytes = args.tensor_bytes * local_world_size
            p50_ms = statistics.median(timings)
            result = {
                "backend": args.backend,
                "ranks_per_node": local_world_size,
                "tensor_bytes_per_rank": args.tensor_bytes,
                "aggregate_payload_bytes": payload_bytes,
                "iterations": args.iterations,
                "p50_ms": p50_ms,
                "p95_ms": _percentile(timings, 0.95),
                "min_ms": min(timings),
                "max_ms": max(timings),
                "aggregate_gb_s": payload_bytes / (p50_ms / 1000.0) / 1e9,
                "per_rank_gb_s": args.tensor_bytes / (p50_ms / 1000.0) / 1e9,
            }
            for name in (
                "channel_count",
                "fifo_depth",
                "gin_connection_count",
                "gin_context_count",
                "gin_credit_batch",
                "network_channel_budget",
                "network_channels_per_peer",
                "network_step_bytes",
            ):
                if name in last_metrics:
                    result[name] = last_metrics[name]
            print("AWEX_MULTINODE_P2P_BANDWIDTH " + json.dumps(result, sort_keys=True))
    finally:
        close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

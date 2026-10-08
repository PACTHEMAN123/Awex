"""Late-process joins for device v2; CPU mock is explicitly not CUDA validation.

Run as a module from the checkout. The driver starts only initial workers; each
new rollout worker is spawned after its requested publication boundary. With
--external, run the emitted worker commands on selected GPU hosts instead.
"""

from __future__ import annotations

# Import order is intentional: configured NCCL must load before PyTorch.
# ruff: noqa: E402, I001

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace

# Bind HCA policy to the worker's physical/node-local GPU before importing the
# device transport. Logical transfer ranks move when readers are appended.
if "--worker" in sys.argv:
    _bootstrap = argparse.ArgumentParser(add_help=False)
    _bootstrap.add_argument("--device", type=int, default=0)
    _bootstrap_args, _ = _bootstrap.parse_known_args()
    os.environ["LOCAL_RANK"] = str(_bootstrap_args.device)

from awex.transfer import nccl_device_v2 as device_v2

import torch
import torch.distributed as dist
from awex.transfer.rollout_membership import (
    Participant,
    RolloutJoinCoordinator,
    RolloutMembership,
)
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan
from awex.util.process_group import init_custom_process_group

NAME = "model.dense.weight"
NORM = "model.norm.weight"


def _key(participant: Participant) -> str:
    return "/".join(map(str, participant))


def _source(shape: tuple[int, int], version: int, tp: int, device: str) -> torch.Tensor:
    index = torch.arange(shape[0] * shape[1], device=device).reshape(shape)
    return (((index * 13 + version * 11 + tp * 31) % 1009).float() / 64 - 7.5).to(
        torch.bfloat16
    )


def _quantize(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = source.shape
    tiled = source.float().reshape(rows // 128, 128, cols // 128, 128)
    scales = tiled.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    values = (
        (tiled * scales.reciprocal()[:, None, :, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    return values.reshape(source.shape), scales


def _plan(
    membership: RolloutMembership,
    participant: Participant,
    shape: tuple[int, int],
    fp8: bool,
) -> TransferPlan:
    rank = membership.rank(participant)
    operations = {}
    for tp in range(membership.inference_tp_size):
        writer = membership.inference_world_size + tp
        for engine in membership.engine_ids:
            reader = membership.rank(("rollout", engine, tp))
            if rank not in (writer, reader):
                continue
            ops = []
            for name, dimensions in ((NAME, shape), (NORM, (128,))):
                source = SimpleNamespace(name=name, shape=dimensions, dtype="bfloat16")
                target = SimpleNamespace(
                    name=name,
                    shape=dimensions,
                    dtype="float8_e4m3fn" if fp8 and name == NAME else "bfloat16",
                )
                slices = tuple(slice(None) for _ in dimensions)
                ops.append(
                    CommunicationOperation(
                        writer,
                        source,
                        (0,) * len(dimensions),
                        reader,
                        target,
                        (0,) * len(dimensions),
                        dimensions,
                        slices,
                        slices,
                    )
                )
            operations[reader if rank == writer else writer] = ops
    return TransferPlan(operations=operations)


def _mock_run(self, batch, sender: bool, sequence: int) -> dict:
    """Replace ONLY CUDA execution for the portable control-plane check.

    Real device v2 send/recv, plan lowering, prepare, and reconfigure still run.
    Gloo broadcast does not validate FIFO, GIN, or relay-kernel execution.
    """
    parameters, participant, shape, fp8 = self._mock_state
    for tp in range(self.infer_instance_world_size):
        own = participant[2] == tp
        source_rank = self.infer_instance_world_size * self.num_infer_engines + tp
        payload = (
            parameters[NAME]
            if sender and own
            else torch.empty(shape, dtype=torch.bfloat16)
        )
        dist.broadcast(payload, src=source_rank, group=self.group)
        norm = (
            parameters[NORM]
            if sender and own
            else torch.empty(128, dtype=torch.bfloat16)
        )
        dist.broadcast(norm, src=source_rank, group=self.group)
        if not sender and own:
            if fp8:
                quant, scales = _quantize(payload)
                parameters[NAME].copy_(quant)
                parameters[NAME + "_scale_inv"].copy_(scales)
            else:
                parameters[NAME].copy_(payload)
            parameters[NORM].copy_(norm)
    hit = getattr(self, "_mock_last_epoch", None) == self.membership_epoch
    self._mock_last_epoch = self.membership_epoch
    return {
        "execution": "gloo_cpu_mock",
        "plan_cache_hit": hit,
        "payload_bytes": sum(batch.lengths),
    }


def _store(args, server: bool):
    return dist.TCPStore(
        args.control_address,
        args.control_port,
        None,
        server,
        timedelta(seconds=args.timeout),
        wait_for_workers=False,
    )


def worker(args) -> None:
    participant = (
        args.role,
        args.engine if args.role == "rollout" else "",
        args.tp_rank,
    )
    store = _store(args, False)
    transport = group = None
    parameters = None
    shape = (args.rows, args.cols)
    device = "cpu" if args.backend == "cpu-mock" else "cuda"
    if device == "cuda":
        torch.cuda.set_device(args.device)
    else:
        torch.set_num_threads(1)
        device_v2._ensure_cuda_tensor = lambda *_: None
    os.environ["AWEX_NCCL_DEVICE_V2_FP8_BLOCKWISE"] = "1" if args.fp8 else "0"
    os.environ["AWEX_NCCL_DEVICE_V2_FP8_BLOCK_ROWS"] = "128"
    os.environ["AWEX_NCCL_DEVICE_V2_FP8_BLOCK_COLS"] = "128"
    # Independent singleton defaults keep training process identity unchanged
    # while the borrowed transfer process group grows from epoch to epoch.
    dist.init_process_group(
        "gloo" if device == "cpu" else "nccl",
        store=dist.PrefixStore("solo/" + _key(participant), store),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=args.timeout),
    )
    index = 0
    try:
        while True:
            command = json.loads(
                store.get(f"command/{_key(participant)}/{index}").decode()
            )
            started = time.perf_counter()
            try:
                if command["op"] == "stop":
                    if transport is not None:
                        transport.close()
                    if group is not None:
                        dist.destroy_process_group(group)
                    store.set(
                        f"reply/{_key(participant)}/{index}", json.dumps({"ok": True})
                    )
                    break
                if command["op"] == "prepare":
                    membership = RolloutMembership(**command["membership"])
                    new_group = init_custom_process_group(
                        backend="gloo" if device == "cpu" else "nccl",
                        store=dist.PrefixStore(f"transfer/{membership.epoch}", store),
                        rank=membership.rank(participant),
                        world_size=membership.world_size,
                        group_name=f"dynamic-{membership.epoch}",
                        timeout=timedelta(seconds=args.timeout),
                    )
                    plan = _plan(membership, participant, shape, args.fp8)
                    if parameters is None:
                        dtype = (
                            torch.float8_e4m3fn
                            if args.fp8 and args.role == "rollout"
                            else torch.bfloat16
                        )
                        parameters = {
                            NAME: torch.full(shape, -0.25, dtype=dtype, device=device),
                            NORM: torch.full(
                                (128,), -0.25, dtype=torch.bfloat16, device=device
                            ),
                        }
                        if args.fp8 and args.role == "rollout":
                            parameters[NAME + "_scale_inv"] = torch.full(
                                (shape[0] // 128, shape[1] // 128), -17.0, device=device
                            )
                        transport = device_v2.NCCLDeviceV2Transport(
                            new_group,
                            membership.rank(participant),
                            membership.world_size,
                            infer_instance_world_size=membership.inference_tp_size,
                            num_infer_engines=len(membership.engine_ids),
                            ring_broadcast=args.ring != "off",
                            ring_swizzle=args.ring == "swizzle",
                            membership_epoch=membership.epoch,
                            timeout_ms=args.timeout * 1000,
                        )
                        if device == "cpu":
                            transport._mock_state = (
                                parameters,
                                participant,
                                shape,
                                args.fp8,
                            )
                            transport._run = MethodType(_mock_run, transport)
                        prepare = (
                            transport.prepare_send
                            if args.role == "training"
                            else transport.prepare_recv
                        )
                        prepare(parameters, plan, allow_staging=False)
                    else:
                        transport.reconfigure(
                            new_group, membership, participant, parameters, plan
                        )
                        dist.destroy_process_group(group)
                    group = new_group
                    result = {
                        "epoch": membership.epoch,
                        "rank": membership.rank(participant),
                        "pid": os.getpid(),
                        "pointers": {
                            name: tensor.data_ptr()
                            for name, tensor in parameters.items()
                        },
                    }
                elif command["op"] == "publish":
                    version = command["version"]
                    source = _source(shape, version, args.tp_rank, device)
                    norm = (
                        torch.arange(128, device=device).to(torch.bfloat16) / 128
                        + version
                        + args.tp_rank
                    )
                    if args.role == "training":
                        parameters[NAME].copy_(source)
                        parameters[NORM].copy_(norm)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    dist.barrier(group=group)
                    transfer_start = time.perf_counter()
                    exchange = (
                        transport.send if args.role == "training" else transport.recv
                    )
                    metrics = exchange(parameters, plan, version)
                    transfer_ms = (time.perf_counter() - transfer_start) * 1000
                    if args.role == "training":
                        if not torch.equal(parameters[NAME], source):
                            raise AssertionError("Training BF16 weights were modified")
                    else:
                        if args.fp8:
                            expected, scales = _quantize(source)
                            if not torch.equal(
                                parameters[NAME].view(torch.uint8),
                                expected.view(torch.uint8),
                            ):
                                raise AssertionError("FP8 weight bytes differ")
                            torch.testing.assert_close(
                                parameters[NAME + "_scale_inv"],
                                scales,
                                rtol=1e-6,
                                atol=0,
                            )
                        elif not torch.equal(parameters[NAME], source):
                            raise AssertionError("BF16 weight bytes differ")
                    if not torch.equal(parameters[NORM], norm):
                        raise AssertionError("Norm weight bytes differ")
                    dist.barrier(group=group)
                    result = {
                        "epoch": transport.membership_epoch,
                        "version": version,
                        "pid": os.getpid(),
                        "pointers": {
                            name: tensor.data_ptr()
                            for name, tensor in parameters.items()
                        },
                        "transfer_ms": transfer_ms,
                        "metrics": metrics,
                        "verified": True,
                    }
                else:
                    raise ValueError("Unknown command")
                result.update(
                    ok=True,
                    command_ms=(time.perf_counter() - started) * 1000,
                    hostname=socket.gethostname(),
                    torch_version=torch.__version__,
                )
            except Exception:
                store.set(
                    f"reply/{_key(participant)}/{index}",
                    json.dumps({"ok": False, "error": traceback.format_exc()}),
                )
                raise
            store.set(f"reply/{_key(participant)}/{index}", json.dumps(result))
            index += 1
    finally:
        if transport is not None:
            transport.close()
        dist.destroy_process_group()


def driver(args) -> dict:
    final_world = args.tp * (1 + args.target_engines[-1])
    if final_world > 256:
        raise ValueError("Requested topology exceeds device v2's 256-rank limit")
    if (
        args.backend == "device-v2"
        and not args.external
        and torch.cuda.device_count() < final_world
    ):
        raise ValueError(
            f"Local device-v2 mode requires {final_world} visible GPUs; use --external for multiple hosts"
        )
    store = _store(args, True)
    membership = RolloutMembership(
        0, args.tp, args.tp, tuple(f"engine-{i}" for i in range(args.initial_engines))
    )
    coordinator = RolloutJoinCoordinator(membership)
    processes = []
    indexes = {}
    identities = {}
    records = []
    counts = args.target_engines
    joins = dict(zip(args.join_after, counts))

    def launch(participant):
        indexes[participant] = 0
        gpu = (
            participant[2]
            if participant[0] == "training"
            else args.tp + int(participant[1].split("-")[-1]) * args.tp + participant[2]
        )
        command = [
            args.worker_python or sys.executable,
            "-m",
            "awex.tests.experimental.nccl_device_v2_dynamic_e2e",
            "--worker",
            "--role",
            participant[0],
            "--engine",
            participant[1],
            "--tp-rank",
            str(participant[2]),
            "--device",
            str(gpu),
            "--control-address",
            args.control_address,
            "--control-port",
            str(store.port),
            "--backend",
            args.backend,
            "--ring",
            args.ring,
            "--rows",
            str(args.rows),
            "--cols",
            str(args.cols),
            "--timeout",
            str(args.timeout),
        ]
        if args.fp8:
            command.append("--fp8")
        if args.external:
            # This is an argv array, not a shell command. Set worker --device to
            # its node-local GPU index when launching on a different host.
            print(
                json.dumps(
                    {
                        "event": "launch_worker",
                        "participant": participant,
                        "argv": command,
                    }
                ),
                flush=True,
            )
        else:
            processes.append(subprocess.Popen(command))

    def dispatch(members, command):
        for participant in members.participants:
            store.set(
                f"command/{_key(participant)}/{indexes[participant]}",
                json.dumps(command),
            )
        replies = {}
        for participant in members.participants:
            reply = json.loads(
                store.get(f"reply/{_key(participant)}/{indexes[participant]}").decode()
            )
            indexes[participant] += 1
            if not reply["ok"]:
                raise RuntimeError(reply["error"])
            identity = {
                key: reply[key]
                for key in ("pid", "pointers", "hostname")
                if key in reply and "pid" in reply
            }
            if identity:
                if participant in identities and identities[participant] != identity:
                    raise AssertionError(
                        f"Existing process/model storage replaced: {participant}"
                    )
                identities[participant] = identity
            replies[_key(participant)] = reply
        return replies

    try:
        for participant in membership.participants:
            launch(participant)
        records.append(
            {
                "event": "initial_prepare",
                "ranks": dispatch(
                    membership, {"op": "prepare", "membership": asdict(membership)}
                ),
            }
        )
        for version in range(args.updates):
            membership = coordinator.begin_publication(version)
            replies = dispatch(membership, {"op": "publish", "version": version})
            for participant in membership.participants:
                reply = replies[_key(participant)]
                coordinator.acknowledge_updated(
                    participant, reply["epoch"], reply["version"]
                )
            coordinator.finish_publication()
            record = {
                "event": "publication",
                "version": version,
                "epoch": membership.epoch,
                "engine_ids": coordinator.serving_engine_ids,
                "source_payload_bytes": args.tp * (args.rows * args.cols * 2 + 256),
                "ranks": replies,
            }
            records.append(record)
            print(
                json.dumps(
                    {key: value for key, value in record.items() if key != "ranks"}
                ),
                flush=True,
            )
            completed = version + 1
            if completed in joins:
                target = joins[completed]
                for i in range(len(membership.engine_ids), target):
                    coordinator.request_join(f"engine-{i}", args.tp)
                expanded = coordinator.prepare_join()
                for participant in expanded.participants:
                    if participant not in indexes:
                        launch(participant)
                started = time.perf_counter()
                replies = dispatch(
                    expanded, {"op": "prepare", "membership": asdict(expanded)}
                )
                for participant in expanded.participants:
                    if replies[_key(participant)]["rank"] != expanded.rank(participant):
                        raise AssertionError(
                            "Worker prepared the wrong new transfer rank"
                        )
                    coordinator.acknowledge_prepared(
                        participant, replies[_key(participant)]["epoch"]
                    )
                coordinator.commit_join()
                assert not coordinator.serving_engine_ids
                record = {
                    "event": "join",
                    "after_updates": completed,
                    "epoch": expanded.epoch,
                    "engine_ids": expanded.engine_ids,
                    "next_version": version + 1,
                    "prepare_ms": (time.perf_counter() - started) * 1000,
                    "ranks": replies,
                }
                records.append(record)
                print(
                    json.dumps(
                        {key: value for key, value in record.items() if key != "ranks"}
                    ),
                    flush=True,
                )
        dispatch(coordinator.membership, {"op": "stop"})
        summary = {
            "passed": True,
            "backend": args.backend,
            "cuda_kernel_validated": args.backend == "device-v2",
            "fp8": args.fp8,
            "ring": args.ring,
            "updates": args.updates,
            "tp": args.tp,
            "shape": [args.rows, args.cols],
            "initial_engines": args.initial_engines,
            "join_after": args.join_after,
            "target_engines": args.target_engines,
            "records": records,
        }
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "AWEX_DYNAMIC_JOIN_PASS "
            + json.dumps(
                {key: value for key, value in summary.items() if key != "records"}
            ),
            flush=True,
        )
        return summary
    except Exception as exc:
        coordinator.fail(str(exc))
        raise
    finally:
        # Only child processes owned by this driver are stopped; external GPU
        # workers exit on their own bounded store/collective timeout on failure.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--backend", choices=("cpu-mock", "device-v2"), default="cpu-mock"
    )
    parser.add_argument("--external", action="store_true")
    parser.add_argument(
        "--worker-python", help="Interpreter for emitted remote worker commands"
    )
    parser.add_argument("--control-address", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--initial-engines", type=int, default=1)
    parser.add_argument("--join-after", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--target-engines", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument(
        "--ring", choices=("off", "naive", "swizzle"), default="swizzle"
    )
    parser.add_argument("--output")
    parser.add_argument("--role", choices=("training", "rollout"), default="rollout")
    parser.add_argument("--engine", default="engine-0")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.rows < 128 or args.cols < 128 or args.rows % 128 or args.cols % 128:
        parser.error("Matrix shape must be positive multiples of 128")
    if args.worker:
        if args.control_port == 0:
            parser.error("Workers require the driver's control port")
        worker(args)
    else:
        targets = [args.initial_engines] + args.target_engines
        if len(args.join_after) != len(args.target_engines) or any(
            a >= b for a, b in zip(targets, targets[1:])
        ):
            parser.error("Join boundaries require strictly increasing engine counts")
        if sorted(set(args.join_after)) != args.join_after or any(
            not 1 <= boundary < args.updates for boundary in args.join_after
        ):
            parser.error("Join boundaries must increase and leave a following update")
        if args.external and args.control_address == "127.0.0.1":
            parser.error("External workers require a reachable control address")
        driver(args)


if __name__ == "__main__":
    main()

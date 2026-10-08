"""One owned worker launcher per selected host for elastic device v2 testing.

Start agents once on the selected hosts. The benchmark driver asks them to
spawn late workers at join boundaries; agents never SSH or restart survivors.
This is a bounded, single-run test service, not a general remote executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import socket
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

MODULE = "awex.tests.experimental.nccl_device_v2_dynamic_e2e"


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    files = [
        root / "transfer" / name
        for name in (
            "nccl_device_v2.py",
            "nccl_device_v2_gin.py",
            "rollout_membership.py",
            "transfer_plan.py",
        )
    ]
    files += sorted((root / "transfer" / "nccl_device_v2").glob("*"))
    files += [
        Path(__file__),
        Path(__file__).with_name("nccl_device_v2_dynamic_e2e.py"),
        Path(__file__).with_name("dynamic_rollout_profile.py"),
    ]
    digest = hashlib.sha256()
    for path in files:
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_agent(args: argparse.Namespace) -> None:
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    store = dist.TCPStore(
        args.control_address,
        args.control_port,
        None,
        False,
        timedelta(seconds=args.timeout),
        wait_for_workers=False,
    )
    processes = []
    devices = set()
    index = 0
    try:
        while True:
            key = f"agent/{args.node}/command/{index}"
            command = json.loads(store.get(key).decode())
            stopping = command["op"] == "stop"
            try:
                if stopping:
                    result = {"ok": True}
                elif command["op"] == "describe":
                    result = {
                        "ok": True,
                        "hostname": socket.gethostname(),
                        "gpu_count": torch.cuda.device_count(),
                        "source_fingerprint": source_fingerprint(),
                    }
                elif command["op"] == "launch":
                    role, engine, tp = command["participant"]
                    gpu = command["device"]
                    if (
                        role not in ("training", "rollout")
                        or not isinstance(tp, int)
                        or tp < 0
                    ):
                        raise ValueError("Invalid worker participant")
                    if not isinstance(gpu, int) or gpu < 0 or gpu in devices:
                        raise ValueError("Worker GPU must be unique on this node")
                    if command["backend"] not in ("cpu-mock", "device-v2"):
                        raise ValueError("Invalid backend")
                    if (
                        command["backend"] == "device-v2"
                        and gpu >= torch.cuda.device_count()
                    ):
                        raise ValueError("Worker GPU is not visible on this node")
                    argv = [
                        args.worker_python or sys.executable,
                        "-m",
                        MODULE,
                        "--worker",
                        "--role",
                        role,
                        "--engine",
                        engine,
                        "--tp-rank",
                        str(tp),
                        "--device",
                        str(gpu),
                        "--control-address",
                        args.control_address,
                        "--control-port",
                        str(args.control_port),
                    ]
                    for setting in ("backend", "ring", "rows", "cols", "timeout"):
                        argv.extend(["--" + setting, str(command[setting])])
                    for setting in ("compute_batch", "compute_repeats"):
                        if setting in command:
                            argv.extend(
                                [
                                    "--" + setting.replace("_", "-"),
                                    str(command[setting]),
                                ]
                            )
                    if command["fp8"]:
                        argv.append("--fp8")
                    process = subprocess.Popen(argv)
                    processes.append(process)
                    devices.add(gpu)
                    result = {
                        "ok": True,
                        "pid": process.pid,
                        "hostname": socket.gethostname(),
                        "device": gpu,
                    }
                else:
                    raise ValueError("Unknown node-agent command")
            except Exception:
                result = {"ok": False, "error": traceback.format_exc()}
            store.set(f"agent/{args.node}/reply/{index}", json.dumps(result))
            index += 1
            if stopping:
                break
    finally:
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
    parser.add_argument("--node", required=True)
    parser.add_argument("--control-address", required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-python")
    args = parser.parse_args()
    run_agent(args)


if __name__ == "__main__":
    main()

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

"""Run the inference-side Awex reader without a training-node control link."""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from awex import logging

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _server_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    for name in list(env):
        if name.startswith("TORCHELASTIC_"):
            env.pop(name)
    env.update({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"})
    return env


def _inference_device_groups(args) -> list[list[str]]:
    required = args.vllm_tp_size * args.num_engines
    visible = [
        device.strip()
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    if not visible:
        visible = [str(device) for device in range(required)]
    if len(visible) < required:
        raise RuntimeError(
            f"Need {required} visible GPUs for {args.num_engines} vLLM engines "
            f"at TP{args.vllm_tp_size}, found {len(visible)}."
        )
    return [
        visible[start : start + args.vllm_tp_size]
        for start in range(0, required, args.vllm_tp_size)
    ]


def _start_vllm_server(
    args, engine_rank: int = 0, devices: list[str] | None = None
) -> subprocess.Popen:
    port = args.port + engine_rank
    cmd = [
        sys.executable,
        "-m",
        "awex.awex_vllm_server",
        "--model",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(args.vllm_tp_size),
        "--pipeline-parallel-size",
        "1",
        "--no-enable-log-requests",
        "--enforce-eager",
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
    ]
    env = _server_environment()
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    logger.info(
        "Starting inference-node vLLM engine %s/%s on devices %s: %s",
        engine_rank,
        args.num_engines,
        env.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        " ".join(cmd),
    )
    return subprocess.Popen(cmd, env=env)


def _wait_for_health(process: subprocess.Popen, host: str, port: int, timeout: int):
    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited with code {process.returncode} "
                "before becoming healthy."
            )
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(f"vLLM server failed to start within {timeout} seconds.")


def _post(url: str, payload: dict, timeout: int, action: str) -> None:
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"{action} failed: {response.text}")


def _init_engine(args, engine_rank: int) -> None:
    base_url = f"http://{args.client_host}:{args.port + engine_rank}"
    payload = {
        "meta_server_addr": args.meta_server_addr,
        "engine_rank": engine_rank,
        "num_engines": args.num_engines,
        "comm_backend": args.comm_backend,
        "enable_debug_mode": False,
        "nnodes": 1,
        "node_rank": 0,
    }
    if args.validate:
        payload["weights_validation_steps"] = 1
        payload["validate_weights_every_n_steps"] = 1
    _post(
        f"{base_url}/areal_awex_init",
        payload,
        timeout=120,
        action=f"Awex reader initialization for engine {engine_rank}",
    )


def _update_engine(args, engine_rank: int, step_id: int) -> None:
    _post(
        f"http://{args.client_host}:{args.port + engine_rank}/areal_awex_update",
        {"step_id": step_id, "kwargs": {}},
        timeout=args.update_timeout,
        action=f"Awex reader update {step_id} for engine {engine_rank}",
    )


def main(args) -> None:
    if args.profile:
        os.environ["AWEX_PROFILE"] = "1"
        os.environ["AWEX_PROFILE_WARMUP_UPDATES"] = str(args.warmup_updates)
    if args.sync_transfer_start:
        os.environ["AWEX_PROFILE_SYNC_START"] = "1"

    device_groups = _inference_device_groups(args)
    processes = []
    try:
        processes = [
            _start_vllm_server(args, engine_rank, device_groups[engine_rank])
            for engine_rank in range(args.num_engines)
        ]
        with ThreadPoolExecutor(max_workers=args.num_engines) as executor:
            futures = [
                executor.submit(
                    _wait_for_health,
                    processes[engine_rank],
                    args.client_host,
                    args.port + engine_rank,
                    args.startup_timeout,
                )
                for engine_rank in range(args.num_engines)
            ]
            for future in futures:
                future.result()

        with ThreadPoolExecutor(max_workers=args.num_engines) as executor:
            list(
                executor.map(
                    lambda rank: _init_engine(args, rank),
                    range(args.num_engines),
                )
            )

        for update_index in range(args.num_updates):
            step_id = update_index - 1
            logger.info(
                "========== Inference reader step %s (%s/%s) ==========",
                step_id,
                update_index + 1,
                args.num_updates,
            )
            with ThreadPoolExecutor(max_workers=args.num_engines) as executor:
                list(
                    executor.map(
                        lambda rank: _update_engine(args, rank, step_id),
                        range(args.num_engines),
                    )
                )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the inference-only side of an Awex vLLM weight exchange"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--meta-server-addr", required=True)
    parser.add_argument(
        "--comm-backend",
        default="nccl_device_v2",
        help="Weight exchange backend used by the reader and writer.",
    )
    parser.add_argument("--vllm-tp-size", type=_positive_int, default=1)
    parser.add_argument("--num-engines", type=_positive_int, default=1)
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=_unit_interval_float,
        default=0.9,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--client-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-updates", type=_positive_int, default=1)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--startup-timeout", type=_positive_int, default=300)
    parser.add_argument("--update-timeout", type=_positive_int, default=10000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--sync-transfer-start", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.warmup_updates < 0 or args.warmup_updates >= args.num_updates:
        parser.error("--warmup-updates must be in [0, --num-updates)")
    main(args)

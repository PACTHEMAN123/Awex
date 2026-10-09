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

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
import torch.distributed as dist
from shardstream import logging
from shardstream._utils import device as device_util
from shardstream._utils.profile import emit_profile, profile_phase
from shardstream.control.store import start_meta_server, stop_meta_server
from shardstream.integrations.publication import (
    create_publication_mechanism,
    publication_mechanism_names,
)

from shardstream_benchmarks.parallel import resolve_megatron_parallelism

logger = logging.getLogger(__name__)

# NOTE: vLLM plugin discovery requires shardstream to be installed (e.g., editable install)
# so that the `vllm.general_plugins` entry point is discoverable.

enable_debug_mode = False

DEFAULT_VLLM_TP_SIZE = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _inference_endpoint(value: str) -> tuple[int, str, int]:
    try:
        rank_text, host, port_text = value.split(",", 2)
        engine_rank = int(rank_text)
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must have the form ENGINE_RANK,HOST,PORT"
        ) from exc
    if engine_rank < 0:
        raise argparse.ArgumentTypeError("engine rank must be non-negative")
    if not host:
        raise argparse.ArgumentTypeError("host must not be empty")
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return engine_rank, host, port


vllm_inference_config = {
    "model_path": "/home/model/Qwen3-0.6B",
    "tp_size": DEFAULT_VLLM_TP_SIZE,
    "pp_size": 1,
    "dp_size": 1,
    "ep_size": 1,
    "num_engines": 1,
    "engine_rank": 0,
    "comm_backend": "transport",
    "enable_debug_mode": enable_debug_mode,
}


class MultiVLLMWeightsExchangeIT:
    """
    Megatron ranks use the first WORLD_SIZE visible GPUs. The vLLM child
    processes, started only by training rank 0, each use a disjoint inference
    TP group from the remaining GPUs.

    Key rule: Do NOT rely on changing os.environ["CUDA_VISIBLE_DEVICES"] in the same process
              after torch.cuda has been touched. Instead:
      - Each training process is pinned by LOCAL_RANK.
      - The child process receives its own CUDA_VISIBLE_DEVICES list.

    Launch with one torchrun process per Megatron rank. WORLD_SIZE must satisfy
    both the dense TP and expert TP x EP parallel layouts.
    """

    def __init__(
        self,
        inference_config=None,
        comm_backend=None,
        train_tp_size=1,
        train_pp_size=1,
        train_cp_size=1,
        train_ep_size=1,
        train_expert_tp_size=None,
        use_mbridge=False,
        host="127.0.0.1",
        port=8000,
        remote_inference=False,
        meta_server_host="",
        meta_server_port=0,
        publication_store_host="",
        validate=False,
        dump_weights_list_for_validation=None,
        dump_weights_dir_for_validation=None,
        publication_mechanism="shardstream",
        publication_bucket_mb=256,
        publication_timeout_seconds=1800,
        inference_endpoints=None,
    ):
        self.comm_backend = comm_backend
        self.publication_mechanism_name = publication_mechanism
        self.device_backend = device_util.get_device_type()
        self.train_parallelism = resolve_megatron_parallelism(
            tp_size=train_tp_size,
            pp_size=train_pp_size,
            cp_size=train_cp_size,
            ep_size=train_ep_size,
            expert_tp_size=train_expert_tp_size,
        )
        self.train_tp_size = self.train_parallelism.tp_size
        self.train_pp_size = self.train_parallelism.pp_size
        self.train_cp_size = self.train_parallelism.cp_size
        self.train_ep_size = self.train_parallelism.ep_size
        self.train_expert_tp_size = self.train_parallelism.expert_tp_size
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", str(self.world_size))
        )
        self.is_driver = self.rank == 0
        self.train_parallelism.validate_world_size(self.world_size)
        if (
            publication_mechanism == "shardstream"
            and self.world_size > 1
            and comm_backend == "file"
        ):
            raise RuntimeError("Multi-rank training requires the NCCL or HCCL backend.")

        self.meta_server_addr = None
        self.inference_config = inference_config or copy.deepcopy(vllm_inference_config)
        self.inference_config["comm_backend"] = comm_backend
        self.host = host
        self.port = port
        self.remote_inference = remote_inference
        self.meta_server_host = meta_server_host
        self.meta_server_port = meta_server_port
        self.publication_store_host = publication_store_host
        self.use_mbridge = use_mbridge
        self.validate = validate
        self.dump_weights_list_for_validation = dump_weights_list_for_validation or []
        self.dump_weights_dir_for_validation = dump_weights_dir_for_validation
        self.inference_endpoints = list(inference_endpoints or [])

        self.publication = create_publication_mechanism(
            publication_mechanism,
            self,
            bucket_size=publication_bucket_mb << 20,
            timeout_seconds=publication_timeout_seconds,
        )
        self.train_config = {
            "comm_backend": self.publication.training_engine_backend,
            "enable_debug_mode": enable_debug_mode,
        }

        self.vllm_visible_devices, self.megatron_device = self._select_devices()

        self.megatron_engine = None
        self.mcore_bridge = None
        self.vllm_processes = []

    def _select_devices(self):
        inference_tp = self.inference_config["tp_size"]
        num_engines = self.inference_config["num_engines"]
        total_inference_gpus = inference_tp * num_engines

        visible_env = device_util.visible_devices_env_value().strip()
        if visible_env:
            # Example: "0,1,2" -> [0,1,2] (physical ids as provided by user)
            visible_devices = [
                int(x) for x in visible_env.split(",") if x.strip() != ""
            ]
        else:
            # Fallback: use torch to detect. (May touch CUDA, but that's OK with set_device below.)
            visible_devices = list(range(device_util.device_count()))

        inference_gpus = (
            total_inference_gpus if self.is_driver and not self.remote_inference else 0
        )
        need = self.local_world_size + inference_gpus
        if len(visible_devices) < need:
            raise RuntimeError(
                f"Need at least {need} visible devices ({self.local_world_size} "
                f"local Megatron ranks + {inference_gpus} for {num_engines} "
                f"vLLM engines at TP{inference_tp}). "
                f"Found {len(visible_devices)} via visible devices env='{visible_env or '(unset)'}'."
            )
        if not 0 <= self.local_rank < self.local_world_size:
            raise RuntimeError(
                f"LOCAL_RANK ({self.local_rank}) must be in "
                f"[0, {self.local_world_size})."
            )

        megatron_device = visible_devices[self.local_rank]
        vllm_devices = (
            visible_devices[
                self.local_world_size : self.local_world_size + total_inference_gpus
            ]
            if self.is_driver and not self.remote_inference
            else []
        )
        return vllm_devices, megatron_device

    def initialize(self):
        if self.publication.uses_shardstream_meta_server:
            self._start_meta_server()
        self._init_distributed()
        if self.publication.uses_shardstream_meta_server:
            self._share_meta_server_address()
        self._init_megatron_engine()
        self.publication.initialize_training()
        if self.is_driver:
            if not self.remote_inference:
                self._start_vllm_server()
            self._wait_for_all_health()
            self.publication.initialize_driver()
        self._training_barrier()

    def destroy(self):
        self._training_barrier()
        if self.megatron_engine is not None:
            self.megatron_engine.close()
        self._training_barrier()
        if self.is_driver:
            self.publication.close()
            for process in self.vllm_processes:
                if process.poll() is None:
                    process.terminate()
            for process in self.vllm_processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        if self.is_driver and self.meta_server_addr is not None:
            stop_meta_server()
        self._training_barrier()

    def _init_distributed(self):
        if self.world_size == 1:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_PORT", "17443")
            os.environ.setdefault("MASTER_ADDR", "localhost")
        default_socket_ifname = (
            "eth0" if self.world_size > self.local_world_size else "lo"
        )
        os.environ.setdefault("GLOO_SOCKET_IFNAME", default_socket_ifname)

        device_util.set_device(self.local_rank)
        if not dist.is_initialized():
            backend = "hccl" if self.device_backend == "npu" else "nccl"
            dist.init_process_group(backend)

        logger.info(
            "Megatron rank %s/%s uses physical device id=%s (logical device %s)",
            self.rank,
            self.world_size,
            self.megatron_device,
            self.local_rank,
        )
        logger.info(
            "Training-process visible devices env=%s",
            device_util.visible_devices_env_value() or "(unset)",
        )

    def _training_barrier(self):
        if self.world_size <= 1 or not dist.is_initialized():
            return
        if self.device_backend == "cuda":
            dist.barrier(device_ids=[device_util.current_device()])
        else:
            dist.barrier()

    def _start_meta_server(self):
        if self.is_driver:
            ip, port = start_meta_server(
                host=self.meta_server_host, port=self.meta_server_port
            )
            self.meta_server_addr = f"{ip}:{port}"

    def _share_meta_server_address(self):
        addresses = [self.meta_server_addr]
        if self.world_size > 1:
            dist.broadcast_object_list(addresses, src=0)

        self.meta_server_addr = addresses[0]
        self.inference_config["meta_server_addr"] = self.meta_server_addr
        self.train_config["meta_server_addr"] = self.meta_server_addr

    def publication_endpoints(self):
        if self.inference_endpoints:
            return [
                endpoint
                for endpoint in self.inference_endpoints
                if not getattr(self, "elastic_enabled", False)
                or endpoint[0] < self.inference_config["num_engines"]
            ]
        return [
            (engine_rank, self.host, self.port + engine_rank)
            for engine_rank in range(self.inference_config["num_engines"])
        ]

    def _start_vllm_server(self):
        visible_env = device_util.visible_devices_env_names()[0]
        inference_tp = self.inference_config["tp_size"]
        num_engines = self.inference_config["num_engines"]
        for engine_rank in range(num_engines):
            env = os.environ.copy()
            start = engine_rank * inference_tp
            devices = self.vllm_visible_devices[start : start + inference_tp]
            env[visible_env] = ",".join(map(str, devices))
            env.setdefault("SHARDSTREAM_DEVICE_TYPE", device_util.get_device_type())
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

            cmd = [
                sys.executable,
                "-m",
                "shardstream.integrations.vllm.serve",
                "--model",
                self.inference_config["model_path"],
                "--host",
                self.host,
                "--port",
                str(self.port + engine_rank),
                "--tensor-parallel-size",
                str(inference_tp),
                "--pipeline-parallel-size",
                str(self.inference_config["pp_size"]),
                "--disable-log-requests",
                "--enforce-eager",
            ]
            if self.inference_config.get("enable_expert_parallel"):
                cmd.append("--enable-expert-parallel")
            from shardstream_benchmarks.fp8_config import fp8_server_args

            cmd.extend(fp8_server_args())
            gpu_memory_utilization = self.inference_config.get("gpu_memory_utilization")
            if gpu_memory_utilization is not None:
                cmd.extend(
                    [
                        "--gpu-memory-utilization",
                        str(gpu_memory_utilization),
                    ]
                )
            logger.info(
                "Starting vLLM engine %s/%s: %s",
                engine_rank,
                num_engines,
                " ".join(cmd),
            )
            logger.info(
                "vLLM engine %s subprocess %s=%s",
                engine_rank,
                visible_env,
                env.get(visible_env, ""),
            )
            self.vllm_processes.append(subprocess.Popen(cmd, env=env))

    def _wait_for_all_health(self):
        endpoints = self.publication_endpoints()
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [
                executor.submit(self._wait_for_health, endpoint)
                for endpoint in endpoints
            ]
            for future in futures:
                future.result()

    def _wait_for_health(self, endpoint, timeout=180):
        engine_rank, host, port = endpoint
        url = f"http://{host}:{port}/health"
        start = time.time()
        while time.time() - start < timeout:
            process = (
                self.vllm_processes[engine_rank]
                if engine_rank < len(self.vllm_processes)
                else None
            )
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"vLLM engine {engine_rank} exited with code {process.returncode}."
                )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(1)
        raise RuntimeError(f"vLLM engine {engine_rank} failed to start within timeout.")

    def _init_megatron_engine(self):
        self.train_config["tensor_model_parallel_size"] = self.train_tp_size
        self.train_config["pipeline_model_parallel_size"] = self.train_pp_size
        self.train_config["context_parallel_size"] = self.train_cp_size
        self.train_config["expert_model_parallel_size"] = self.train_ep_size
        self.train_config["expert_tensor_parallel_size"] = self.train_expert_tp_size

        try:
            logger.info(
                "Megatron pinned device=%s:%s name=%s",
                device_util.get_device_type(),
                device_util.current_device(),
                device_util.get_device_name(device_util.current_device()),
            )
        except Exception:
            pass

        self.mcore_model, self.mcore_hf_config = self.setup_megatron()
        from shardstream.integrations.engine.megatron import MegatronEngine

        self.megatron_engine = MegatronEngine(
            self.train_config, self.mcore_hf_config, self.mcore_model
        )
        self.megatron_engine.publication_bridge = self.mcore_bridge
        self.megatron_engine.initialize()
        logger.info("Megatron backend initialized")

    def setup_megatron(self):
        # Ensure MindSpeed patches are applied before importing Megatron when on NPU.
        from shardstream.integrations.sharding.mindspeed import ensure_mindspeed_patched

        ensure_mindspeed_patched("weights_exchange_vllm_it")

        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        from shardstream_benchmarks.model_loader import megatron_model_from_hf

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.train_tp_size,
            pipeline_model_parallel_size=self.train_pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=self.train_cp_size,
            expert_model_parallel_size=self.train_ep_size,
            expert_tensor_parallel_size=self.train_expert_tp_size,
        )

        try:
            model_parallel_cuda_manual_seed(0)
        except Exception as exc:
            if device_util.get_device_type() != "npu":
                raise
            logger.warning(
                "model_parallel_cuda_manual_seed failed on NPU (%s); "
                "falling back to torch.npu.manual_seed.",
                exc,
            )
            if getattr(torch, "npu", None) is not None:
                torch.npu.manual_seed(0)

        loaded = megatron_model_from_hf(
            model_path=self.inference_config["model_path"],
            use_mbridge=self.use_mbridge,
            return_bridge=(
                self.publication_mechanism_name
                in ("verl_nccl_broadcast", "verl_native_nccl")
            ),
        )
        if len(loaded) == 3:
            model, hf_config, self.mcore_bridge = loaded
        else:
            model, hf_config = loaded
        return model[0], hf_config

    def exchange_weights(self):
        if self.megatron_engine is None:
            raise RuntimeError("Megatron backend not initialized")

        version = int(self.megatron_engine.global_step)
        if getattr(self, "elastic_model_profile", False):
            with torch.no_grad():
                markers = [
                    p
                    for name, p in self.mcore_model.named_parameters()
                    if name.endswith("final_layernorm.weight")
                ]
                if len(markers) != 1:
                    raise RuntimeError(
                        "Expected one Megatron final norm version marker (PP1)"
                    )
                markers[0].fill_(1 + (version + 2) / 64)
            if self.is_driver:
                self.model_profile("clear", version)
            self._training_barrier()

        end_to_end_start = time.perf_counter()
        self.publication.publish()
        publication_ms = (time.perf_counter() - end_to_end_start) * 1000
        logger.info("Update weights finished")
        if self.is_driver:
            step_id = int(self.megatron_engine.global_step)
            emit_profile(
                logger,
                event="end_to_end_update",
                role="driver",
                backend=self.publication_mechanism_name,
                comm_backend=self.comm_backend,
                phase=profile_phase(step_id),
                step_id=step_id,
                rank=int(self.rank),
                end_to_end_update_time_ms=(time.perf_counter() - end_to_end_start)
                * 1000.0,
            )
        if getattr(self, "elastic_model_profile", False) and self.is_driver:
            verified = self.model_profile("verify", version)
            self.elastic_records.append(
                {
                    "event": "publication",
                    "version": version,
                    "num_engines": self.inference_config["num_engines"],
                    "ranks": verified,
                    "end_to_end_update_ms": publication_ms,
                }
            )

    def endpoint_command(self, route, payload, endpoints=None):
        endpoints = self.publication_endpoints() if endpoints is None else endpoints
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [
                executor.submit(
                    requests.post,
                    f"http://{host}:{port}/{route}",
                    json=payload,
                    timeout=600,
                )
                for _, host, port in endpoints
            ]
            results = {}
            for endpoint, future in zip(endpoints, futures):
                response = future.result()
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Engine {endpoint[0]} {route} failed: {response.text}"
                    )
                results[str(endpoint[0])] = response.json()["ranks"]
            return results

    def model_profile(self, operation, version, endpoints=None):
        return self.endpoint_command(
            "areal_shardstream_model_profile",
            {"operation": operation, "version": version},
            endpoints,
        )

    def join_rollout_models(self, epoch, target):
        from shardstream.profiling.compute import (
            BackgroundBF16Compute,
        )

        version = int(self.megatron_engine.global_step)
        old_endpoints = self.publication_endpoints()
        started_ns = time.time_ns()
        weight = next(p for p in self.mcore_model.parameters() if p.ndim == 2)
        compute = BackgroundBF16Compute(weight[:1024, :1024], batch=512, repeats=256)
        compute.start(300)
        inference_compute = {}
        try:
            if self.is_driver:
                self.model_profile("compute_start", version, old_endpoints)
                client = self.megatron_engine.weights_exchange_writer.meta_server_client
                launch_start_ns = time.time_ns()
                client.put_object(f"elastic/launch/{epoch}", {"num_engines": target})
                client.get_object(f"elastic/launched/{epoch}", timeout=600)
                previous = self.inference_config["num_engines"]
                self.inference_config["num_engines"] = target
                new_endpoints = [
                    e for e in self.publication_endpoints() if e[0] >= previous
                ]
                for endpoint in new_endpoints:
                    self._wait_for_health(endpoint, timeout=600)
                    self.publication._initialize_endpoint(endpoint)
            self._training_barrier()
            specification = {"epoch": epoch, "num_engines": target}
            prepare_start_ns = time.time_ns()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = (
                    executor.submit(
                        self.endpoint_command,
                        "areal_shardstream_prepare_membership",
                        specification,
                    )
                    if self.is_driver
                    else None
                )
                prepared = (
                    self.megatron_engine.weights_exchange_writer.prepare_membership(
                        specification
                    )
                )
                if future is not None:
                    inference_prepared = future.result()
            self._training_barrier()
            ready_ns = time.time_ns()
            if self.is_driver:
                # Old engines must still hold their last full snapshot here.
                self.model_profile("verify", version, old_endpoints)
                inference_compute = self.model_profile("compute_stop", version)
        finally:
            compute_events = compute.stop(300)
        training_result = {"prepare": prepared, "compute_events": compute_events}
        gathered = [None] * self.world_size if self.is_driver else None
        if self.world_size > 1:
            dist.gather_object(training_result, gathered, dst=0)
        else:
            gathered = [training_result]
        if self.is_driver:
            record = {
                "event": "join",
                "epoch": epoch,
                "num_engines": target,
                "start_ns": started_ns,
                "launch_start_ns": launch_start_ns,
                "prepare_start_ns": prepare_start_ns,
                "ready_ns": ready_ns,
                "join_ms": (ready_ns - started_ns) / 1e6,
                "launch_and_startup_ms": (prepare_start_ns - launch_start_ns) / 1e6,
                "prepare_ms": (ready_ns - prepare_start_ns) / 1e6,
                "training_ranks": gathered,
                "inference_ranks": inference_prepared,
                "inference_compute": inference_compute,
            }
            self.elastic_records.append(record)
            logger.info(
                "Elastic model join ready: epoch=%s engines=%s join_ms=%.3f prepare_ms=%.3f",
                epoch,
                target,
                record["join_ms"],
                record["prepare_ms"],
            )
        self._training_barrier()


def main(args):
    os.environ.setdefault("NCCL_DEBUG", "WARNING")
    if getattr(args, "nccl_device_chunk_mb", None) is not None:
        os.environ["SHARDSTREAM_CHUNK_BYTES"] = str(
            args.nccl_device_chunk_mb * 1024 * 1024
        )
    if args.profile:
        os.environ["SHARDSTREAM_PROFILE"] = "1"
        os.environ["SHARDSTREAM_PROFILE_WARMUP_UPDATES"] = str(args.warmup_updates)
    if args.sync_transfer_start:
        os.environ["SHARDSTREAM_PROFILE_SYNC_START"] = "1"
    comm_backend = args.comm_backend
    inference_config = copy.deepcopy(vllm_inference_config)
    if args.model_path:
        inference_config["model_path"] = args.model_path
    inference_config["tp_size"] = args.vllm_tp_size
    inference_config["num_engines"] = args.num_engines
    inference_config["enable_expert_parallel"] = args.vllm_enable_expert_parallel
    inference_config["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization

    weights_exchange_it = MultiVLLMWeightsExchangeIT(
        inference_config=inference_config,
        comm_backend=comm_backend,
        train_tp_size=args.train_tp_size,
        train_pp_size=args.train_pp_size,
        train_cp_size=args.train_cp_size,
        train_ep_size=args.train_ep_size,
        train_expert_tp_size=args.train_expert_tp_size,
        use_mbridge=args.use_mbridge,
        host=args.host,
        port=args.port,
        remote_inference=args.remote_inference,
        meta_server_host=args.meta_server_host,
        meta_server_port=args.meta_server_port,
        publication_store_host=args.publication_store_host,
        validate=args.validate,
        dump_weights_list_for_validation=args.dump_weights_list_for_validation,
        dump_weights_dir_for_validation=args.dump_weights_dir_for_validation,
        publication_mechanism=args.publication_mechanism,
        publication_bucket_mb=args.publication_bucket_mb,
        publication_timeout_seconds=args.publication_timeout_seconds,
        inference_endpoints=args.inference_endpoint,
    )
    weights_exchange_it.elastic_enabled = bool(args.join_after)
    weights_exchange_it.elastic_model_profile = args.elastic_model_profile
    weights_exchange_it.elastic_records = []
    joins = dict(zip(args.join_after, args.target_engines))
    epoch = 0

    try:
        weights_exchange_it.initialize()
        for update_index in range(args.num_updates):
            step_id = update_index - 1
            weights_exchange_it.megatron_engine.set_global_step(step_id)
            logger.info(
                "========== Test weights exchange step %s (%s/%s) ==========",
                step_id,
                update_index + 1,
                args.num_updates,
            )
            weights_exchange_it.exchange_weights()
            if update_index + 1 in joins:
                epoch += 1
                weights_exchange_it.join_rollout_models(epoch, joins[update_index + 1])
        if weights_exchange_it.is_driver and args.elastic_output:
            Path(args.elastic_output).write_text(
                json.dumps(
                    {
                        "passed": True,
                        "model_path": args.model_path,
                        "dtype": "bfloat16",
                        "real_model_weights": True,
                        "train_world_size": weights_exchange_it.world_size,
                        "inference_tp_size": args.vllm_tp_size,
                        "records": weights_exchange_it.elastic_records,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        weights_exchange_it.destroy()
        _destroy_process_group(timeout=5)


def _destroy_process_group(timeout: float = 5.0) -> None:
    if not dist.is_initialized():
        return

    error = {}

    def _destroy():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            error["exc"] = exc

    thread = threading.Thread(target=_destroy, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.warning("Timed out destroying process group; continuing.")
    elif error:
        logger.warning("Failed to destroy process group: %s", error["exc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ShardStream real-model weight exchange")
    parser.add_argument("--join-after", type=_positive_int, nargs="*", default=[])
    parser.add_argument("--target-engines", type=_positive_int, nargs="*", default=[])
    parser.add_argument(
        "--elastic-model-profile",
        action="store_true",
        help="Exact full-model GPU validation and BF16 compute during joins (extra reference model storage)",
    )
    parser.add_argument("--elastic-output")
    parser.add_argument(
        "-b",
        "--comm_backend",
        default="transport",
        choices=["transport"],
        help="ShardStream CUDA transport.",
    )
    parser.add_argument(
        "--publication-mechanism",
        choices=publication_mechanism_names(),
        default="shardstream",
        help="Weight publication mechanism exercised by the harness.",
    )
    parser.add_argument(
        "--publication-bucket-mb",
        type=_positive_int,
        default=256,
        metavar="MiB",
        help="Per-buffer size for bucketed publication mechanisms.",
    )
    parser.add_argument(
        "--publication-timeout-seconds",
        type=_positive_int,
        default=1800,
        metavar="SECONDS",
        help="Initialization and update timeout for publication mechanisms.",
    )
    parser.add_argument(
        "--model-path",
        default=vllm_inference_config["model_path"],
        help="HF model path used by Megatron and the vLLM server.",
    )
    parser.add_argument(
        "--train-tp-size",
        type=_positive_int,
        default=1,
        metavar="N",
        help=(
            "Megatron dense tensor-parallel size. Torchrun WORLD_SIZE must "
            "satisfy both the dense and expert parallel layouts."
        ),
    )
    parser.add_argument(
        "--train-pp-size",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Megatron pipeline-parallel size.",
    )
    parser.add_argument(
        "--train-cp-size",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Megatron context-parallel size.",
    )
    parser.add_argument(
        "--train-ep-size",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Megatron expert-parallel size.",
    )
    parser.add_argument(
        "--train-expert-tp-size",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Megatron expert tensor-parallel size (default: train TP size, "
            "matching Megatron Core)."
        ),
    )
    parser.add_argument(
        "--vllm-tp-size",
        type=_positive_int,
        default=vllm_inference_config["tp_size"],
        metavar="N",
        help="Tensor-parallel size of each vLLM engine.",
    )
    parser.add_argument(
        "--num-engines",
        type=_positive_int,
        default=2,
        metavar="N",
        help="Number of independent vLLM engines to update.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=_unit_interval_float,
        default=None,
        metavar="FRACTION",
        help=(
            "Optional vLLM per-GPU memory utilization fraction; by default "
            "vLLM uses its own setting."
        ),
    )
    parser.add_argument(
        "--vllm-enable-expert-parallel",
        action="store_true",
        help="Enable vLLM expert parallel mode for MoE inference.",
    )
    parser.add_argument(
        "--nccl-device-chunk-mb",
        type=_non_negative_int,
        default=None,
        metavar="MiB",
        help=(
            "Fixed nccl_device task chunk size in MiB (default: 4; "
            "0 restores one task per tensor)."
        ),
    )
    parser.add_argument(
        "--num-updates",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Number of consecutive weight updates to execute.",
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=0,
        metavar="N",
        help="Number of initial updates marked as warmup in profile output.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit structured SHARDSTREAM_PROFILE timing records.",
    )
    parser.add_argument(
        "--sync-transfer-start",
        action="store_true",
        help=(
            "Synchronize all reader and writer ranks after transfer preparation "
            "and immediately before timed backend execution."
        ),
    )
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Device backend to use (auto/cuda/npu/cpu).",
    )
    parser.add_argument(
        "--use-mbridge",
        action="store_true",
        help="Load HF weights into Megatron via mbridge (skip DCP conversion).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--inference-endpoint",
        action="append",
        type=_inference_endpoint,
        default=[],
        metavar="ENGINE_RANK,HOST,PORT",
        help=(
            "Remote inference endpoint. Repeat once per engine to distribute "
            "engines across multiple hosts."
        ),
    )
    parser.add_argument(
        "--remote-inference",
        action="store_true",
        help=(
            "Connect to already-running vLLM engines on --host instead of "
            "launching locally."
        ),
    )
    parser.add_argument(
        "--meta-server-host",
        default="",
        help="Address on which the training driver exposes the Awex meta server.",
    )
    parser.add_argument(
        "--meta-server-port",
        type=int,
        default=0,
        help="Fixed Awex meta-server port (0 selects a free port).",
    )
    parser.add_argument(
        "--publication-store-host",
        default="",
        help="Training-driver address reachable by remote NCCL bucket receivers.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable weights validation (NCCL or file backend).",
    )
    parser.add_argument(
        "--dump-weights-list-for-validation",
        default="",
        help="Comma-separated parameter names to dump during validation.",
    )
    parser.add_argument(
        "--dump-weights-dir-for-validation",
        default="",
        help="Directory to dump validation tensors.",
    )
    args = parser.parse_args()
    targets = [args.num_engines, *args.target_engines]
    if len(args.join_after) != len(args.target_engines) or any(
        a >= b for a, b in zip(targets, targets[1:])
    ):
        parser.error("Join boundaries require strictly increasing engine counts")
    if sorted(set(args.join_after)) != args.join_after or any(
        not 1 <= n < args.num_updates for n in args.join_after
    ):
        parser.error("Join boundaries must increase and leave a following update")
    if args.join_after and (
        not args.remote_inference
        or args.publication_mechanism != "shardstream"
        or args.comm_backend != "transport"
    ):
        parser.error("Model joins require remote inference and Awex device v2")
    if args.elastic_model_profile and (
        args.train_pp_size != 1 or args.comm_backend != "transport"
    ):
        parser.error("Full model profiling currently requires PP1 and device v2")
    if args.warmup_updates < 0 or args.warmup_updates >= args.num_updates:
        parser.error("--warmup-updates must be in [0, --num-updates)")
    if args.inference_endpoint:
        if not args.remote_inference:
            parser.error("--inference-endpoint requires --remote-inference")
        endpoint_ranks = sorted(endpoint[0] for endpoint in args.inference_endpoint)
        if endpoint_ranks != list(range(targets[-1])):
            parser.error(
                "--inference-endpoint ranks must cover every engine rank in "
                "[0, --num-engines) exactly once"
            )
    if args.device_backend and args.device_backend != "auto":
        os.environ["SHARDSTREAM_DEVICE_TYPE"] = args.device_backend
    if device_util.get_device_type() == "npu" and args.comm_backend == "nccl":
        logger.warning("Switching comm_backend from nccl to hccl for NPU backend.")
        args.comm_backend = "hccl"
    if args.dump_weights_list_for_validation:
        args.dump_weights_list_for_validation = [
            name.strip()
            for name in args.dump_weights_list_for_validation.split(",")
            if name.strip()
        ]
    else:
        args.dump_weights_list_for_validation = []
    args.dump_weights_dir_for_validation = args.dump_weights_dir_for_validation or None
    main(args)

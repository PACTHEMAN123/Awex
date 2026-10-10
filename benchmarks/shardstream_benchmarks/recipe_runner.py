"""Run pinned recipes on separate Ray training and vLLM hosts."""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from shardstream_benchmarks.recipes import load_recipes


class NodeProcesses:
    """Own only the process groups launched by this recipe invocation."""

    def __init__(
        self,
        source,
        runtime,
        stage,
        run_dir,
        gpu_count=None,
        placement="packed",
        replica_tp=1,
        hca_span=1,
    ):
        self.source = Path(source)
        self.runtime = Path(runtime)
        self.stage = Path(stage)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.processes = {}
        self.gpus = None
        self.affinity = None
        if placement == "nic-spread":
            from shardstream_benchmarks.placement import plan_nic_spread

            self.gpus, self.affinity = plan_nic_spread(
                os.environ["CUDA_VISIBLE_DEVICES"].split(","),
                gpu_count,
                replica_tp,
                hca_span,
            )
            (self.run_dir / "placement.json").write_text(
                json.dumps({"gpus": self.gpus, "affinity": self.affinity}, indent=2)
            )

    def placement(self):
        return {"gpus": self.gpus, "affinity": self.affinity}

    def launch(self, name, arguments, environment):
        env = os.environ.copy()
        env.update(environment)
        if self.gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpus)
            env["SHARDSTREAM_RANK_AFFINITY"] = json.dumps(self.affinity)
            env["SHARDSTREAM_HCA_POLICY"] = "topology"
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        gpu_indices = env.pop("SHARDSTREAM_RECIPE_GPU_INDICES", None)
        if gpu_indices is not None:
            assigned = env["CUDA_VISIBLE_DEVICES"].split(",")
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                assigned[int(index)] for index in gpu_indices.split(",")
            )
        env["PYTHONPATH"] = str(self.stage)
        env["SHARDSTREAM_RECIPE_REPOSITORY"] = str(self.source)
        env["PATH"] = f"{self.runtime}/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
        libs = [
            "/usr/local/cuda/lib64",
            str(self.stage / "lib64"),
            str(self.runtime / "lib"),
            str(self.runtime / "lib/python3.12/site-packages/nvidia/cu13/lib"),
            str(self.runtime / "lib/python3.12/site-packages/torch/lib"),
        ]
        env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
        env.update(
            {
                "VLLM_PLUGINS": "shardstream_adapter",
                "SHARDSTREAM_NCCL_LIB": "/usr/local/cuda/lib64",
                "LD_PRELOAD": "/usr/local/cuda/lib64/libnccl.so.2",
                "NCCL_SOCKET_IFNAME": "eth0",
                "GLOO_SOCKET_IFNAME": "eth0",
                "NCCL_IB_DISABLE": "0",
                "NCCL_IB_GID_INDEX": "3",
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,NET,ENV",
                "NCCL_CUMEM_ENABLE": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "SHARDSTREAM_RING_BROADCAST": "1",
                "SHARDSTREAM_RING_SWIZZLE": "1",
                "SHARDSTREAM_FP8_BLOCKWISE": "0",
                "SHARDSTREAM_PROFILE": "1",
                "SHARDSTREAM_PROFILE_WARMUP_UPDATES": "1",
                "SHARDSTREAM_PROFILE_SYNC_START": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [str(self.runtime / "bin/python"), *arguments]
        (self.run_dir / f"{name}-command.json").write_text(
            json.dumps(command, indent=2)
        )
        with (self.run_dir / f"{name}.log").open("wb") as log:
            self.processes[name] = subprocess.Popen(
                command,
                env=env,
                cwd=self.source,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return self.processes[name].pid

    def status(self):
        return {name: process.poll() for name, process in self.processes.items()}

    def stop(self):
        for process in self.processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)


def request(url, payload=None, timeout=120, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = dict(headers or {})
    if data is not None:
        headers["Content-Type"] = "application/json"
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=timeout
    ) as response:
        return json.load(response) if payload is not None else response.status


def generation(endpoint, model_path):
    content = "Compute 2 + 3. Answer briefly."
    response = request(
        endpoint + "/v1/chat/completions",
        {
            "model": model_path,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 64,
        },
        # Keep DP8 before/after requests on the same model replica. The vLLM
        # load balancer may otherwise route them to different DP ranks.
        headers={"X-data-parallel-rank": "0"},
    )
    if (
        not response.get("choices")
        or response.get("usage", {}).get("completion_tokens", 0) < 1
    ):
        raise RuntimeError("Generation did not produce tokens")
    # Reasoning models can spend the short smoke budget entirely in reasoning.
    return {
        "message": response["choices"][0]["message"],
        "usage": response.get("usage"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=tuple(load_recipes()))
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--training-hosts", nargs="+", required=True)
    parser.add_argument("--rollout-hosts", nargs="+", required=True)
    parser.add_argument(
        "--gpu-placement", choices=("packed", "nic-spread"), default="nic-spread"
    )
    parser.add_argument(
        "--rollout-replicas",
        type=int,
        help="Override replica count while retaining the recipe's per-replica TP/DP/EP",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source", default="/tmp/shardstream-model-recipes-1010")
    parser.add_argument("--runtime", default="/tmp/model-recipes-1010-runtime")
    parser.add_argument("--stage", default="/tmp/model-recipes-1010-stage")
    parser.add_argument(
        "--run-root", default="/mnt/fuse/verl-e2e/runs/model-recipes-20261010/e2e"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    # Ensure task-owned children are cleaned up when the driver is interrupted.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    pinned_recipe = load_recipes()[args.recipe]
    recipe = pinned_recipe.with_rollout_replicas(args.rollout_replicas)
    recipe.validate_placement(args.training_hosts, args.rollout_hosts)
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.ray_address)
    nodes = {node["NodeManagerAddress"]: node for node in ray.nodes() if node["Alive"]}
    for host in args.training_hosts + args.rollout_hosts:
        if host not in nodes:
            raise ValueError(f"Selected Ray host is absent: {host}")
    invocation = recipe.id + "-" + uuid.uuid4().hex[:10]
    run_dir = str(Path(args.run_root) / invocation)
    actors = []
    rollout = []
    training = []
    record = {
        "recipe": recipe.id,
        "source": recipe.source,
        "training": recipe.training,
        "rollout": recipe.rollout,
        "pinned_rollout": pinned_recipe.rollout,
        "rollout_replicas_override": args.rollout_replicas,
        "training_hosts": args.training_hosts,
        "rollout_hosts": args.rollout_hosts,
        "run_dir": run_dir,
        "passed": False,
        "generation_dp_rank": 0,
        "gpu_placement": args.gpu_placement,
        "node_placements": {},
    }
    try:
        for role, hosts, world in [
            ("training", args.training_hosts, recipe.training["world_size"]),
            ("rollout", args.rollout_hosts, recipe.rollout["world_size"]),
        ]:
            for host in hosts:
                actor = (
                    ray.remote(NodeProcesses)
                    .options(
                        num_cpus=1,
                        num_gpus=int(nodes[host]["Resources"]["GPU"])
                        if args.gpu_placement == "nic-spread"
                        else world // len(hosts),
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            nodes[host]["NodeID"], soft=False
                        ),
                    )
                    .remote(
                        args.source,
                        args.runtime,
                        args.stage,
                        run_dir,
                        world // len(hosts),
                        args.gpu_placement,
                        recipe.rollout["tp"] if role == "rollout" else 1,
                        2
                        if role == "rollout"
                        and recipe.id == "qwen3.5-9b"
                        and world == 4
                        else 1,
                    )
                )
                actors.append(actor)
                (training if role == "training" else rollout).append(actor)
                record["node_placements"][host] = ray.get(actor.placement.remote())
        endpoints = []
        instances_per_host = recipe.rollout["instances"] // len(rollout)
        if instances_per_host < 1 or recipe.rollout["instances"] % len(rollout):
            raise ValueError("Each vLLM instance must fit entirely on one rollout host")
        engine_gpus = recipe.rollout["tp"] * recipe.rollout["dp"]
        for host_rank, actor in enumerate(rollout):
            for local_engine in range(instances_per_host):
                engine_rank = host_rank * instances_per_host + local_engine
                host, port = args.rollout_hosts[host_rank], 18800 + local_engine
                devices = ",".join(
                    str(i)
                    for i in range(
                        local_engine * engine_gpus, (local_engine + 1) * engine_gpus
                    )
                )
                # Indices address the GPUs reserved for this actor, not arbitrary GPUs.
                ray.get(
                    actor.launch.remote(
                        f"rollout-{engine_rank}",
                        recipe.rollout_arguments(args.model_path, host, port),
                        {"SHARDSTREAM_RECIPE_GPU_INDICES": devices},
                    )
                )
                endpoints.append((engine_rank, host, port))
        deadline = time.monotonic() + args.timeout
        for _, host, port in endpoints:
            while True:
                status = ray.get([actor.status.remote() for actor in rollout])
                if any(
                    code is not None
                    for processes in status
                    for code in processes.values()
                ):
                    raise RuntimeError(f"vLLM exited during startup: {status}")
                try:
                    request(f"http://{host}:{port}/health", timeout=5)
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("vLLM health timeout")
                    time.sleep(3)
        urls = [f"http://{host}:{port}" for _, host, port in endpoints]
        record["generation_before"] = [generation(url, args.model_path) for url in urls]
        if recipe.rollout["dp"] > 1:
            record["generation_before_repeat"] = [
                generation(url, args.model_path) for url in urls
            ]
            if [r["message"] for r in record["generation_before"]] != [
                r["message"] for r in record["generation_before_repeat"]
            ]:
                raise RuntimeError(
                    "Generation baseline is not repeatable before weight transfer"
                )
        train_args = [
            "-m",
            "shardstream_benchmarks.model_exchange",
            "--recipe",
            recipe.id,
            "--model-path",
            args.model_path,
            "--remote-inference",
            "--validate",
            "--num-engines",
            str(recipe.rollout["instances"]),
            "--vllm-tp-size",
            str(recipe.rollout["tp"]),
            "--vllm-dp-size",
            str(recipe.rollout["dp"]),
            "--num-updates",
            "3",
            "--warmup-updates",
            "1",
            "--profile",
            "--sync-transfer-start",
            "--meta-server-host",
            args.training_hosts[0],
            "--meta-server-port",
            "18890",
        ]
        for engine, host, port in endpoints:
            train_args.extend(["--inference-endpoint", f"{engine},{host},{port}"])
        if args.rollout_replicas is not None:
            train_args.extend(["--rollout-replicas", str(args.rollout_replicas)])
        for node_rank, actor in enumerate(training):
            torchrun = [
                "-m",
                "torch.distributed.run",
                "--nnodes",
                str(len(training)),
                "--nproc-per-node",
                str(recipe.training["world_size"] // len(training)),
                "--node-rank",
                str(node_rank),
                "--rdzv-backend",
                "c10d",
                "--rdzv-endpoint",
                args.training_hosts[0] + ":18891",
                "--rdzv-id",
                invocation,
                *train_args,
            ]
            ray.get(actor.launch.remote("training", torchrun, {}))
        while True:
            states = ray.get([actor.status.remote() for actor in training])
            codes = [state["training"] for state in states]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Training/transfer failed: {codes}")
            if all(code == 0 for code in codes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Transfer timeout")
            time.sleep(3)
        record["generation_after"] = [generation(url, args.model_path) for url in urls]
        if [result["message"] for result in record["generation_before"]] != [
            result["message"] for result in record["generation_after"]
        ]:
            raise RuntimeError(
                "Deterministic generation changed after checkpoint transfer"
            )
        record["passed"] = True
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        for actor in actors:
            try:
                ray.get(actor.stop.remote(), timeout=45)
            except Exception:
                # Continue cleanup if an actor died during a failed run.
                pass
            finally:
                ray.kill(actor)
        ray.shutdown()


if __name__ == "__main__":
    main()

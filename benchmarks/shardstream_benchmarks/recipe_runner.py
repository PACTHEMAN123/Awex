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

from shardstream_benchmarks.recipes import load_recipes, rollout_host_counts


def training_launch_arguments(hosts, world_size, node_rank, invocation, train_args):
    # Metadata binds to hosts[0], so rank zero must stay on that host.
    return [
        "-m", "torch.distributed.run", "--nnodes", str(len(hosts)),
        "--nproc-per-node", str(world_size // len(hosts)),
        "--node-rank", str(node_rank), "--rdzv-backend", "static",
        "--rdzv-endpoint", hosts[0] + ":18891", "--rdzv-id", invocation,
        *train_args,
    ]


def nccl_buffer_environment(buffer_bytes=None, net_chunk_bytes=None):
    """Validate sizes against the pinned NCCL 2.31.2 signed fields."""
    environment = {}
    if buffer_bytes is not None:
        if not 0 < buffer_bytes < 2**31:
            raise ValueError("NCCL buffer size must be positive and below 2 GiB")
        environment["NCCL_BUFFSIZE"] = str(buffer_bytes)
    if net_chunk_bytes is not None:
        # NCCL silently clamps chunks to BUFFSIZE / NCCL_STEPS (8).
        if not 0 < net_chunk_bytes <= (buffer_bytes or 4 * 2**20) // 8:
            raise ValueError("NCCL net chunk must be positive and at most BUFFSIZE / 8")
        environment["NCCL_P2P_NET_CHUNKSIZE"] = str(net_chunk_bytes)
    return environment


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
        nccl_lib="/tmp/shardstream-prepare-nccl2312/nccl/lib",
    ):
        self.source = Path(source)
        self.runtime = Path(runtime)
        self.stage = Path(stage)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.processes = {}
        self.profiled = False
        self.gpus = None
        self.affinity = None
        self.nccl_lib = str(nccl_lib)
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
        extra_pythonpath = env.pop("SHARDSTREAM_BENCH_EXTRA_PYTHONPATH", "")
        if extra_pythonpath:
            env["PYTHONPATH"] += ":" + extra_pythonpath
        env["SHARDSTREAM_RECIPE_REPOSITORY"] = str(self.source)
        env["PATH"] = f"{self.runtime}/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
        libs = [
            self.nccl_lib,
            "/usr/local/cuda/lib64",
            str(self.stage / "lib64"),
            str(self.runtime / "lib"),
            str(self.runtime / "lib/python3.12/site-packages/nvidia/cu13/lib"),
            str(self.runtime / "lib/python3.12/site-packages/torch/lib"),
        ]
        extra_libs = env.pop("SHARDSTREAM_BENCH_EXTRA_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + extra_libs + ":" + env.get("LD_LIBRARY_PATH", "")
        env.update(
            {
                "VLLM_PLUGINS": "shardstream_adapter",
                "VLLM_SERVER_DEV_MODE": "1" if environment.get("SHARDSTREAM_FP8_BLOCKWISE", "0") == "1"
                else env.get("VLLM_SERVER_DEV_MODE", "0"),
                "SHARDSTREAM_NCCL_LIB": self.nccl_lib,
                "LD_PRELOAD": self.nccl_lib + "/libnccl.so.2",
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
                "SHARDSTREAM_FP8_BLOCKWISE": environment.get("SHARDSTREAM_FP8_BLOCKWISE", "0"),
                "SHARDSTREAM_FP8_BLOCK_ROWS": "128",
                "SHARDSTREAM_FP8_BLOCK_COLS": "128",
                "SHARDSTREAM_PROFILE": "1",
                "SHARDSTREAM_PROFILE_WARMUP_UPDATES": str(
                    int(environment.get("SHARDSTREAM_BENCH_WARMUP", 1))
                    + int(environment.get("SHARDSTREAM_FP8_BLOCKWISE", "0") == "1")),
                "SHARDSTREAM_PROFILE_SYNC_START": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        if environment.get("SHARDSTREAM_BENCH_NATIVE_MULTIRAIL") == "1":
            # NCCL caches environment options at its first initialization,
            # including the vLLM/Megatron groups created before publication.
            env.update(NCCL_NET_GDR_LEVEL="SYS", NCCL_MIN_NCHANNELS="32",
                       NCCL_MAX_NCHANNELS="64", NCCL_CROSS_NIC="1",
                       NCCL_PXN_DISABLE="1", NCCL_P2P_PXN_LEVEL="0")
            ports = sorted({row["hca"].lstrip("=") for row in self.affinity.values()})
            if len(ports) != 4:
                raise RuntimeError(f"Native baseline requires four actual HCA ports: {ports}")
            env.update(NCCL_IB_HCA="=" + ",".join(ports),
                       UCX_NET_DEVICES=",".join(ports), UCX_MAX_RMA_RAILS="4",
                       UCX_MAX_RNDV_RAILS="4", UCX_MAX_EAGER_RAILS="4",
                       UCX_RNDV_SCHEME="get_zcopy", UCX_IB_GID_INDEX="3",
                       UCX_MAX_HCA_PER_GPU="inf", UCX_SINGLE_NET_DEVICE="n")
        if environment.get("SHARDSTREAM_BENCH_AWEX_RDMA") == "1":
            import socket

            env["AWEX_MOONCAKE_HOST"] = socket.gethostbyname(socket.gethostname())
            env["MC_GID_INDEX"] = "3"
            # Full-model Awex registers thousands of native tensor slices.
            # Their P2P MR metadata exceeds Mooncake's default 1 MiB limit.
            # This changes the control-message limit, not a transfer buffer.
            env["MC_HANDSHAKE_MAX_LENGTH"] = "67108864"
            env["AWEX_MOONCAKE_DEVICE"] = ",".join(port.split(":")[0] for port in ports)
        if environment.get("SHARDSTREAM_BENCH_AWEX_ASTATE") == "1":
            import socket

            env["POD_IP"] = socket.gethostbyname(socket.gethostname())
            env["SHARDSTREAM_ASTATE_MANIFEST"] = "/tmp/figure11-1025/astate-artifact.json"
            env["TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION"] = "false"
        command = [str(self.runtime / "bin/python"), *arguments]
        if env.get("SHARDSTREAM_TIMELINE") == "1":
            self.profiled = True
            trace_dir = self.run_dir / "nsys"
            trace_dir.mkdir(exist_ok=True)
            command = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
                       "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                       "--capture-range-end=none", "--trace-fork-before-exec=true",
                       "--cuda-graph-trace=node", "--output=" + str(trace_dir / name),
                       *command]
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

    def read_json(self, name):
        if Path(name).name != name:
            raise ValueError("Expected a run-local artifact name")
        return json.loads((self.run_dir / name).read_text())

    def stop(self):
        for process in self.processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.processes.values():
            try:
                process.wait(timeout=120 if self.profiled else 15)
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


def generation(endpoint, model_path, base_model=False):
    content = "Compute 2 + 3. Answer briefly."
    response = request(
        endpoint + ("/v1/completions" if base_model else "/v1/chat/completions"),
        {
            "model": model_path,
            **({"prompt": content + "\nAnswer:"} if base_model else
               {"messages": [{"role": "user", "content": content}]}),
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
        "message": ({"role": "assistant", "content": response["choices"][0]["text"]}
                    if base_model else response["choices"][0]["message"]),
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
    parser.add_argument("--num-updates", type=int, default=3)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--publication-mechanism", default="shardstream")
    parser.add_argument("--layout")
    parser.add_argument("--publication-bucket-mb", type=int, default=256)
    parser.add_argument("--model-weight-profile", action="store_true")
    parser.add_argument("--communication-memory-profile", action="store_true")
    parser.add_argument("--fifo-layout-profile", action="store_true")
    parser.add_argument("--nsys", action="store_true", help="Capture updates 2 through 4; adds no CUDA synchronization")
    parser.add_argument("--awex-nonbatch", action="store_true")
    parser.add_argument("--nccl-buffer-bytes", type=int)
    parser.add_argument("--nccl-net-chunk-bytes", type=int)
    parser.add_argument("--weight-format", choices=("bf16", "fp8"), default="bf16",
                        help="FP8 uses fused BF16-to-FP8 transport and a full checkpoint oracle")
    parser.add_argument("--disable-prefix-caching", action="store_true",
                        help="Use the same uncached prefill for before/repeat/after generation checks")
    parser.add_argument("--extra-python-path", default="")
    parser.add_argument("--extra-library-path", default="")
    parser.add_argument("--transport-gin-connections", type=int, choices=(1, 2, 4))
    parser.add_argument("--transport-max-channels", type=int, choices=(16, 32, 64))
    parser.add_argument("--transport-gin-fifo-depth", type=int, choices=(4, 8, 16))
    parser.add_argument("--transport-compact-fifo", action="store_true")
    parser.add_argument("--dp-source-policy", choices=("replica", "block", "stream"), default="replica")
    parser.add_argument("--dp-source-block-mb", type=int, default=16)
    parser.add_argument("--dp-source-ring-order", choices=("rank", "source-node", "dp"), default="rank")
    parser.add_argument("--replica-routing", choices=("engine", "logical"), default="engine")
    parser.add_argument(
        "--nccl-lib", default="/tmp/shardstream-prepare-nccl2312/nccl/lib"
    )
    args = parser.parse_args()
    if args.awex_nonbatch and args.publication_mechanism != "awex_native_nccl":
        parser.error("--awex-nonbatch requires awex_native_nccl")
    try:
        nccl_environment = nccl_buffer_environment(args.nccl_buffer_bytes, args.nccl_net_chunk_bytes)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.warmup_updates < args.num_updates:
        parser.error("Require 0 <= warmup updates < total updates")
    if args.dp_source_block_mb <= 0:
        parser.error("DP source block size must be positive")
    if args.weight_format == "fp8" and (
        not args.model_weight_profile or args.publication_mechanism not in (
            "shardstream", "verl_native_nccl")
    ):
        parser.error("FP8 requires ShardStream or veRL NCCL and --model-weight-profile")
    transport_environment = {
        name: str(value) for name, value in (
            ("SHARDSTREAM_GIN_CONNECTIONS", args.transport_gin_connections),
            ("NCCL_GIN_NCONNECTIONS", args.transport_gin_connections),
            ("SHARDSTREAM_MAX_CHANNELS", args.transport_max_channels),
            ("SHARDSTREAM_GIN_FIFO_DEPTH", args.transport_gin_fifo_depth),
        ) if value is not None
    }
    if transport_environment and args.publication_mechanism != "shardstream":
        parser.error("Transport preparation tuning applies only to ShardStream")
    if args.transport_compact_fifo:
        if args.publication_mechanism != "shardstream":
            parser.error("Compact FIFO applies only to ShardStream")
        transport_environment["SHARDSTREAM_COMPACT_FIFO"] = "1"
    transport_environment.update(nccl_environment)
    transport_environment.update(
        SHARDSTREAM_TIMELINE="1" if args.nsys else "0",
        SHARDSTREAM_COMM_MEMORY_PROFILE="1" if args.communication_memory_profile else "0",
        SHARDSTREAM_PROFILE_FIFO_LAYOUT="1" if args.fifo_layout_profile else "0",
        SHARDSTREAM_FP8_BLOCKWISE="1" if args.weight_format == "fp8" else "0",
        SHARDSTREAM_BENCH_MODEL_PATH=args.model_path,
        SHARDSTREAM_DP_SOURCE_POLICY=args.dp_source_policy,
        SHARDSTREAM_REPLICA_ROUTING=args.replica_routing,
        SHARDSTREAM_DP_SOURCE_BLOCK_BYTES=str(args.dp_source_block_mb * 1024 * 1024),
        SHARDSTREAM_RING_SOURCE_NODE_SWIZZLE="1" if args.dp_source_ring_order == "source-node" else "0",
        SHARDSTREAM_RING_DP_ENTRY="1" if args.dp_source_ring_order == "dp" else "0",
    )

    # Ensure task-owned children are cleaned up when the driver is interrupted.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    pinned_recipe = load_recipes()[args.recipe]
    recipe = pinned_recipe.with_layout(args.layout).with_rollout_replicas(args.rollout_replicas)
    recipe.validate_placement(args.training_hosts, args.rollout_hosts)
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Ray workers do not inherit the driver's PYTHONPATH when joining a running
    # cluster. Their imports must resolve against the same pinned node-local
    # stage used by the child training and inference processes.
    ray.init(
        address=args.ray_address,
        runtime_env={"env_vars": {"PYTHONPATH": args.stage}},
    )
    # Ray installs its own SIGTERM handler during init. Restore ours so the
    # finally block can stop the exact remote process groups this run owns.
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
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
        "layout": args.layout,
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
        "prefix_caching_disabled": args.disable_prefix_caching,
        "gpu_placement": args.gpu_placement,
        "num_updates": args.num_updates,
        "warmup_updates": args.warmup_updates,
        "nccl_lib": args.nccl_lib,
        "publication_mechanism": args.publication_mechanism,
        "publication_bucket_mb": args.publication_bucket_mb,
        "model_weight_profile": args.model_weight_profile,
        "weight_format": args.weight_format,
        "transport_environment": transport_environment,
        "node_placements": {},
    }
    try:
        environment = {
            **transport_environment,
            "VERL_VLLM_FP8_QUANT_ENABLED": "1" if (
                args.weight_format == "fp8" and args.publication_mechanism == "verl_native_nccl"
            ) else "0",
            "SHARDSTREAM_BENCH_ENABLE_AWEX": "1" if args.publication_mechanism in ("awex_native_nccl", "awex_native_rdma", "awex_native_astate") else "0",
            "SHARDSTREAM_BENCH_NATIVE_MULTIRAIL": "1" if args.publication_mechanism in (
                "awex_native_nccl", "awex_native_rdma", "awex_native_astate", "verl_native_nccl", "verl_native_nixl", "verl_native_mooncake"
            ) else "0",
            "SHARDSTREAM_BENCH_EXTRA_PYTHONPATH": args.extra_python_path,
            "SHARDSTREAM_BENCH_EXTRA_LIBRARY_PATH": args.extra_library_path,
            "SHARDSTREAM_BENCH_WARMUP": str(args.warmup_updates),
            "SHARDSTREAM_BENCH_RAY_ADDRESS": args.ray_address,
        }
        if args.publication_mechanism == "awex_native_rdma":
            environment["SHARDSTREAM_BENCH_AWEX_RDMA"] = "1"
            environment["AWEX_MOONCAKE_PROTOCOL"] = "rdma"
        if args.publication_mechanism == "awex_native_astate":
            environment["SHARDSTREAM_BENCH_AWEX_ASTATE"] = "1"
        if args.awex_nonbatch:
            environment["AWEX_USE_BATCH_SEND_RECV"] = "0"
        rollout_counts = rollout_host_counts(recipe.rollout["instances"], len(args.rollout_hosts))
        engine_gpus = recipe.rollout["tp"] * recipe.rollout["dp"]
        for role, hosts, world in [
            ("training", args.training_hosts, recipe.training["world_size"]),
            ("rollout", args.rollout_hosts, recipe.rollout["world_size"]),
        ]:
            for host_index, host in enumerate(hosts):
                gpu_count = (rollout_counts[host_index] * engine_gpus
                             if role == "rollout" else world // len(hosts))
                actor = (
                    ray.remote(NodeProcesses)
                    .options(
                        num_cpus=1,
                        num_gpus=int(nodes[host]["Resources"]["GPU"])
                        if args.gpu_placement == "nic-spread"
                        else gpu_count,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            nodes[host]["NodeID"], soft=False
                        ),
                    )
                    .remote(
                        args.source,
                        args.runtime,
                        args.stage,
                        run_dir,
                        gpu_count,
                        args.gpu_placement,
                        recipe.rollout["tp"] if role == "rollout" else 1,
                        2
                        if role == "rollout"
                        and recipe.id == "qwen3.5-9b"
                        and world == 4
                        else 1,
                        args.nccl_lib,
                    )
                )
                actors.append(actor)
                (training if role == "training" else rollout).append(actor)
                record["node_placements"][host] = ray.get(actor.placement.remote())
        endpoints = []
        engine_rank = 0
        for host_rank, actor in enumerate(rollout):
            for local_engine in range(rollout_counts[host_rank]):
                host, port = args.rollout_hosts[host_rank], 18800 + local_engine
                devices = ",".join(
                    str(i)
                    for i in range(
                        local_engine * engine_gpus, (local_engine + 1) * engine_gpus
                    )
                )
                # Indices address the GPUs reserved for this actor, not arbitrary GPUs.
                rollout_arguments = recipe.rollout_arguments(args.model_path, host, port)
                if args.weight_format == "fp8":
                    from shardstream_benchmarks.fp8_config import fp8_server_args

                    rollout_arguments.extend(fp8_server_args(enabled=True, block_shape=(128, 128)))
                if args.disable_prefix_caching:
                    rollout_arguments.append("--no-enable-prefix-caching")
                ray.get(
                    actor.launch.remote(
                        f"rollout-{engine_rank}",
                        rollout_arguments,
                        {**environment, "SHARDSTREAM_RECIPE_GPU_INDICES": devices},
                    )
                )
                endpoints.append((engine_rank, host, port))
                engine_rank += 1
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
        # Native FP8 starts with dummy storage. Its checkpoint bootstrap and
        # generation baseline run in the training driver before measured updates.
        if args.weight_format == "bf16":
            record["generation_before"] = [generation(url, args.model_path,
                                                      recipe.id == 'moonlight-16b-a3b') for url in urls]
        if recipe.rollout["dp"] > 1 and args.weight_format == "bf16":
            record["generation_before_repeat"] = [
                generation(url, args.model_path, recipe.id == 'moonlight-16b-a3b') for url in urls
            ]
            if [r["message"] for r in record["generation_before"]] != [
                r["message"] for r in record["generation_before_repeat"]
            ]:
                raise RuntimeError(
                    "Generation baseline is not repeatable before weight transfer"
                )
        if args.communication_memory_profile:
            record["communication_memory_baseline"] = {
                url: request(url + "/communication_memory_baseline", {}) for url in urls
            }
        train_args = [
            "-m",
            "shardstream_benchmarks.model_exchange",
            "--recipe",
            recipe.id,
            "--model-path",
            args.model_path,
            "--remote-inference",
            "--num-engines",
            str(recipe.rollout["instances"]),
            "--vllm-tp-size",
            str(recipe.rollout["tp"]),
            "--vllm-dp-size",
            str(recipe.rollout["dp"]),
            "--num-updates",
            str(args.num_updates),
            "--warmup-updates",
            str(args.warmup_updates),
            "--profile",
            "--sync-transfer-start",
            "--meta-server-host",
            args.training_hosts[0],
            "--meta-server-port",
            "18890",
            "--publication-mechanism", args.publication_mechanism,
            "--publication-bucket-mb", str(args.publication_bucket_mb),
        ]
        if args.model_weight_profile:
            train_args.extend(["--model-weight-profile", "--elastic-output",
                               str(Path(run_dir) / "training-publications.json")])
        else:
            train_args.append("--validate")
        for engine, host, port in endpoints:
            train_args.extend(["--inference-endpoint", f"{engine},{host},{port}"])
        if args.rollout_replicas is not None:
            train_args.extend(["--rollout-replicas", str(args.rollout_replicas)])
        if args.layout is not None:
            train_args.extend(["--layout", args.layout])
        for node_rank, actor in enumerate(training):
            torchrun = training_launch_arguments(
                args.training_hosts, recipe.training["world_size"], node_rank,
                invocation, train_args,
            )
            ray.get(actor.launch.remote("training", torchrun, environment))
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
        if args.weight_format == "fp8":
            bootstrap = ray.get(training[0].read_json.remote("bootstrap-generation.json"))
            record.update(bootstrap)
            from shardstream_benchmarks.fp8_config import set_publication_pause

            record["generation_resume_results"] = set_publication_pause(
                urls, paused=False, request=request)
        record["generation_after"] = [generation(url, args.model_path,
                                                 recipe.id == 'moonlight-16b-a3b') for url in urls]
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

"""Run veRL's installed checkpoint engines against real Megatron/vLLM models.

The bucket packing, NCCL broadcast and RDMA ring are the original veRL code.
Only lifecycle/HTTP plumbing is supplied here. Multiprocess vLLM workers use
Ray's original NCCLGroup directly because Ray's actor-only API guard rejects
these processes. Rendezvous keys address one local GPU per publication rank.
"""

import asyncio
import base64
import importlib
import inspect
import multiprocessing
import os
import pickle
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import requests
import torch
import torch.distributed as dist

from shardstream import logging
from shardstream.integrations.publication.registry import (
    PublicationMechanism, publication_endpoints,
    register_publication_mechanism, register_vllm_publication_receiver,
)

logger = logging.getLogger(__name__)


def _request_endpoints(endpoints, route, payloads, timeout_seconds):
    # CuPy's blocking NCCL communicator constructor can hold the training
    # process's GIL. A persistent control process keeps HTTP requests progressing
    # while that constructor waits for the rollout ranks. Model tensors remain
    # in the original training/rollout processes throughout the native transfer.
    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = [executor.submit(requests.post, f"http://{host}:{port}/{route}",
                                   json=payloads[engine], timeout=timeout_seconds)
                   for engine, host, port in endpoints]
        result = []
        for endpoint, future in zip(endpoints, futures):
            response = future.result()
            if response.status_code != 200:
                raise RuntimeError(f"Engine {endpoint[0]} {route}: {response.text}")
            result.extend(_rows(response.json()["results"]))
        return result


def _encode(value):
    # veRL metadata contains dataclasses and binary RDMA descriptors. It travels
    # only between the explicitly selected benchmark processes.
    return base64.b64encode(pickle.dumps(value)).decode()


def _decode(value):
    return pickle.loads(base64.b64decode(value))


def _rows(value):
    if isinstance(value, dict):
        if "worker_rank" in value:
            yield value
        else:
            for item in value.values():
                yield from _rows(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _rows(item)


class _MultiprocessCollective:
    """The veRL collective API backed by the same Ray/CuPy NCCL implementation."""

    def __init__(self):
        self.groups = {}

    def is_group_initialized(self, group_name):
        return group_name in self.groups

    def init_collective_group(self, world_size, rank, backend, group_name):
        if backend != "nccl" or group_name in self.groups:
            raise ValueError("Expected a new NCCL group")
        import ray
        from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup

        if not ray.is_initialized():
            ray.init(address=os.environ["SHARDSTREAM_BENCH_RAY_ADDRESS"],
                     namespace="five-model-baselines-native", logging_level="ERROR")

        class SingleGPUGroup(NCCLGroup):
            def _generate_group_key(self, comm_key):
                # Actors see GPU0; multiprocess workers can see other local
                # indices. Each publication rank still contributes one GPU.
                return "one-gpu@" + self.group_name

        self.groups[group_name] = SingleGPUGroup(world_size, rank, group_name)

    def barrier(self, group_name):
        # Ray's barrier visits every visible GPU in an actor. Here only the
        # worker's current device belongs to this publication rank.
        self.groups[group_name].allreduce([
            torch.ones(1, dtype=torch.int64, device=torch.cuda.current_device())
        ])
        torch.cuda.synchronize()

    def broadcast(self, tensor, src_rank, group_name):
        from ray.util.collective.types import BroadcastOptions
        options = BroadcastOptions()
        options.root_rank = src_rank
        options.root_tensor = 0
        self.groups[group_name].broadcast([tensor], options)

    def destroy_collective_group(self, group_name):
        self.groups.pop(group_name).destroy_group()


def _engine_class(backend):
    names = {"nccl": "NCCLCheckpointEngine", "nixl": "NIXLCheckpointEngine",
             "mooncake": "MooncakeCheckpointEngine"}
    module = importlib.import_module(f"verl.checkpoint_engine.{backend}_checkpoint_engine")
    if backend == "nccl" and not isinstance(module.collective, _MultiprocessCollective):
        module.collective = _MultiprocessCollective()
    return getattr(module, names[backend])


def _new_engine(backend, config, master):
    engine_class = _engine_class(backend)
    if backend in ("nccl", "nixl"):
        import cupy
        cupy.cuda.Device(torch.cuda.current_device()).use()
    kwargs = {"bucket_size": int(config["bucket_size"]), "is_master": master}
    if backend == "nccl":
        kwargs.update(group_name=config["group_name"], rebuild_group=False)
    elif backend == "mooncake":
        # The upstream constructor selects a GPU from RANK. Keep its selection
        # on the vLLM worker's device even when CVD exposes a complete TP group.
        prior_rank = os.environ.get("RANK")
        os.environ["RANK"] = str(torch.cuda.current_device())
        try:
            return engine_class(**kwargs, rollout_dtype=None,
                                device_name=config.get("device_name", ""))
        finally:
            if prior_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = prior_rank
    return engine_class(**kwargs)


def _load_one(model, name, weight):
    if hasattr(model, "target") and hasattr(model, "draft"):
        prefix = f"model.layers.{model.num_hidden_layers}."
        if name.startswith(prefix):
            aliases = {prefix + "embed_tokens.weight": "model.embed_tokens.weight",
                       prefix + "shared_head.head.weight": "lm_head.weight"}
            model.draft.load_weights([(aliases.get(name, name), weight)])
        else:
            model.target.load_weights([(name, weight)])
    else:
        model.load_weights([(name, weight)])


class NativeReceiver:
    backend = None

    def __init__(self, config, worker_rank):
        self.config = config
        self.worker_rank = worker_rank
        self.engine = None

    def initialize(self):
        started = time.perf_counter()
        self.engine = _new_engine(self.backend, self.config, False)
        metadata = self.engine.prepare()
        self.loop = asyncio.new_event_loop()
        device = torch.cuda.current_device()
        def run_loop():
            torch.cuda.set_device(device)
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        return {"worker_rank": self.worker_rank, "metadata": _encode(metadata),
                "prepare_ms": (time.perf_counter() - started) * 1000}

    def configure(self, config):
        started = time.perf_counter()
        offset = int(self.config["rank_offset"]) + self.worker_rank
        self.engine.init_process_group(**_decode(config["rank_kwargs"])[offset])
        return {"worker_rank": self.worker_rank,
                "group_ms": (time.perf_counter() - started) * 1000}

    def update(self, model, step_id):
        count = total = 0
        receive_wait_ms = 0
        started = time.perf_counter()
        generator = self.engine.receive_weights(global_steps=step_id)
        async def next_weight():
            return await generator.__anext__()
        def weights():
            nonlocal count, total, receive_wait_ms
            while True:
                wait_start = time.perf_counter()
                try:
                    name, weight = asyncio.run_coroutine_threadsafe(
                        next_weight(), self.loop
                    ).result(timeout=self.config["timeout_seconds"])
                except StopAsyncIteration:
                    break
                receive_wait_ms += (time.perf_counter() - wait_start) * 1000
                count += 1
                total += weight.numel() * weight.element_size()
                if hasattr(model, "draft") and name.startswith(
                    f"model.layers.{model.num_hidden_layers}."
                ):
                    _load_one(model, name, weight)
                else:
                    yield name, weight
        # Consume each bucket view before advancing the native async iterator;
        # keep vLLM's original loader initialized once for the full update.
        target = getattr(model, "target", model)
        target.load_weights(weights())
        torch.cuda.synchronize()
        receive_apply_ms = (time.perf_counter() - started) * 1000
        post_started = time.perf_counter()
        architecture = model.config.architectures[0]
        if architecture in ("Glm4MoeLiteForCausalLM", "DeepseekV3ForCausalLM"):
            from shardstream.integrations.models.glm4_moe_lite import GLM47FlashVLLMWeightConverter
            GLM47FlashVLLMWeightConverter.post_update(None, model)
        torch.cuda.synchronize()
        return {"worker_rank": self.worker_rank, "received_tensors": count,
                "payload_bytes": total, "receive_wait_ms": receive_wait_ms,
                "load_apply_ms": receive_apply_ms - receive_wait_ms,
                "postprocess_ms": (time.perf_counter() - post_started) * 1000,
                "receive_apply_ms": (time.perf_counter() - started) * 1000}

    def close(self):
        if self.engine is not None:
            self.engine.finalize()
            if self.backend == "nccl":
                module = importlib.import_module("verl.checkpoint_engine.nccl_checkpoint_engine")
                module.collective.destroy_collective_group(self.config["group_name"])
            self.engine = None
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=15)
            if self.loop_thread.is_alive():
                raise TimeoutError("Native receiver event loop did not stop")
            self.loop.close()
        return {"closed": True}


class NativePublication(PublicationMechanism):
    backend = None
    training_engine_backend = None

    def __init__(self, harness, bucket_size=256 << 20, timeout_seconds=1800, **kwargs):
        super().__init__(harness, **kwargs)
        self.config = {"bucket_size": bucket_size, "timeout_seconds": timeout_seconds,
                       "group_name": "five-model-" + uuid.uuid4().hex}
        self.engine = None
        self.metadata = None
        self.timeout_seconds = timeout_seconds
        self.last_metrics = {}
        self.control = None

    def initialize_training(self):
        # Bridge uses the same canonical full-tensor export/gather as the
        # Megatron veRL engine. Preserve router FP32 and checkpoint BF16 dtypes.
        self.bridge = self.harness.mcore_bridge
        if self.bridge is None or not callable(getattr(self.bridge, "export_hf_weights", None)):
            raise RuntimeError("Native veRL baseline requires the loaded Megatron Bridge exporter")
        if self.harness.is_driver:
            self.engine = _new_engine(self.backend, self.config, True)
            self.metadata = self.engine.prepare()

    def _requests(self, route, payloads):
        return self.control.submit(_request_endpoints, publication_endpoints(self.harness),
                                   route, payloads, self.timeout_seconds).result()

    def initialize_driver(self):
        harness = self.harness
        world_per_engine = int(harness.inference_config["tp_size"]) * int(
            harness.inference_config.get("dp_size", 1))
        endpoints = publication_endpoints(harness)
        self.control = ProcessPoolExecutor(max_workers=1,
                                           mp_context=multiprocessing.get_context('spawn'))
        metadata = []
        # HTTP endpoint order defines the original veRL rollout rank ordering.
        for engine, host, port in endpoints:
            rows = self._requests_one(host, port, "publication_init", {
                "mechanism": self.name,
                "config": {**self.config, "rank_offset": engine * world_per_engine},
            })
            if [row["worker_rank"] for row in rows] != list(range(world_per_engine)):
                raise RuntimeError("Incomplete or duplicated rollout worker metadata")
            metadata.extend(_decode(row["metadata"]) for row in rows)
        full_metadata = [self.metadata] + [None] * (harness.world_size - 1) + metadata
        actors, rollout = _engine_class(self.backend).build_topology(
            harness.world_size, len(metadata), full_metadata)
        def kwargs_at(mapping, rank):
            return {key: value[rank] for key, value in mapping.items() if key != "method"}
        rollout_kwargs = [kwargs_at(rollout, rank) for rank in range(len(metadata))]
        future = self.control.submit(_request_endpoints, endpoints, "publication_configure", {
            engine: {"rank_kwargs": _encode(rollout_kwargs)}
            for engine, _, _ in endpoints}, self.timeout_seconds)
        self.engine.init_process_group(**kwargs_at(actors, 0))
        self.preparation = future.result()

    def _requests_one(self, host, port, route, payload):
        response = requests.post(f"http://{host}:{port}/{route}", json=payload,
                                 timeout=self.timeout_seconds)
        if response.status_code != 200:
            raise RuntimeError(f"{route} {host}:{port}: {response.text}")
        return sorted(_rows(response.json()["results"]), key=lambda row: row["worker_rank"])

    def _export(self):
        models = [self.harness.mcore_model]
        signature = inspect.signature(self.bridge.export_hf_weights)
        kwargs = {"show_progress": False} if "show_progress" in signature.parameters else {}
        for name, tensor in self.bridge.export_hf_weights(models, **kwargs):
            yield name, tensor

    def publish(self):
        harness = self.harness
        version = int(harness.megatron_engine.global_step)
        harness._training_barrier()
        if not harness.is_driver:
            for _ in self._export():
                pass
            return
        payloads = {engine: {"step_id": version}
                    for engine, _, _ in publication_endpoints(harness)}
        count = total = 0
        def counted():
            nonlocal count, total
            for name, weight in self._export():
                count += 1
                total += weight.numel() * weight.element_size()
                yield name, weight
        future = self.control.submit(_request_endpoints, publication_endpoints(harness),
                                     "publication_update", payloads, self.timeout_seconds)
        asyncio.run(self.engine.send_weights(counted(), global_steps=version))
        receivers = future.result()
        torch.cuda.synchronize()
        expected = int(harness.inference_config["tp_size"]) * int(
            harness.inference_config.get("dp_size", 1)) * len(publication_endpoints(harness))
        if len(receivers) != expected or any(
            row["received_tensors"] != count or row["payload_bytes"] != total
            for row in receivers
        ):
            raise RuntimeError("Native checkpoint receiver coverage or byte count mismatch")
        self.last_metrics = {"payload_bytes": total, "tensor_count": count,
                             "receivers": receivers, "backend": self.backend}
        logger.info("Native veRL publication metrics: %s", self.last_metrics)

    def close(self):
        if not self.harness.is_driver or self.engine is None:
            return
        for _, host, port in publication_endpoints(self.harness):
            try:
                response = requests.post(f"http://{host}:{port}/publication_close",
                                         timeout=min(self.timeout_seconds, 60))
                response.raise_for_status()
            except requests.RequestException:
                logger.exception("Closing native receiver failed")
        self.engine.finalize()
        if self.backend == "nccl":
            module = importlib.import_module("verl.checkpoint_engine.nccl_checkpoint_engine")
            module.collective.destroy_collective_group(self.config["group_name"])
        self.engine = None
        if self.control is not None:
            self.control.shutdown(wait=True, cancel_futures=True)
            self.control = None


for _backend in ("nccl", "nixl", "mooncake"):
    _name = "verl_native_" + _backend
    register_publication_mechanism(_name)(type(
        "VerlNative" + _backend.title(), (NativePublication,), {"backend": _backend}
    ))
    register_vllm_publication_receiver(_name)(type(
        "VerlNative" + _backend.title() + "Receiver", (NativeReceiver,), {"backend": _backend}
    ))

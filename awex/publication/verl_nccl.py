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

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Dict, Iterable, Iterator, List, Tuple

import requests
import torch
import torch.distributed as dist

from awex import logging
from awex.publication.registry import (
    PublicationMechanism,
    publication_endpoints,
    register_publication_mechanism,
    register_vllm_publication_receiver,
)
from awex.util import device as device_util
from awex.util.common import get_free_port
from awex.util.profile import emit_profile, profile_phase

logger = logging.getLogger(__name__)


@dataclass
class TensorChunkMeta:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    chunk_offset: int
    chunk_size: int
    bucket_offset: int

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["shape"] = tuple(value["shape"])
        return cls(**value)


@dataclass
class BucketMeta:
    length: int
    is_last: bool
    chunks: List[TensorChunkMeta]

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "length": self.length,
                "is_last": self.is_last,
                "chunks": [asdict(chunk) for chunk in self.chunks],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, value: bytes):
        decoded = json.loads(value.decode("utf-8"))
        return cls(
            length=int(decoded["length"]),
            is_last=bool(decoded["is_last"]),
            chunks=[TensorChunkMeta.from_dict(item) for item in decoded["chunks"]],
        )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported tensor dtype in publication metadata: {name}")
    return dtype


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _create_tcp_store(
    host: str,
    port: int,
    world_size: int,
    is_master: bool,
    timeout_seconds: int,
):
    kwargs = {
        "host_name": host,
        "port": port,
        "world_size": world_size,
        "is_master": is_master,
        "timeout": timedelta(seconds=timeout_seconds),
        "use_libuv": False,
    }
    if is_master:
        kwargs["wait_for_workers"] = False
    try:
        return dist.TCPStore(**kwargs)
    except TypeError:
        kwargs.pop("wait_for_workers", None)
        kwargs.pop("use_libuv", None)
        return dist.TCPStore(**kwargs)


def _create_nccl_process_group(
    store,
    prefix: str,
    rank: int,
    world_size: int,
    timeout_seconds: int,
):
    process_group_cls = getattr(dist, "ProcessGroupNCCL", None)
    if process_group_cls is None:
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        process_group_cls = ProcessGroupNCCL
    prefixed_store = dist.PrefixStore(prefix, store)
    options_cls = getattr(process_group_cls, "Options", None)
    if options_cls is None:
        return process_group_cls(prefixed_store, rank, world_size)
    options = options_cls()
    if hasattr(options, "_timeout"):
        options._timeout = timedelta(seconds=timeout_seconds)
    return process_group_cls(prefixed_store, rank, world_size, options)


def _broadcast(process_group, tensor: torch.Tensor):
    options = dist.BroadcastOptions()
    options.rootRank = 0
    options.rootTensor = 0
    return process_group.broadcast([tensor], options)


def _flatten_result_dicts(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _flatten_result_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _flatten_result_dicts(nested)


class McoreFullTensorExporter:
    """Reconstruct canonical named tensors on training rank 0.

    This stage is deliberately separate from the transport. It prefers the
    mbridge export generator used to create the Megatron model. When no bridge
    exporter is available, it reconstructs one logical training replica with
    the harness' Megatron-to-HF converter.
    """

    def __init__(self, train_engine, inference_tp_size: int):
        self.train_engine = train_engine
        self.inference_tp_size = inference_tp_size
        self.rank = dist.get_rank()
        self.model = train_engine.model
        if not isinstance(self.model, (list, tuple)):
            self.model = [self.model]
        self.bridge = getattr(train_engine, "publication_bridge", None)
        self.bridge_export = None
        self.parameters_meta = None
        self.weight_converter = None

    def initialize(self) -> None:
        if self.bridge is not None:
            for method_name in ("export_weights", "export_hf_weights"):
                method = getattr(self.bridge, method_name, None)
                if callable(method):
                    self.bridge_export = method
                    logger.info(
                        "Using mbridge %s for verl-style full-tensor export",
                        method_name,
                    )
                    return
            logger.warning(
                "The model bridge has no weight export generator; falling back "
                "to harness metadata reconstruction"
            )

        from awex.meta.train_meta_resolver import McoreParamMetaResolver
        from awex.models.registry import get_train_weights_converter
        from awex.sharding.mcore_sharding import get_mcore_rank_info

        infer_conf = {
            "hf_config": self.train_engine.hf_config,
            "infer_atten_tp_size": self.inference_tp_size,
            "infer_engine_config": {
                "comm_backend": "nccl",
                "tp_size": self.inference_tp_size,
            },
            "device_backend": device_util.get_device_type(),
        }
        resolver = McoreParamMetaResolver(
            self.train_engine,
            self.train_engine.hf_config,
            infer_conf,
        )
        self.parameters_meta = sorted(
            resolver.get_parameters_meta(), key=lambda item: item.name
        )
        rank_info = get_mcore_rank_info()
        infer_conf["train_pp_stage_layer_id_map"] = resolver.get_pp_stage_layer_id_map()
        self.weight_converter = get_train_weights_converter(
            self.train_engine.engine_name,
            self.train_engine.hf_config.architectures[0],
            self.train_engine.hf_config,
            rank_info,
            infer_conf,
            tf_config=self._transformer_config(),
        )

    def _transformer_config(self):
        for model in self.model:
            for attr in ("transformer_config", "config"):
                config = getattr(model, attr, None)
                if config is not None:
                    return config
        raise RuntimeError("Megatron model has no transformer config")

    def _convert_local_parameters(self) -> Dict[str, torch.Tensor]:
        from awex.converter.mcore_converter import get_mcore_model_parameters

        converted = {}
        for vp_stage, model in enumerate(self.model):
            for source_name, parameter in get_mcore_model_parameters(model).items():
                outputs = self.weight_converter.convert_param(
                    source_name, parameter.detach(), vp_stage=vp_stage
                )
                for target_name, target in outputs:
                    if target_name in converted:
                        raise ValueError(
                            f"Duplicate converted parameter on rank {self.rank}: "
                            f"{target_name}"
                        )
                    converted[target_name] = target
        if getattr(self.train_engine.hf_config, "tie_word_embeddings", False):
            if (
                "lm_head.weight" not in converted
                and "model.embed_tokens.weight" in converted
            ):
                converted["lm_head.weight"] = converted["model.embed_tokens.weight"]
        return converted

    @torch.no_grad()
    def iter_full_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        if self.bridge_export is not None:
            for name, tensor in self.bridge_export(self.model):
                if self.rank == 0:
                    yield name, tensor
            return
        if self.parameters_meta is None or self.weight_converter is None:
            raise RuntimeError("Mcore full-tensor exporter is not initialized")
        local_parameters = self._convert_local_parameters()
        for parameter_meta in self.parameters_meta:
            if not parameter_meta.replicas:
                raise ValueError(f"No training replica for {parameter_meta.name}")
            replica = parameter_meta.replicas[0]
            full_tensor = None
            if self.rank == 0:
                full_tensor = torch.empty(
                    parameter_meta.global_shape,
                    dtype=parameter_meta.dtype,
                    device=device_util.get_torch_device(),
                )

            for shard in replica.shards:
                if self.rank == shard.global_rank:
                    try:
                        shard_tensor = local_parameters[parameter_meta.name]
                    except KeyError as exc:
                        raise KeyError(
                            f"Rank {self.rank} owns metadata for "
                            f"{parameter_meta.name} but did not convert it"
                        ) from exc
                    if tuple(shard_tensor.shape) != tuple(shard.shape):
                        raise ValueError(
                            f"Converted shape mismatch for {parameter_meta.name}: "
                            f"tensor={tuple(shard_tensor.shape)} meta={tuple(shard.shape)}"
                        )
                    shard_tensor = shard_tensor.contiguous()
                else:
                    shard_tensor = torch.empty(
                        shard.shape,
                        dtype=shard.dtype,
                        device=device_util.get_torch_device(),
                    )
                dist.broadcast(shard_tensor, src=shard.global_rank)
                if full_tensor is not None:
                    slices = tuple(
                        slice(offset, offset + size)
                        for offset, size in zip(shard.global_offset, shard.shape)
                    )
                    full_tensor[slices].copy_(shard_tensor)

            local_parameters.pop(parameter_meta.name, None)
            if full_tensor is not None:
                yield parameter_meta.name, full_tensor


class VerlNcclBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        world_size: int,
        group_id: str,
        bucket_size: int,
        timeout_seconds: int,
    ):
        self.host = host
        self.port = port
        self.world_size = world_size
        self.group_id = group_id
        self.bucket_size = bucket_size
        self.timeout_seconds = timeout_seconds
        self.store = None
        self.process_group = None
        self.buffers = None

    def initialize_process_group(self) -> None:
        self.store = _create_tcp_store(
            self.host,
            self.port,
            self.world_size,
            is_master=True,
            timeout_seconds=self.timeout_seconds,
        )
        self.process_group = _create_nccl_process_group(
            self.store,
            f"{self.group_id}/nccl",
            rank=0,
            world_size=self.world_size,
            timeout_seconds=self.timeout_seconds,
        )
        self.buffers = [
            torch.empty(
                self.bucket_size,
                dtype=torch.uint8,
                device=device_util.get_torch_device(),
            )
            for _ in range(2)
        ]
        handshake = torch.zeros(
            1, dtype=torch.uint8, device=device_util.get_torch_device()
        )
        _broadcast(self.process_group, handshake).wait()

    def _metadata_key(self, step_id: int, bucket_index: int) -> str:
        return f"{self.group_id}/step/{step_id}/bucket/{bucket_index}"

    @torch.no_grad()
    def broadcast_weights(
        self, step_id: int, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> dict:
        if self.process_group is None or self.buffers is None:
            raise RuntimeError("verl NCCL sender is not initialized")
        buffer_index = 0
        bucket_index = 0
        bucket_offset = 0
        chunks: List[TensorChunkMeta] = []
        pending = [None, None]
        total_bytes = 0
        tensor_count = 0

        def flush(is_last: bool) -> None:
            nonlocal buffer_index, bucket_index, bucket_offset, chunks
            if pending[buffer_index] is not None:
                raise RuntimeError("Attempted to publish from a busy bucket buffer")
            metadata = BucketMeta(
                length=bucket_offset,
                is_last=is_last,
                chunks=chunks,
            )
            self.store.set(
                self._metadata_key(step_id, bucket_index), metadata.to_bytes()
            )
            pending[buffer_index] = _broadcast(
                self.process_group,
                self.buffers[buffer_index][:bucket_offset],
            )
            next_buffer_index = 1 - buffer_index
            previous = pending[next_buffer_index]
            if previous is not None:
                previous.wait()
                pending[next_buffer_index] = None
            buffer_index = next_buffer_index
            bucket_index += 1
            bucket_offset = 0
            chunks = []

        for name, weight in weights:
            tensor_count += 1
            contiguous = weight.detach().contiguous()
            tensor_bytes = contiguous.view(torch.uint8).reshape(-1)
            nbytes = _tensor_nbytes(contiguous)
            if nbytes == 0:
                raise ValueError(f"Cannot publish zero-sized tensor: {name}")
            total_bytes += nbytes
            chunk_offset = 0
            while chunk_offset < nbytes:
                if bucket_offset == self.bucket_size:
                    flush(is_last=False)
                available = self.bucket_size - bucket_offset
                chunk_size = min(available, nbytes - chunk_offset)
                destination = self.buffers[buffer_index][
                    bucket_offset : bucket_offset + chunk_size
                ]
                destination.copy_(
                    tensor_bytes[chunk_offset : chunk_offset + chunk_size]
                )
                chunks.append(
                    TensorChunkMeta(
                        name=name,
                        shape=tuple(contiguous.shape),
                        dtype=_dtype_name(contiguous.dtype),
                        chunk_offset=chunk_offset,
                        chunk_size=chunk_size,
                        bucket_offset=bucket_offset,
                    )
                )
                bucket_offset += chunk_size
                chunk_offset += chunk_size

        if tensor_count == 0:
            raise ValueError("Full-tensor exporter produced no weights")
        flush(is_last=True)
        for work in pending:
            if work is not None:
                work.wait()
        device_util.synchronize()
        return {
            "payload_bytes": total_bytes,
            "tensor_count": tensor_count,
            "bucket_count": bucket_index,
        }


class TensorChunkAssembler:
    def __init__(self):
        self.name = None
        self.tensor = None
        self.buffer = None
        self.offset = 0

    def consume(
        self, metadata: BucketMeta, bucket: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        completed = []
        for chunk_meta in metadata.chunks:
            dtype = _dtype_from_name(chunk_meta.dtype)
            source = bucket[
                chunk_meta.bucket_offset : (
                    chunk_meta.bucket_offset + chunk_meta.chunk_size
                )
            ]
            numel = 1
            for dimension in chunk_meta.shape:
                numel *= dimension
            tensor_nbytes = numel * dtype.itemsize
            if chunk_meta.chunk_offset == 0 and chunk_meta.chunk_size == tensor_nbytes:
                if self.tensor is not None:
                    raise ValueError(
                        f"Incomplete tensor {self.name} before {chunk_meta.name}"
                    )
                completed.append(
                    (
                        chunk_meta.name,
                        self._view_or_copy(source, dtype, chunk_meta.shape),
                    )
                )
                continue

            if self.tensor is None:
                if chunk_meta.chunk_offset != 0:
                    raise ValueError(
                        f"First chunk for {chunk_meta.name} starts at "
                        f"{chunk_meta.chunk_offset}"
                    )
                self.name = chunk_meta.name
                self.tensor = torch.empty(
                    chunk_meta.shape, dtype=dtype, device=bucket.device
                )
                self.buffer = self.tensor.view(torch.uint8).reshape(-1)
                self.offset = 0
            if self.name != chunk_meta.name or self.offset != chunk_meta.chunk_offset:
                raise ValueError(
                    f"Out-of-order tensor chunk: expected {self.name}@{self.offset}, "
                    f"got {chunk_meta.name}@{chunk_meta.chunk_offset}"
                )
            self.buffer[
                chunk_meta.chunk_offset : (
                    chunk_meta.chunk_offset + chunk_meta.chunk_size
                )
            ].copy_(source)
            self.offset += chunk_meta.chunk_size
            if self.offset == tensor_nbytes:
                completed.append((self.name, self.tensor))
                self.name = None
                self.tensor = None
                self.buffer = None
                self.offset = 0
        if metadata.is_last and self.tensor is not None:
            raise ValueError(f"Last bucket left tensor {self.name} incomplete")
        return completed

    @staticmethod
    def _view_or_copy(source, dtype, shape):
        itemsize = dtype.itemsize
        if source.storage_offset() % itemsize == 0:
            return source.view(dtype).view(shape)
        tensor = torch.empty(shape, dtype=dtype, device=source.device)
        tensor.view(torch.uint8).reshape(-1).copy_(source)
        return tensor


@register_vllm_publication_receiver("verl_nccl_broadcast")
class VerlNcclBroadcastReceiver:
    def __init__(self, config: dict, worker_rank: int):
        self.config = config
        self.worker_rank = worker_rank
        self.store = None
        self.process_group = None
        self.buffers = None

    def initialize(self) -> dict:
        world_size = int(self.config["world_size"])
        publication_rank = (
            int(self.config.get("rank_offset", 0)) + self.worker_rank + 1
        )
        if not 1 <= publication_rank < world_size:
            raise ValueError(
                f"Invalid publication rank {publication_rank} for world size {world_size}"
            )
        self.store = _create_tcp_store(
            self.config["store_host"],
            int(self.config["store_port"]),
            world_size,
            is_master=False,
            timeout_seconds=int(self.config["timeout_seconds"]),
        )
        self.process_group = _create_nccl_process_group(
            self.store,
            f"{self.config['group_id']}/nccl",
            rank=publication_rank,
            world_size=world_size,
            timeout_seconds=int(self.config["timeout_seconds"]),
        )
        bucket_size = int(self.config["bucket_size"])
        self.buffers = [
            torch.empty(
                bucket_size,
                dtype=torch.uint8,
                device=device_util.get_torch_device(),
            )
            for _ in range(2)
        ]
        handshake = torch.empty(
            1, dtype=torch.uint8, device=device_util.get_torch_device()
        )
        _broadcast(self.process_group, handshake).wait()
        return {"publication_rank": publication_rank}

    def _metadata_key(self, step_id: int, bucket_index: int) -> str:
        return f"{self.config['group_id']}/step/{step_id}/bucket/{bucket_index}"

    @torch.no_grad()
    def update(self, model, step_id: int) -> dict:
        if self.process_group is None or self.buffers is None:
            raise RuntimeError("verl NCCL receiver is not initialized")
        assembler = TensorChunkAssembler()
        bucket_index = 0
        buffer_index = 0
        metadata = BucketMeta.from_bytes(
            self.store.get(self._metadata_key(step_id, bucket_index))
        )
        work = _broadcast(
            self.process_group, self.buffers[buffer_index][: metadata.length]
        )
        work.wait()
        total_bytes = 0
        loaded_tensors = 0

        while True:
            total_bytes += metadata.length
            next_metadata = None
            next_work = None
            next_buffer_index = 1 - buffer_index
            if not metadata.is_last:
                next_metadata = BucketMeta.from_bytes(
                    self.store.get(self._metadata_key(step_id, bucket_index + 1))
                )
                next_work = _broadcast(
                    self.process_group,
                    self.buffers[next_buffer_index][: next_metadata.length],
                )

            weights = assembler.consume(metadata, self.buffers[buffer_index])
            if weights:
                model.load_weights(iter(weights))
                loaded_tensors += len(weights)
            device_util.synchronize()
            if metadata.is_last:
                break
            next_work.wait()
            bucket_index += 1
            buffer_index = next_buffer_index
            metadata = next_metadata

        return {
            "publication_rank": (
                int(self.config.get("rank_offset", 0)) + self.worker_rank + 1
            ),
            "payload_bytes": total_bytes,
            "received_tensors": loaded_tensors,
            "bucket_count": bucket_index + 1,
        }

    def close(self) -> dict:
        self.buffers = None
        self.process_group = None
        self.store = None
        return {"closed": True}


@register_publication_mechanism("verl_nccl_broadcast")
class VerlNcclBroadcastPublicationMechanism(PublicationMechanism):
    """Reproduce verl's full-tensor, bucketed NCCL broadcast data path."""

    def __init__(
        self,
        harness,
        bucket_size: int = 256 << 20,
        timeout_seconds: int = 1800,
        **kwargs,
    ):
        super().__init__(harness, **kwargs)
        if harness.device_backend != "cuda":
            raise ValueError("verl_nccl_broadcast requires CUDA/NCCL")
        if int(harness.inference_config.get("pp_size", 1)) != 1:
            raise ValueError("verl_nccl_broadcast currently requires vLLM PP size 1")
        if int(harness.inference_config.get("dp_size", 1)) != 1:
            raise ValueError("verl_nccl_broadcast currently requires vLLM DP size 1")
        if bucket_size <= 0:
            raise ValueError("verl_nccl_broadcast bucket size must be positive")
        if timeout_seconds <= 0:
            raise ValueError("verl_nccl_broadcast timeout must be positive")
        if harness.validate:
            logger.warning(
                "--validate does not run Awex's numerical weight comparison for "
                "verl_nccl_broadcast; this mechanism verifies transfer byte, "
                "bucket, and tensor counts"
            )
        self.bucket_size = bucket_size
        self.timeout_seconds = timeout_seconds
        self.exporter = None
        self.sender = None
        self.group_id = uuid.uuid4().hex if harness.is_driver else None

    def initialize_training(self) -> None:
        harness = self.harness
        self.exporter = McoreFullTensorExporter(
            harness.megatron_engine,
            inference_tp_size=harness.inference_config["tp_size"],
        )
        self.exporter.initialize()
        if not harness.is_driver:
            return
        store_host = getattr(harness, "publication_store_host", "") or "127.0.0.1"
        store_port = get_free_port()
        inference_world_size = (
            int(harness.inference_config["tp_size"])
            * int(harness.inference_config.get("num_engines", 1))
        )
        self.sender = VerlNcclBroadcastSender(
            host=store_host,
            port=store_port,
            world_size=inference_world_size + 1,
            group_id=self.group_id,
            bucket_size=self.bucket_size,
            timeout_seconds=self.timeout_seconds,
        )

    def initialize_driver(self) -> None:
        harness = self.harness
        endpoints = publication_endpoints(harness)
        inference_tp_size = int(harness.inference_config["tp_size"])
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = []
            for engine_rank, host, port in endpoints:
                config = {
                    "store_host": self.sender.host,
                    "store_port": self.sender.port,
                    "world_size": self.sender.world_size,
                    "group_id": self.group_id,
                    "bucket_size": self.bucket_size,
                    "timeout_seconds": self.timeout_seconds,
                    "rank_offset": engine_rank * inference_tp_size,
                }
                futures.append(
                    (
                        engine_rank,
                        executor.submit(
                            requests.post,
                            f"http://{host}:{port}/publication_init",
                            json={"mechanism": self.name, "config": config},
                            timeout=self.timeout_seconds,
                        ),
                    )
                )
            self.sender.initialize_process_group()
            for engine_rank, future in futures:
                response = future.result()
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Publication init failed for engine {engine_rank}: "
                        f"{response.text}"
                    )

    def publish(self) -> None:
        harness = self.harness
        step_id = int(harness.megatron_engine.global_step)
        request_future = None
        executor = None
        if harness.is_driver:
            executor = ThreadPoolExecutor(max_workers=1)
            request_future = executor.submit(self._request_updates, step_id)

        harness._training_barrier()
        start = time.perf_counter()
        if harness.is_driver:
            metrics = self.sender.broadcast_weights(
                step_id, self.exporter.iter_full_tensors()
            )
        else:
            for _ in self.exporter.iter_full_tensors():
                pass
            metrics = {}

        if request_future is not None:
            try:
                responses = request_future.result()
            finally:
                executor.shutdown(wait=True)
            receiver_metrics = [
                result
                for response in responses
                for result in _flatten_result_dicts(response.json().get("results"))
                if "publication_rank" in result and "payload_bytes" in result
            ]
            expected_receivers = int(harness.inference_config["tp_size"]) * int(
                harness.inference_config.get("num_engines", 1)
            )
            if len(receiver_metrics) != expected_receivers:
                raise RuntimeError(
                    "Publication update returned metrics for "
                    f"{len(receiver_metrics)} receivers; expected {expected_receivers}"
                )
            for receiver in receiver_metrics:
                if int(receiver["payload_bytes"]) != int(metrics["payload_bytes"]):
                    raise RuntimeError(
                        "Publication payload mismatch for rank "
                        f"{receiver['publication_rank']}: sender="
                        f"{metrics['payload_bytes']} receiver={receiver['payload_bytes']}"
                    )
                if int(receiver["bucket_count"]) != int(metrics["bucket_count"]):
                    raise RuntimeError(
                        "Publication bucket-count mismatch for rank "
                        f"{receiver['publication_rank']}: sender="
                        f"{metrics['bucket_count']} receiver={receiver['bucket_count']}"
                    )
                if int(receiver["received_tensors"]) != int(metrics["tensor_count"]):
                    raise RuntimeError(
                        "Publication tensor-count mismatch for rank "
                        f"{receiver['publication_rank']}: sender="
                        f"{metrics['tensor_count']} receiver="
                        f"{receiver['received_tensors']}"
                    )
            duration_ms = (time.perf_counter() - start) * 1000.0
            emit_profile(
                logger,
                event="publication_mechanism_update",
                role="driver",
                backend=self.name,
                phase=profile_phase(step_id),
                step_id=step_id,
                rank=0,
                duration_ms=duration_ms,
                **metrics,
            )

    def _request_updates(self, step_id: int):
        endpoints = publication_endpoints(self.harness)
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [
                (
                    engine_rank,
                    executor.submit(
                        requests.post,
                        f"http://{host}:{port}/publication_update",
                        json={"step_id": step_id},
                        timeout=self.timeout_seconds,
                    ),
                )
                for engine_rank, host, port in endpoints
            ]
            responses = []
            for engine_rank, future in futures:
                response = future.result()
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Publication update failed for engine {engine_rank}: "
                        f"{response.text}"
                    )
                responses.append(response)
            return responses

    def close(self) -> None:
        harness = self.harness
        for engine_rank, host, port in publication_endpoints(harness):
            try:
                response = requests.post(
                    f"http://{host}:{port}/publication_close",
                    timeout=min(self.timeout_seconds, 60),
                )
                if response.status_code != 200:
                    logger.warning(
                        "Publication close failed for engine %s: %s",
                        engine_rank,
                        response.text,
                    )
            except requests.RequestException as exc:
                logger.warning(
                    "Publication close request failed for engine %s: %s",
                    engine_rank,
                    exc,
                )
        if self.sender is not None:
            self.sender.buffers = None
            self.sender.process_group = None
            self.sender.store = None

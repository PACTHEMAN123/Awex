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

"""End-to-end Awex BF16-to-FP8 exchange using one mounted Qwen weight."""

from __future__ import annotations  # noqa: I001

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

# This import must precede torch so AWEX_NCCL_LIB wins SONAME resolution.
from awex.transfer import nccl_device_v2 as device_v2
from safetensors import safe_open
import torch
import torch.distributed as dist

from awex.config import InferenceConfig
from awex.meta.meta_server import MetaServerClient, start_meta_server, stop_meta_server
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
import awex.reader.nccl_reader as nccl_reader_module
import awex.reader.weights_reader as weights_reader_module
from awex.reader.weights_reader import WeightsReader
from awex.sharding.param_sharding import ShardingType
from awex.util.common import check_train_infer_params_meta
import awex.writer.nccl_writer as nccl_writer_module
from awex.writer.nccl_writer import NCCLWeightsWriter
import awex.writer.weights_writer as weights_writer_module

_DEFAULT_PARAMETER = "model.layers.0.mlp.experts.0.gate_proj.weight"


class _HFConfig:
    architectures = ["AwexStreamingCastE2EModel"]
    tie_word_embeddings = False

    def to_dict(self) -> dict:
        return {
            "architectures": self.architectures,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


class _SingleTensorModel:
    def __init__(self, name: str, tensor: torch.Tensor, config: _HFConfig):
        self.name = name
        self.tensor = tensor
        self.config = config

    def named_parameters(self):
        return [(self.name, self.tensor)]

    def state_dict(self):
        return {self.name: self.tensor}


class _IdentityConverter:
    def convert_param(self, name: str, parameter: torch.Tensor, **_kwargs):
        return [(name, parameter)]


class _InferenceMetaResolver:
    def __init__(self, parameters_meta: list[ParameterMeta]):
        self.parameters_meta = parameters_meta
        self.rank0_info = SimpleNamespace(attn_tp_size=1)

    def get_parameters_meta(self) -> list[ParameterMeta]:
        return self.parameters_meta

    def get_model_arch_name(self) -> str:
        return _HFConfig.architectures[0]


class _TrainingMetaResolver:
    parameters_meta: list[ParameterMeta] = []

    def __init__(self, *_args, **_kwargs):
        pass

    def get_parameters_meta(self) -> list[ParameterMeta]:
        return self.parameters_meta

    def get_pp_stage_layer_id_map(self) -> dict:
        return {}


class _Scheduler:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.local_rank = gpu_id
        self.awes_weights_reader = None
        self.flush_count = 0

    def flush_cache(self) -> bool:
        self.flush_count += 1
        return True


class _InferenceEngine:
    engine_name = "awex_e2e"
    num_engines = 1
    engine_rank = 0

    def __init__(
        self,
        config: InferenceConfig,
        model: _SingleTensorModel,
        scheduler: _Scheduler,
        hf_config: _HFConfig,
    ):
        self.config = config
        self.model = model
        self.hf_config = hf_config
        self.model_context = {
            "scheduler": scheduler,
            "infer_engine_config": config,
        }

    def execute_task_in_model_worker(self, task, **kwargs):
        return task(model=self.model, model_context=self.model_context, **kwargs)


class _TrainingEngine:
    engine_name = "awex_e2e"
    enable_debug_mode = False
    enable_colocate_mode = False
    comm_backend = "nccl_device_v2"

    def __init__(
        self,
        meta_server_addr: str,
        model: _SingleTensorModel,
        hf_config: _HFConfig,
    ):
        self.meta_server_addr = meta_server_addr
        self.model = model
        self.hf_config = hf_config
        self.config = {
            "weights_validation_steps": 0,
            "validate_weights_every_n_steps": 1,
            "disable_weights_exchange_pipeline": False,
            "debug_mode_config": {},
        }


class _RecordingTransport(device_v2.NCCLDeviceV2Transport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_metrics: dict = {}

    def send(self, parameters: dict, plan, step_id: int) -> dict[str, float]:
        self.last_metrics = super().send(parameters, plan, step_id)
        return self.last_metrics

    def recv(self, parameters: dict, plan, step_id: int) -> dict[str, float]:
        self.last_metrics = super().recv(parameters, plan, step_id)
        return self.last_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parameter", default=_DEFAULT_PARAMETER)
    return parser.parse_args()


def _load_weight(model_path: Path, parameter: str) -> torch.Tensor:
    with (model_path / "model.safetensors.index.json").open() as stream:
        weight_map = json.load(stream)["weight_map"]
    try:
        shard = weight_map[parameter]
    except KeyError as error:
        raise ValueError(
            f"Parameter is not present in the model: {parameter}"
        ) from error
    with safe_open(model_path / shard, framework="pt", device="cpu") as weights:
        tensor = weights.get_tensor(parameter)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected a BF16 source tensor, got {tensor.dtype}")
    return tensor.contiguous()


def _parameter_meta(
    name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> ParameterMeta:
    numel = 1
    for dimension in shape:
        numel *= dimension
    shard = ParameterShardMeta(
        tp_rank=0,
        attn_tp_rank=0,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=0,
        world_size=1,
        engine_rank=0,
        name=name,
        shape=shape,
        numel=numel,
        dtype=dtype,
        global_offset=(0,) * len(shape),
        sharding_type=ShardingType.NO_SHARDING,
        num_shards=1,
        sharding_dim=0,
    )
    return ParameterMeta(
        name=name,
        global_numel=numel,
        global_shape=shape,
        dtype=dtype,
        shards=[shard],
        replicas=[ParameterReplicaMeta(shards=[shard])],
    )


def _rank_info(*_args, **_kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        global_rank=0,
        world_size=1,
        tp_rank=0,
        tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
    )


def _install_e2e_adapters(training_meta: list[ParameterMeta]) -> None:
    _TrainingMetaResolver.parameters_meta = training_meta
    weights_writer_module.McoreParamMetaResolver = _TrainingMetaResolver
    weights_writer_module.get_train_weights_converter = (
        lambda *_args, **_kwargs: _IdentityConverter()
    )
    weights_writer_module.get_rank_info_extractor = lambda _name: _rank_info
    weights_reader_module.get_infer_weights_converter = (
        lambda *_args, **_kwargs: _IdentityConverter()
    )
    weights_reader_module.get_rank_info_extractor = lambda _name: _rank_info
    nccl_writer_module.NCCLDeviceV2Transport = _RecordingTransport
    nccl_reader_module.NCCLDeviceV2Transport = _RecordingTransport


def _assert_sender_metrics(metrics: dict, numel: int, second_step: bool) -> None:
    expected_tensor_bytes = numel * torch.bfloat16.itemsize
    expected_wire_bytes = numel * torch.float8_e4m3fn.itemsize
    if metrics["tensor_bytes"] != expected_tensor_bytes:
        raise AssertionError(f"Unexpected source byte count: {metrics}")
    if metrics["payload_bytes"] != expected_wire_bytes:
        raise AssertionError(f"Unexpected wire byte count: {metrics}")
    if metrics["wire_compression_ratio"] != 2.0:
        raise AssertionError(f"Expected 2x compression: {metrics}")
    if metrics["streaming_cast_tasks"] != 1:
        raise AssertionError(f"Expected one streaming cast task: {metrics}")
    if second_step and not metrics["plan_cache_hit"]:
        raise AssertionError(f"Second step missed the plan cache: {metrics}")


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("Awex end-to-end test requires exactly two ranks")

    meta_server = None
    if rank == 0:
        meta_server = start_meta_server()
    dist.init_process_group("gloo")
    addresses = [meta_server]
    dist.broadcast_object_list(addresses, src=0)
    meta_server_addr = f"{addresses[0][0]}:{addresses[0][1]}"

    torch.cuda.set_device(local_rank)
    base_weight = _load_weight(Path(args.model_path), args.parameter)
    shape = tuple(base_weight.shape)
    training_meta = [_parameter_meta(args.parameter, shape, torch.bfloat16)]
    inference_meta = [_parameter_meta(args.parameter, shape, torch.float8_e4m3fn)]
    check_train_infer_params_meta(
        training_meta,
        inference_meta,
        raise_exception=True,
        allow_dtype_mismatch=True,
    )
    _install_e2e_adapters(training_meta)

    hf_config = _HFConfig()
    scheduler = None
    reader = None
    writer = None
    custom_group = None
    try:
        if rank == 0:
            destination = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
            scheduler = _Scheduler(local_rank)
            config = InferenceConfig(
                model_path=str(args.model_path),
                tp_size=1,
                pp_size=1,
                dp_size=1,
                num_engines=1,
                engine_rank=0,
                meta_server_addr=meta_server_addr,
                comm_backend="nccl_device_v2",
                enable_debug_mode=False,
            )
            engine = _InferenceEngine(
                config,
                _SingleTensorModel(args.parameter, destination, hf_config),
                scheduler,
                hf_config,
            )
            reader = WeightsReader(
                engine, meta_resolver=_InferenceMetaResolver(inference_meta)
            )
            reader.initialize()
        else:
            source = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
            client = MetaServerClient(*meta_server_addr.split(":"))
            client.put_object("training_params_meta", training_meta)
            engine = _TrainingEngine(
                meta_server_addr,
                _SingleTensorModel(args.parameter, source, hf_config),
                hf_config,
            )
            writer = NCCLWeightsWriter(engine)

        dist.barrier()
        for step_id in (0, 1):
            step_weight = base_weight if step_id == 0 else base_weight.neg()
            if rank == 0:
                destination.zero_()
                expected = step_weight.to(torch.float8_e4m3fn).cuda()
            else:
                source.copy_(step_weight)
            dist.barrier()

            if rank == 0:
                reader.update_weights(step_id=step_id)
                if not torch.equal(destination.float(), expected.float()):
                    max_abs = (
                        (destination.float() - expected.float()).abs().max().item()
                    )
                    raise AssertionError(
                        f"End-to-end Qwen weight mismatch at step {step_id}: "
                        f"max_abs={max_abs}"
                    )
                worker = scheduler.awes_weights_reader
                custom_group = worker.weights_update_group
                metrics = worker.device_transport.last_metrics
                if scheduler.flush_count != step_id + 1:
                    raise AssertionError("Inference cache was not flushed after update")
            else:
                writer.write_weights(step_id=step_id)
                custom_group = writer.weights_update_group
                metrics = writer.device_transport.last_metrics
                _assert_sender_metrics(
                    metrics, base_weight.numel(), second_step=step_id == 1
                )

            dist.barrier()
            print(
                f"rank={rank} step={step_id} awex_e2e parameter={args.parameter} "
                f"shape={shape} metrics={metrics}",
                flush=True,
            )
    finally:
        transport = None
        if rank == 0 and scheduler is not None and scheduler.awes_weights_reader:
            transport = getattr(scheduler.awes_weights_reader, "device_transport", None)
        elif rank == 1 and writer is not None:
            transport = getattr(writer, "device_transport", None)
        if transport is not None:
            transport.close()
        if custom_group is not None:
            dist.destroy_process_group(custom_group)
        if dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0:
            stop_meta_server()


if __name__ == "__main__":
    main()

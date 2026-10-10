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


import gc
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributed as dist

from shardstream import logging
from shardstream._utils.common import (
    check_train_infer_params_meta,
    compute_statistics,
    from_binary,
    stripped_env_vars,
)
from shardstream._utils.gpu import get_gpu_status
from shardstream.control.store import MetaServerClient
from shardstream.integrations.conversion.megatron import get_mcore_model_parameters
from shardstream.integrations.metadata.training import McoreParamMetaResolver
from shardstream.integrations.models.registry import get_train_weights_converter
from shardstream.integrations.sharding import get_rank_info_extractor
from shardstream.layout import StaticTensorLayout
from shardstream.metadata.resolver import ParameterMeta

logger = logging.getLogger(__name__)


class WeightExchangeWriter(ABC):
    def __init__(self, train_engine):
        self.train_engine = train_engine
        self.enable_debug_mode = train_engine.enable_debug_mode

    @abstractmethod
    def initialize(self, **kwargs):
        """Initialize the weight exchange writer."""
        pass

    @abstractmethod
    def write_weights(self, step_id, **kwargs):
        pass

    def close(self) -> None:
        """Release writer-owned resources."""
        return None


class WeightsExchangeShardingWriter(WeightExchangeWriter):
    def __init__(self, train_engine):
        super().__init__(train_engine)
        self.meta_server_addr = train_engine.meta_server_addr
        logger.info(f"Meta server address: {self.meta_server_addr}")
        self.meta_server_client = MetaServerClient(*self.meta_server_addr.split(":"))
        self.infer_conf = None
        self.infer_engine_config = None
        self.model = self.train_engine.model
        if not isinstance(self.model, (list, tuple)):
            self.model = [self.model]
        self.hf_config = self.train_engine.hf_config
        self.model_arch_name = self.hf_config.architectures[0]
        self.comm_backend = getattr(self.train_engine, "comm_backend", "nccl")
        self.validated_steps = 0
        self.start_step = -1
        self.config = self.train_engine.config
        self.weights_validation_steps = self.config.get("weights_validation_steps", 0)
        self.validate_weights_every_n_steps = self.config.get(
            "validate_weights_every_n_steps", 1
        )
        self.dump_weights_for_validation = self.config.get(
            "dump_weights_for_validation", False
        )
        self.disable_pipeline = self.config.get(
            "disable_weights_exchange_pipeline", False
        )
        self.enable_nccl_debug_mode = self.config.get("debug_mode_config", {}).get(
            "enable_nccl_debug_mode", False
        )
        self.dump_weights_list_for_validation = self.config.get(
            "dump_weights_list_for_validation", []
        )
        self.dump_weights_dir_for_validation = self.config.get(
            "dump_weights_dir_for_validation", os.getcwd()
        )
        logger.info(f"Disable pipeline for weights writer: {self.disable_pipeline}")
        logger.info(f"Env variables for weights writer: {stripped_env_vars()}")
        self.lock = threading.Lock()
        self.timeout = 10000
        self.initialized = False
        self.num_infer_engines = None
        self.engine_name = train_engine.engine_name
        self.enable_mem_debug = os.environ.get("SHARDSTREAM_MEM_DEBUG", "0") == "1"
        self.already_initialized = False
        self.destroy_pg_after_update = (
            os.getenv("SHARDSTREAM_DESTROY_PG_AFTER_UPDATE", "0") == "1"
        )
        self.use_batch_send_recv = (
            os.getenv("SHARDSTREAM_USE_BATCH_SEND_RECV", "1") == "1"
        )

    def initialize(self, **kwargs):
        pass

    def _initialize(self):
        rank = dist.get_rank()
        logger.info(f"Initializing weights exchange sharding writer for rank {rank}")
        self.infer_conf = self.meta_server_client.get_object(
            "infer_conf", timeout=self.timeout
        )
        logger.info(f"Got inference config from meta server: {self.infer_conf}")
        self.infer_engine_config = self.infer_conf["infer_engine_config"]
        self.infer_world_size = self.infer_conf["infer_world_size"]
        self.rank_info = get_rank_info_extractor(self.engine_name)()
        logger.info(f"Writer rank info: {self.rank_info}")
        self.training_world_size = self.rank_info.world_size
        self.transfer_world_size = self.infer_world_size + self.training_world_size
        self.transfer_rank = self.infer_world_size + self.rank_info.global_rank
        logger.info(
            f"Writer transfer rank: {self.transfer_rank}, transfer world size: {self.transfer_world_size}"
        )
        self.parameter_meta_resolver = McoreParamMetaResolver(
            self.train_engine, self.train_engine.hf_config, self.infer_conf
        )
        self.parameters_meta = self.parameter_meta_resolver.get_parameters_meta()
        logger.info(
            "Finished querying and building parameters meta from all training workers"
        )
        if rank == 0:
            self.meta_server_client.put_object(
                "training_params_meta", self.parameters_meta
            )
            logger.info("Put training parameters meta to meta server")
        new_meta = [
            p.to_local_parameter_meta(self.rank_info.global_rank)
            for p in self.parameters_meta
        ]
        self.total_local_num_elements = sum(
            (shard.numel for p in new_meta for shard in p.shards)
        )
        self.total_local_param_size = sum(
            (shard.numel * shard.dtype.itemsize for p in new_meta for shard in p.shards)
        )
        logger.info(
            f"[Writer {self.transfer_rank}] Total local number of elements: {self.total_local_num_elements}, total local parameter size: {self.total_local_param_size}"
        )
        self.current_worker_parameters_meta = new_meta
        self.current_worker_parameters_map = {
            parameter_meta.name: parameter_meta for parameter_meta in new_meta
        }
        self._history_write_weights_time = {}
        logger.info(
            f"Start to get inference parameters meta from meta server for rank {dist.get_rank()}"
        )
        if rank == 0:
            infer_params_meta_binary = self.meta_server_client.get_binary(
                "infer_params_meta", timeout=self.timeout
            )
            dist.broadcast_object_list([infer_params_meta_binary], src=0)
        else:
            result = [None]
            dist.broadcast_object_list(result, src=0)
            infer_params_meta_binary = result[0]
        self.infer_params_meta: List[ParameterMeta] = from_binary(
            infer_params_meta_binary
        )
        logger.info("Finished getting inference parameters meta from meta server")
        check_train_infer_params_meta(
            self.parameters_meta,
            self.infer_params_meta,
            raise_exception=not self.enable_debug_mode,
        )
        self.weight_converter = get_train_weights_converter(
            self.train_engine.engine_name,
            self.model_arch_name,
            self.hf_config,
            self.rank_info,
            {
                **self.infer_conf,
                "train_pp_stage_layer_id_map": self.parameter_meta_resolver.get_pp_stage_layer_id_map(),
            },
            tf_config=_maybe_get_tf_config(self.model),
        )
        logger.info("Start to get number of inference engines from meta server")
        self.num_infer_engines = self.meta_server_client.get_object(
            "num_infer_engines", timeout=self.timeout
        )
        logger.info("Finished getting number of inference engines from meta server")
        self.infer_instance_world_size = self.infer_params_meta[0].shards[0].world_size
        logger.info(
            f"Finished building parameters for weights writer for rank {dist.get_rank()}"
        )

    @torch.no_grad()
    def convert_parameters(self, required_names=None):
        parameters = []
        for vp_stage, model in enumerate(self.model):
            parameters.extend(
                (
                    (vp_stage, name, param.detach())
                    for (name, param) in get_mcore_model_parameters(model).items()
                )
            )
        logger.info(f"[Writer {self.transfer_rank}] Start to convert parameters")
        converted = {}
        required = set(required_names) if required_names else None
        for vp_stage, name, param in parameters:
            for hf_name, hf_param in self.weight_converter.convert_param(
                name, param, vp_stage=vp_stage
            ):
                if required is not None and hf_name not in required:
                    continue
                converted[hf_name] = hf_param
        if (
            getattr(self.hf_config, "tie_word_embeddings", False)
            and self.rank_info.pp_rank == self.rank_info.pp_size - 1
            and ("lm_head.weight" not in converted)
            and ("model.embed_tokens.weight" in converted)
            and (required is None or "lm_head.weight" in required)
        ):
            converted["lm_head.weight"] = converted["model.embed_tokens.weight"]
        if required is not None:
            logger.info(
                "[Writer %s] Finished converting parameters: selected %s / required %s",
                self.transfer_rank,
                len(converted),
                len(required),
            )
        else:
            logger.info(f"[Writer {self.transfer_rank}] Finished converting parameters")
        if self.enable_mem_debug:
            self._log_converted_tensor_stats(converted, required)
        return converted

    @torch.no_grad()
    def compile_device_parameters(self, required_names):
        """Bind canonical names to stable source views for a device-only plan."""
        convert_to_layout = getattr(
            self.weight_converter, "convert_param_to_device_layout", None
        )
        if convert_to_layout is None:
            return None
        required = set(required_names)
        compiled = {}
        for vp_stage, model in enumerate(self.model):
            for source_name, source_parameter in get_mcore_model_parameters(
                model
            ).items():
                source_parameter = source_parameter.detach()
                converted = convert_to_layout(
                    source_name, source_parameter, vp_stage=vp_stage
                )
                for target_name, target in converted:
                    if target_name not in required:
                        continue
                    if target_name in compiled:
                        raise ValueError(
                            f"Duplicate compiled device parameter: {target_name}"
                        )
                    tensors = (
                        target.spans
                        if isinstance(target, StaticTensorLayout)
                        else (target,)
                    )
                    source_storage = source_parameter.untyped_storage().data_ptr()
                    if any(
                        (
                            tensor.untyped_storage().data_ptr() != source_storage
                            for tensor in tensors
                        )
                    ):
                        raise ValueError(
                            f"Device plan must bind original parameter storage, but {source_name} -> {target_name} materialized new storage"
                        )
                    if any((not tensor.is_contiguous() for tensor in tensors)):
                        raise ValueError(
                            f"Device plan requires contiguous source spans: {source_name} -> {target_name}"
                        )
                    compiled[target_name] = target
        if (
            getattr(self.hf_config, "tie_word_embeddings", False)
            and self.rank_info.pp_rank == self.rank_info.pp_size - 1
            and ("lm_head.weight" in required)
            and ("lm_head.weight" not in compiled)
            and ("model.embed_tokens.weight" in compiled)
        ):
            compiled["lm_head.weight"] = compiled["model.embed_tokens.weight"]
        missing = required - set(compiled)
        if missing:
            raise ValueError(
                f"Compiled device plan is missing required parameters: {sorted(missing)}"
            )
        logger.info(
            "[Writer %s] Compiled %s stable device parameters; per-step format conversion is disabled",
            self.transfer_rank,
            len(compiled),
        )
        return compiled

    def _log_converted_tensor_stats(self, converted, required):
        if not converted:
            logger.info("[Writer %s][MEM] converted is empty", self.transfer_rank)
            return
        total_bytes = 0
        top_items = []
        for name, tensor in converted.items():
            nbytes = int(tensor.numel()) * int(tensor.element_size())
            total_bytes += nbytes
            top_items.append((nbytes, name, tuple(tensor.shape), str(tensor.dtype)))
        top_items.sort(reverse=True)
        sample = top_items[:8]
        logger.info(
            "[Writer %s][MEM] converted tensors: count=%s required=%s total_bytes=%s top=%s",
            self.transfer_rank,
            len(converted),
            0 if required is None else len(required),
            total_bytes,
            sample,
        )

    @torch.no_grad()
    def write_weights(self, step_id, **kwargs):
        with self.lock:
            logger.info(
                f"Start to write weights for step {step_id}, current thread {threading.current_thread()}"
            )
            try:
                if not self.initialized:
                    logger.info("Start to initialize weights exchange sharding writer")
                    self._initialize()
                    self.initialized = True
                    logger.info(
                        "Finished initializing weights exchange sharding writer"
                    )
                self._validate_weights(step_id, **kwargs)
                start_time = time.time()
                self._write_weights(step_id, **kwargs)
                duration = time.time() - start_time
                compute_statistics(
                    self._history_write_weights_time, step_id, duration, "Write weights"
                )
            except Exception as e:
                logger.exception(f"Error in write_weights: {e}")
                raise e

    def _write_weights(self, step_id, **kwargs):
        logger.info(f"Writing weights for step {step_id}")
        logger.info(f"GPU status before write weights:\n{get_gpu_status()}")
        for vp_stage, model in enumerate(self.model):
            for name, param in get_mcore_model_parameters(model).items():
                temp_parameters = self.weight_converter.convert_param(
                    name, param, vp_stage=vp_stage
                )
                temp_parameters = dict(temp_parameters)
                tensor_pairs = []
                for name, parameter in temp_parameters.items():
                    if name not in self.current_worker_parameters_map:
                        raise ValueError(
                            f"Parameter {name} not found in current worker parameters map"
                        )
                    param_meta = self.current_worker_parameters_map[name]
                    assert len(param_meta.shards) == 1
                    tensor_pairs.append(
                        (name, parameter, param_meta.shards[0], param_meta)
                    )
                self.write_tensors(step_id, tensor_pairs, **kwargs)
                temp_parameters.clear()
        gc.collect()
        logger.info(f"GPU status after write weights:\n{get_gpu_status()}")
        self.finish_step(step_id)
        logger.info(f"Finished writing weights for step {step_id}")

    def _validate_weights(self, step_id, **kwargs):
        if self.validated_steps == 0:
            self.start_step = step_id
        if self.validated_steps >= self.weights_validation_steps:
            return
        if (step_id - self.start_step) % self.validate_weights_every_n_steps != 0:
            return
        self.validated_steps += 1
        need_converted_dump = bool(self.dump_weights_list_for_validation)
        for model in self.model:
            for name, parameter in model.named_parameters():
                if name in self.dump_weights_list_for_validation:
                    abs_path = os.path.join(
                        self.dump_weights_dir_for_validation,
                        f"writer_{os.getpid()}_native_{name}.{step_id}.pt",
                    )
                    torch.save(parameter.detach().cpu(), abs_path)
                    logger.info(
                        f"[Writer] Saved parameter(native) {name} to {abs_path}"
                    )
        if not need_converted_dump:
            return
        parameters = self.convert_parameters()
        for name, parameter in parameters.items():
            if name in self.dump_weights_list_for_validation:
                abs_path = os.path.join(
                    self.dump_weights_dir_for_validation,
                    f"writer_{os.getpid()}_converted_{name}.{step_id}.pt",
                )
                torch.save(parameter.detach().cpu(), abs_path)
                logger.info(f"[Writer] Saved parameter(converted) {name} to {abs_path}")

    def finish_step(self, step_id):
        pass

    def write_tensors(self, step_id, tensor_pairs: List, **kwargs):
        pass


def get_weights_exchange_writer(train_engine) -> WeightExchangeWriter:
    from shardstream.integrations.writer.worker import TransportWriter

    return TransportWriter(train_engine)


def _maybe_get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            cfg = getattr(model, attr, None)
            if cfg is not None:
                return cfg
    return None

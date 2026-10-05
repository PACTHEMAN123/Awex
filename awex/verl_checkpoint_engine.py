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

"""veRL checkpoint-engine adapters for Awex publication backends."""

from __future__ import annotations

import logging
import time
from typing import Any

import ray
from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from awex.config import InferenceConfig
from awex.engine.mcore import MegatronEngine as AwexMegatronEngine
from awex.meta.meta_server import start_meta_server
from awex.reader.weights_reader import get_weights_exchange_reader

logger = logging.getLogger(__name__)


def _config_int(config: Any, name: str, default: int = 1) -> int:
    value = getattr(config, name, default)
    return int(value if value is not None else default)


class _VerlVLLMInferenceEngine:
    """Expose veRL's Ray-backed vLLM server through Awex's reader API."""

    engine_name = "vllm"

    def __init__(
        self,
        server_adapter,
        *,
        meta_server_addr: str,
        num_engines: int,
        engine_rank: int,
        comm_backend: str,
        timeout_seconds: int,
        enable_debug_mode: bool,
        debug_mode_config: dict[str, Any],
        disable_weights_exchange_pipeline: bool,
        weights_comm_nccl_group_size: int,
    ) -> None:
        self.server_adapter = server_adapter
        self.timeout_seconds = timeout_seconds
        self.hf_config = server_adapter.model_config.hf_config
        rollout_config = server_adapter.config
        self._config = InferenceConfig(
            tp_size=_config_int(rollout_config, "tensor_model_parallel_size"),
            pp_size=_config_int(rollout_config, "pipeline_model_parallel_size"),
            dp_size=_config_int(rollout_config, "data_parallel_size"),
            ep_size=_config_int(rollout_config, "expert_parallel_size"),
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            moe_dense_tp_size=None,
            nnodes=1,
            node_rank=0,
            num_engines=num_engines,
            engine_rank=engine_rank,
            meta_server_addr=meta_server_addr,
            comm_backend=comm_backend,
            enable_debug_mode=enable_debug_mode,
            debug_mode_config=debug_mode_config,
            disable_weights_exchange_pipeline=disable_weights_exchange_pipeline,
            enable_colocate_mode=False,
            weights_exchange_ipc_backend="cuda",
            weights_comm_nccl_group_size=weights_comm_nccl_group_size,
        ).validated()
        self.weights_exchange_reader = None

    @property
    def config(self) -> InferenceConfig:
        return self._config

    @property
    def num_engines(self) -> int:
        return self._config.num_engines

    @property
    def engine_rank(self) -> int:
        return self._config.engine_rank

    def initialize(self) -> None:
        self.weights_exchange_reader = get_weights_exchange_reader(self)
        self.weights_exchange_reader.initialize()

    def update_weights(self, step_id: int) -> None:
        if self.weights_exchange_reader is None:
            raise RuntimeError("Awex vLLM reader is not initialized")
        self.weights_exchange_reader.update_weights(step_id=step_id)

    def execute_task_in_model_worker(self, function, **kwargs):
        if isinstance(function, str):
            method = function
            payload = kwargs
        else:
            method = "awex_execute"
            infer_config = kwargs.get("infer_engine_config")
            if isinstance(infer_config, InferenceConfig):
                kwargs = dict(kwargs)
                kwargs["infer_engine_config"] = infer_config.__dict__
            payload = {
                "task_module": function.__module__,
                "task_qualname": function.__qualname__,
                "task_kwargs": kwargs,
            }

        if not self.server_adapter._ensure_server_handle():
            raise RuntimeError("Awex must run on the vLLM server-owning rollout rank")
        result_ref = self.server_adapter.server_handle.collective_rpc.remote(
            method,
            timeout=self.timeout_seconds,
            args=(),
            kwargs=payload,
        )
        return ray.get(result_ref, timeout=self.timeout_seconds)

    def release_memory_occupation(self, tags=None) -> None:
        return None

    def resume_memory_occupation(self, tags=None) -> None:
        return None


class AwexCheckpointEngine(CheckpointEngine):
    """Bridge veRL Megatron actors directly to Awex vLLM readers."""

    requires_model_engine = True
    handles_receive = True
    wire_format = "awex_direct"

    def __init__(
        self,
        bucket_size: int,
        *,
        comm_backend: str,
        is_master: bool = False,
        timeout_seconds: int = 1800,
        enable_debug_mode: bool = False,
        debug_mode_config: dict[str, Any] | None = None,
        disable_weights_exchange_pipeline: bool = False,
        weights_comm_nccl_group_size: int = 1,
    ) -> None:
        del bucket_size
        self.comm_backend = comm_backend
        self.is_master = is_master
        self.timeout_seconds = timeout_seconds
        self.enable_debug_mode = enable_debug_mode
        self.debug_mode_config = debug_mode_config or {}
        self.disable_weights_exchange_pipeline = disable_weights_exchange_pipeline
        self.weights_comm_nccl_group_size = weights_comm_nccl_group_size
        self.server_adapter = None
        self.meta_server_addr = None
        self.num_engines = None
        self.engine_rank = None
        self.role = None
        self._awex_training_engine = None
        self._awex_inference_engine = None

    def bind_server_adapter(self, server_adapter) -> None:
        self.server_adapter = server_adapter

    def prepare(self) -> dict[str, Any]:
        if self.server_adapter is not None:
            return {
                "role": "rollout",
                "replica_rank": int(self.server_adapter.replica_rank),
                "is_leader": bool(getattr(self.server_adapter, "_has_server", False)),
            }

        metadata = {"role": "actor", "is_master": self.is_master}
        if self.is_master:
            if self.meta_server_addr is None:
                host, port = start_meta_server()
                self.meta_server_addr = f"{host}:{port}"
            metadata["meta_server_addr"] = self.meta_server_addr
        return metadata

    @classmethod
    def build_topology(
        cls,
        actor_wg_world_size: int,
        rollout_world_size: int,
        metadata: list[dict[str, Any]],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        actor_metadata = metadata[:actor_wg_world_size]
        rollout_metadata = metadata[actor_wg_world_size:]
        if len(rollout_metadata) != rollout_world_size:
            raise ValueError(
                f"Expected {rollout_world_size} rollout metadata entries, "
                f"got {len(rollout_metadata)}"
            )

        master_addresses = [
            item.get("meta_server_addr")
            for item in actor_metadata
            if item and item.get("meta_server_addr")
        ]
        if len(master_addresses) != 1:
            raise ValueError(
                "Awex topology requires exactly one actor meta server; "
                f"found {len(master_addresses)}"
            )
        meta_server_addr = master_addresses[0]

        leader_ranks = [
            int(item["replica_rank"])
            for item in rollout_metadata
            if item.get("is_leader")
        ]
        replica_ranks = sorted(set(leader_ranks))
        if replica_ranks != list(range(len(replica_ranks))):
            raise ValueError(
                "Awex rollout replica ranks must be contiguous from zero; "
                f"got {replica_ranks}"
            )
        if len(leader_ranks) != len(replica_ranks):
            raise ValueError(
                "Awex topology requires exactly one server-owning checkpoint "
                "worker per rollout replica"
            )
        if not replica_ranks:
            raise ValueError("Awex topology found no rollout replica leaders")
        num_engines = len(replica_ranks)

        actor_kwargs = {
            "role": ["actor"] * actor_wg_world_size,
            "meta_server_addr": [meta_server_addr] * actor_wg_world_size,
            "num_engines": [num_engines] * actor_wg_world_size,
            "engine_rank": [None] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "role": ["rollout"] * rollout_world_size,
            "meta_server_addr": [meta_server_addr] * rollout_world_size,
            "num_engines": [num_engines] * rollout_world_size,
            "engine_rank": [int(item["replica_rank"]) for item in rollout_metadata],
        }
        return actor_kwargs, rollout_kwargs

    def init_process_group(
        self,
        *,
        role: str,
        meta_server_addr: str,
        num_engines: int,
        engine_rank: int | None,
    ) -> None:
        if role not in {"actor", "rollout"}:
            raise ValueError(f"Unknown Awex checkpoint role: {role}")
        self.role = role
        self.meta_server_addr = meta_server_addr
        self.num_engines = int(num_engines)
        self.engine_rank = engine_rank

    def finalize(self) -> None:
        # Awex keeps its process group and compiled device schedule across
        # publications; Ray process teardown owns final cleanup.
        return None

    async def send_weights(self, model_engine, global_steps: int | None = None):
        if self.role != "actor":
            raise RuntimeError(f"send_weights called for Awex role {self.role!r}")
        if global_steps is None:
            raise ValueError("Awex publication requires a global_steps version")
        if self._awex_training_engine is None:
            try:
                model = model_engine.module
                hf_config = model_engine.model_config.hf_config
            except AttributeError as exc:
                raise TypeError(
                    "Awex checkpoint backends require veRL's MegatronEngine"
                ) from exc
            self._awex_training_engine = AwexMegatronEngine(
                {
                    "comm_backend": self.comm_backend,
                    "meta_server_addr": self.meta_server_addr,
                    "enable_debug_mode": self.enable_debug_mode,
                    "debug_mode_config": self.debug_mode_config,
                    "disable_weights_exchange_pipeline": self.disable_weights_exchange_pipeline,
                },
                hf_config,
                model,
            )
            self._awex_training_engine.initialize()

        start = time.perf_counter()
        self._awex_training_engine.set_global_step(int(global_steps))
        self._awex_training_engine.write_weights()
        duration = time.perf_counter() - start
        if self.is_master:
            logger.info(
                "Awex %s publication step=%s completed in %.6f seconds",
                self.comm_backend,
                global_steps,
                duration,
            )
            return {"timing_s/awex_writer": duration}
        return {}

    async def receive_weights(self, global_steps: int | None = None):
        if self.role != "rollout":
            raise RuntimeError(f"receive_weights called for Awex role {self.role!r}")
        if global_steps is None:
            raise ValueError("Awex publication requires a global_steps version")
        if not bool(getattr(self.server_adapter, "_has_server", False)):
            return None
        if self._awex_inference_engine is None:
            self._awex_inference_engine = _VerlVLLMInferenceEngine(
                self.server_adapter,
                meta_server_addr=self.meta_server_addr,
                num_engines=self.num_engines,
                engine_rank=int(self.engine_rank),
                comm_backend=self.comm_backend,
                timeout_seconds=self.timeout_seconds,
                enable_debug_mode=self.enable_debug_mode,
                debug_mode_config=self.debug_mode_config,
                disable_weights_exchange_pipeline=self.disable_weights_exchange_pipeline,
                weights_comm_nccl_group_size=self.weights_comm_nccl_group_size,
            )
            self._awex_inference_engine.initialize()
        self._awex_inference_engine.update_weights(int(global_steps))
        return None


@CheckpointEngineRegistry.register("awex_nccl")
class AwexNcclCheckpointEngine(AwexCheckpointEngine):
    def __init__(self, bucket_size: int, **kwargs) -> None:
        super().__init__(bucket_size, comm_backend="nccl", **kwargs)


@CheckpointEngineRegistry.register("awex_weightrail")
class AwexWeightRailCheckpointEngine(AwexCheckpointEngine):
    def __init__(self, bucket_size: int, **kwargs) -> None:
        super().__init__(bucket_size, comm_backend="nccl_device_v2", **kwargs)

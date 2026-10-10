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


import asyncio
import inspect
from typing import Any, Dict, Optional

from shardstream import logging
from shardstream.integrations.config import InferenceConfig
from shardstream.integrations.reader.base import get_weights_exchange_reader

logger = logging.getLogger(__name__)


class ShardStreamVLLMServerAdapter:
    """Server-side ShardStream adapter that uses vLLM EngineCoreClient for execution."""

    def __init__(
        self,
        engine_client,
        meta_server_addr: str,
        engine_rank: int = 0,
        num_engines: int = 1,
        comm_backend: str = "transport",
        enable_debug_mode: bool = False,
        debug_mode_config: Optional[Dict[str, Any]] = None,
        disable_weights_exchange_pipeline: bool = False,
        weights_comm_nccl_group_size: int = 1,
        nnodes: Optional[int] = None,
        node_rank: Optional[int] = None,
        weights_validation_steps: int = 0,
        validate_weights_every_n_steps: int = 1,
        dump_weights_list_for_validation: Optional[list[str]] = None,
        dump_weights_dir_for_validation: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._engine_client = engine_client
        self._engine_core = engine_client.engine_core
        self._initialized = False
        self._initializing = False
        self._loop = loop
        vllm_config = engine_client.vllm_config
        parallel_config = vllm_config.parallel_config
        self.hf_config = engine_client.model_config.hf_config
        self.engine_name = "vllm"
        pc_nnodes = getattr(parallel_config, "nnodes", None)
        pc_node_rank = getattr(parallel_config, "node_rank", None)
        resolved_nnodes = nnodes if nnodes is not None else pc_nnodes or 1
        resolved_node_rank = node_rank if node_rank is not None else pc_node_rank or 0
        enable_expert_parallel = bool(
            getattr(parallel_config, "enable_expert_parallel", False)
        )
        pcp_size = int(
            getattr(parallel_config, "prefill_context_parallel_size", 1) or 1
        )
        inferred_ep_size = (
            parallel_config.tensor_parallel_size
            * parallel_config.data_parallel_size
            * pcp_size
            if enable_expert_parallel
            else 1
        )
        self._config = InferenceConfig(
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            dp_size=parallel_config.data_parallel_size,
            ep_size=inferred_ep_size,
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            moe_dense_tp_size=None,
            nnodes=resolved_nnodes,
            node_rank=resolved_node_rank,
            num_engines=num_engines,
            engine_rank=engine_rank,
            meta_server_addr=meta_server_addr,
            comm_backend=comm_backend,
            enable_debug_mode=enable_debug_mode,
            debug_mode_config=debug_mode_config or {},
            disable_weights_exchange_pipeline=disable_weights_exchange_pipeline,
            weights_comm_nccl_group_size=weights_comm_nccl_group_size,
            weights_validation_steps=weights_validation_steps,
            validate_weights_every_n_steps=validate_weights_every_n_steps,
            dump_weights_list_for_validation=dump_weights_list_for_validation or [],
            dump_weights_dir_for_validation=dump_weights_dir_for_validation,
        )
        self.weights_exchange_reader = None

    @property
    def config(self):
        return self._config

    @property
    def num_engines(self):
        return self._config.num_engines

    @property
    def engine_rank(self):
        return self._config.engine_rank

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._initializing:
            return
        self._initializing = True
        logger.info("Initializing ShardStream weights reader in vLLM server process.")
        self.weights_exchange_reader = get_weights_exchange_reader(self)
        self.weights_exchange_reader.initialize()
        self._initialized = True
        self._initializing = False

    def update_weights(self, step_id: int, **kwargs):
        if not self._initialized:
            raise RuntimeError("ShardStream adapter not initialized.")
        self.weights_exchange_reader.update_weights(step_id=step_id, **kwargs)
        # Scheduler caches live outside model workers on vLLM V1.
        if self._call_engine_async("reset_prefix_cache") is False:
            raise RuntimeError(
                "vLLM prefix cache reset failed after weight publication"
            )
        if hasattr(self._engine_client, "reset_encoder_cache"):
            self._call_engine_async("reset_encoder_cache")

    def release_memory_occupation(self, tags=None) -> None:
        logger.info("Release memory occupation via vLLM sleep.")
        self._call_engine_async("sleep", 1)

    def resume_memory_occupation(self, tags=None) -> None:
        logger.info("Resume memory occupation via vLLM wake_up.")
        self._call_engine_async("wake_up", tags)

    def _call_engine_async(self, method: str, *args):
        fn = getattr(self._engine_client, method, None)
        if fn is None:
            raise RuntimeError(f"Engine client has no method {method}")
        if inspect.iscoroutinefunction(fn):
            return self._run_on_loop(fn(*args))
        return fn(*args)

    def _collective_rpc(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        if hasattr(self._engine_client, "collective_rpc"):
            fn = self._engine_client.collective_rpc
            if inspect.iscoroutinefunction(fn):
                return self._run_on_loop(fn(method, timeout, args, kwargs))
            return fn(method, timeout, args, kwargs)
        return self._engine_core.collective_rpc(
            method, timeout=timeout, args=args, kwargs=kwargs
        )

    def _collective_rpc_all_dp_cores(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        rpc_client = self._get_dp_rpc_client()
        if not (
            rpc_client is not None
            and hasattr(rpc_client, "core_engines")
            and hasattr(rpc_client, "_call_utility_async")
        ):
            return [
                self._collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)
            ]
        core_engines = list(getattr(rpc_client, "core_engines", []))
        if not core_engines:
            return [
                self._collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)
            ]
        call_utility = rpc_client._call_utility_async
        if inspect.iscoroutinefunction(call_utility):

            async def _gather_all():
                return await asyncio.gather(
                    *[
                        call_utility(
                            "collective_rpc",
                            method,
                            timeout,
                            args,
                            kwargs,
                            engine=engine,
                        )
                        for engine in core_engines
                    ]
                )

            return self._run_on_loop(_gather_all())
        return [
            call_utility("collective_rpc", method, timeout, args, kwargs, engine=engine)
            for engine in core_engines
        ]

    def _get_dp_rpc_client(self):
        if hasattr(self._engine_client, "core_engines") and hasattr(
            self._engine_client, "_call_utility_async"
        ):
            return self._engine_client
        engine_core = getattr(self._engine_client, "engine_core", None)
        if (
            engine_core is not None
            and hasattr(engine_core, "core_engines")
            and hasattr(engine_core, "_call_utility_async")
        ):
            return engine_core
        return None

    def _run_on_loop(self, coro):
        if self._loop is None:
            raise RuntimeError("No event loop available for async engine calls.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @staticmethod
    def _is_meta_collection_call(method: str, payload: dict) -> bool:
        if method == "_get_model_param_info":
            return True
        if method != "shardstream_execute":
            return False
        task_qualname = payload.get("task_qualname", "")
        return isinstance(task_qualname, str) and task_qualname.endswith(
            "_get_model_param_info"
        )

    def execute_task_in_model_worker(self, fn, **kwargs):
        if not self._initialized and (not self._initializing):
            raise RuntimeError("ShardStream adapter not initialized.")
        if isinstance(fn, str):
            method = fn
            payload = kwargs
        else:
            method = "shardstream_execute"
            infer_engine_config = kwargs.get("infer_engine_config")
            if isinstance(infer_engine_config, InferenceConfig):
                kwargs = dict(kwargs)
                kwargs["infer_engine_config"] = infer_engine_config.__dict__
            payload = {
                "task_module": fn.__module__,
                "task_qualname": fn.__qualname__,
                "task_kwargs": kwargs,
            }
        # vLLM's DP load balancer executes on every core but returns only the
        # first core's result. Collect each core explicitly so a validation
        # failure on any expert/DP rank reaches the publication driver.
        core_results = self._collective_rpc_all_dp_cores(
            method, args=(), kwargs=payload
        )
        merged_results = []
        for result in core_results:
            if isinstance(result, list):
                merged_results.extend(result)
            else:
                merged_results.append(result)
        if self._is_meta_collection_call(method, payload):
            logger.info(
                "Collected model param meta from %s DP core(s), merged %s entries.",
                len(core_results),
                len(merged_results),
            )
        return merged_results

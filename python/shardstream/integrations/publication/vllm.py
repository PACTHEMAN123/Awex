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

from shardstream.integrations.vllm.server import ShardStreamVLLMServerAdapter


class PublicationVLLMServerAdapter:
    """Thin API-server adapter for mechanism-specific vLLM worker state."""

    def __init__(self, engine_client, loop, timeout_seconds: int):
        self._engine_client = engine_client
        self._engine_core = engine_client.engine_core
        self._loop = loop
        self.timeout_seconds = timeout_seconds
        self.mechanism = None
        # Reuse the tested vLLM RPC plumbing without initializing any
        # ShardStream transport, metadata, communicator, or weights reader.
        self._rpc = ShardStreamVLLMServerAdapter.__new__(ShardStreamVLLMServerAdapter)
        self._rpc._engine_client = engine_client
        self._rpc._engine_core = self._engine_core
        self._rpc._loop = loop

    def initialize(self, mechanism: str, config: dict):
        self.mechanism = mechanism
        return self._collective_rpc(
            "publication_init",
            timeout=self.timeout_seconds,
            args=(mechanism, config),
        )

    def update(self, step_id: int):
        if self.mechanism is None:
            raise RuntimeError("Publication adapter is not initialized")
        result = self._collective_rpc(
            "publication_update",
            timeout=self.timeout_seconds,
            args=(step_id,),
        )
        if self._rpc._call_engine_async("reset_prefix_cache") is False:
            raise RuntimeError("vLLM prefix cache reset failed after publication")
        if hasattr(self._engine_client, "reset_encoder_cache"):
            self._rpc._call_engine_async("reset_encoder_cache")
        return result

    def configure(self, config: dict):
        return self._collective_rpc(
            "publication_configure", timeout=self.timeout_seconds, args=(config,)
        )

    def execute_task_in_model_worker(self, task, **kwargs):
        return self._collective_rpc(
            "shardstream_execute", timeout=self.timeout_seconds,
            args=(task.__module__, task.__qualname__, kwargs),
        )

    def close(self):
        if self.mechanism is None:
            return []
        try:
            return self._collective_rpc(
                "publication_close",
                timeout=self.timeout_seconds,
            )
        finally:
            self.mechanism = None

    def _collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        cores = self._rpc._collective_rpc_all_dp_cores(
            method, timeout=timeout, args=args, kwargs=kwargs
        )
        return [item for result in cores
                for item in (result if isinstance(result, list) else [result])]

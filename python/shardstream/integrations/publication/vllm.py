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


class PublicationVLLMServerAdapter:
    """Thin API-server adapter for mechanism-specific vLLM worker state."""

    def __init__(self, engine_client, loop, timeout_seconds: int):
        self._engine_client = engine_client
        self._engine_core = engine_client.engine_core
        self._loop = loop
        self.timeout_seconds = timeout_seconds
        self.mechanism = None

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
        return self._collective_rpc(
            "publication_update",
            timeout=self.timeout_seconds,
            args=(step_id,),
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
        if hasattr(self._engine_client, "collective_rpc"):
            function = self._engine_client.collective_rpc
            if inspect.iscoroutinefunction(function):
                coroutine = function(method, timeout, args, kwargs)
                return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()
            return function(method, timeout, args, kwargs)
        return self._engine_core.collective_rpc(
            method, timeout=timeout, args=args, kwargs=kwargs
        )

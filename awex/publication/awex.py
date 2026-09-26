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

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import requests

from awex.publication.registry import (
    PublicationMechanism,
    publication_endpoints,
    register_publication_mechanism,
)


@register_publication_mechanism("awex")
class AwexPublicationMechanism(PublicationMechanism):
    """Adapter that preserves the original Awex integration-test path."""

    uses_awex_meta_server = True

    def __init__(self, harness, **kwargs):
        super().__init__(harness, **kwargs)
        self.training_engine_backend = harness.comm_backend

    def initialize_driver(self) -> None:
        harness = self.harness
        endpoints = publication_endpoints(harness)
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [
                executor.submit(self._initialize_endpoint, endpoint)
                for endpoint in endpoints
            ]
            for future in futures:
                future.result()

    def _initialize_endpoint(self, endpoint) -> None:
        engine_rank, host, port = endpoint
        harness = self.harness
        payload = {
            "meta_server_addr": harness.meta_server_addr,
            "engine_rank": engine_rank,
            "num_engines": harness.inference_config["num_engines"],
            "comm_backend": harness.inference_config["comm_backend"],
            "enable_debug_mode": harness.inference_config["enable_debug_mode"],
            "nnodes": 1,
            "node_rank": 0,
        }
        if harness.device_backend == "npu":
            payload["weights_exchange_ipc_backend"] = "cpu"
        if harness.validate:
            payload["weights_validation_steps"] = 1
            payload["validate_weights_every_n_steps"] = 1
            if harness.dump_weights_list_for_validation:
                payload["dump_weights_list_for_validation"] = (
                    harness.dump_weights_list_for_validation
                )
            if harness.dump_weights_dir_for_validation:
                payload["dump_weights_dir_for_validation"] = (
                    harness.dump_weights_dir_for_validation
                )
        response = requests.post(
            f"http://{host}:{port}/areal_awex_init", json=payload, timeout=60
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Awex init failed for engine {engine_rank}: {response.text}"
            )

    def publish(self) -> None:
        harness = self.harness
        if harness.comm_backend == "file":
            temp_context = tempfile.TemporaryDirectory()
            path = os.path.join(temp_context.name, "checkpoint")
        else:
            temp_context = nullcontext()
            path = None

        with temp_context:
            if harness.comm_backend == "file":
                harness.megatron_engine.write_weights(path=path)
                self._request_update(path=path)
                return

            executor_context = (
                ThreadPoolExecutor(max_workers=1)
                if harness.is_driver
                else nullcontext()
            )
            with executor_context as executor:
                future = (
                    executor.submit(self._request_update, path=None)
                    if executor is not None
                    else None
                )
                harness._training_barrier()
                harness.megatron_engine.write_weights()
                if future is not None:
                    future.result()

    def _request_update(self, path: str | None) -> None:
        harness = self.harness
        payload = {"step_id": harness.megatron_engine.global_step, "kwargs": {}}
        if path is not None:
            payload["kwargs"]["path"] = path
        endpoints = publication_endpoints(harness)
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [
                executor.submit(
                    requests.post,
                    f"http://{host}:{port}/areal_awex_update",
                    json=payload,
                    timeout=300,
                )
                for _, host, port in endpoints
            ]
            for endpoint, future in zip(endpoints, futures):
                response = future.result()
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Awex update failed for engine {endpoint[0]}: "
                        f"{response.text}"
                    )

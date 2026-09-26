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

import importlib
from abc import ABC, abstractmethod
from typing import Dict, Type


class PublicationMechanism(ABC):
    """Training-side lifecycle used by the publication harness."""

    uses_awex_meta_server = False
    training_engine_backend = "file"

    def __init__(self, harness, **kwargs):
        self.harness = harness

    def initialize_training(self) -> None:
        """Initialize state shared by all training ranks."""
        return None

    def initialize_driver(self) -> None:
        """Initialize transport state after the inference server is healthy."""
        return None

    @abstractmethod
    def publish(self) -> None:
        """Publish the current training weights to the inference workers."""

    def close(self) -> None:
        """Release mechanism-owned resources before the server exits."""
        return None


_MECHANISMS: Dict[str, Type[PublicationMechanism]] = {}
_VLLM_RECEIVERS = {}
_BUILTINS_LOADED = False


def _load_builtin_mechanisms() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    importlib.import_module("awex.publication.awex")
    importlib.import_module("awex.publication.verl_nccl")
    _BUILTINS_LOADED = True


def register_publication_mechanism(name: str):
    def decorator(cls: Type[PublicationMechanism]):
        if name in _MECHANISMS:
            raise ValueError(f"Publication mechanism already registered: {name}")
        _MECHANISMS[name] = cls
        cls.name = name
        return cls

    return decorator


def register_vllm_publication_receiver(name: str):
    def decorator(cls):
        if name in _VLLM_RECEIVERS:
            raise ValueError(f"vLLM publication receiver already registered: {name}")
        _VLLM_RECEIVERS[name] = cls
        return cls

    return decorator


def create_publication_mechanism(name: str, harness, **kwargs):
    _load_builtin_mechanisms()
    try:
        mechanism_cls = _MECHANISMS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_MECHANISMS))
        raise ValueError(
            f"Unknown publication mechanism {name!r}; available: {choices}"
        ) from exc
    return mechanism_cls(harness, **kwargs)


def publication_mechanism_names():
    _load_builtin_mechanisms()
    return tuple(sorted(_MECHANISMS))


def create_vllm_publication_receiver(name: str, config: dict, worker_rank: int):
    _load_builtin_mechanisms()
    try:
        receiver_cls = _VLLM_RECEIVERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_VLLM_RECEIVERS))
        raise ValueError(
            f"No vLLM receiver for publication mechanism {name!r}; "
            f"available: {choices}"
        ) from exc
    return receiver_cls(config, worker_rank)

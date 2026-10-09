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


from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device_type() -> str:
    override = os.environ.get("SHARDSTREAM_DEVICE_TYPE", "").strip().lower()
    if override and override not in {"cuda", "cpu"}:
        raise ValueError("ShardStream supports CUDA, with CPU metadata inspection")
    return override or ("cuda" if torch.cuda.is_available() else "cpu")


def device_count() -> int:
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.device_count()
    return 0


def current_device() -> int:
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.current_device()
    return 0


def set_device(device_id: int) -> None:
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.set_device(device_id)


def synchronize(device_id: int | None = None) -> None:
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.synchronize(device=device_id)


def get_device_name(device_id: int | None = None) -> str:
    device_type = get_device_type()
    if device_type == "cuda":
        idx = current_device() if device_id is None else device_id
        return torch.cuda.get_device_name(idx)
    return "cpu"


def get_torch_device(device_id: int | None = None) -> torch.device:
    device_type = get_device_type()
    if device_type == "cuda":
        idx = current_device() if device_id is None else device_id
        return torch.device(f"{device_type}:{idx}")
    return torch.device("cpu")


def get_device_properties(device_id: int | None = None):
    device_type = get_device_type()
    if device_type == "cuda":
        idx = current_device() if device_id is None else device_id
        return torch.cuda.get_device_properties(idx)
    raise RuntimeError("Device properties only available for CUDA.")


def visible_devices_env_names() -> list[str]:
    get_device_type()
    return ["CUDA_VISIBLE_DEVICES"]


def visible_devices_env_value() -> str:
    for name in visible_devices_env_names():
        value = os.environ.get(name)
        if value:
            return value
    return ""


def get_stream_class() -> type | None:
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.Stream
    return None


def create_stream(device_id: int | None = None):
    stream_cls = get_stream_class()
    if stream_cls is None:
        return None
    if device_id is None:
        return stream_cls()
    return stream_cls(device=device_id)


@contextmanager
def stream(stream_obj) -> Iterator[None]:
    if stream_obj is None:
        yield
        return
    device_type = get_device_type()
    if device_type == "cuda":
        with torch.cuda.stream(stream_obj):
            yield
        return
    yield

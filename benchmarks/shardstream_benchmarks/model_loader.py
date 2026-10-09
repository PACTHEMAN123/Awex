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

import torch
from transformers import AutoConfig


def megatron_model_from_hf(
    model_path: str = "Qwen/Qwen2-1.5B",
    use_mbridge: bool = True,
    return_bridge: bool = False,
):
    from pathlib import Path

    if not use_mbridge:
        raise ValueError(
            "The real-model benchmark loads local HF weights through mbridge"
        )
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    hf_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    loaded = initialize_megatron_and_load_hf_with_mbridge(
        hf_config, str(model_dir), return_bridge=return_bridge
    )
    if return_bridge:
        (model, bridge) = loaded
        return (model if isinstance(model, list) else [model], hf_config, bridge)
    return (loaded if isinstance(loaded, list) else [loaded], hf_config)


def _ensure_mbridge_custom_fsdp_shim() -> None:
    import importlib
    import sys
    import types

    try:
        importlib.import_module("megatron.core.distributed.custom_fsdp")
        return
    except Exception:
        pass
    try:
        import megatron.core.distributed as dist_mod

        fsdp_cls = getattr(dist_mod, "TorchFullyShardedDataParallel", None)
    except Exception:
        fsdp_cls = None
    shim = types.ModuleType("megatron.core.distributed.custom_fsdp")
    if fsdp_cls is None:

        class _DummyFSDP:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("custom_fsdp shim used without FSDP support")

        shim.FullyShardedDataParallel = _DummyFSDP
    else:
        shim.FullyShardedDataParallel = fsdp_cls
    sys.modules["megatron.core.distributed.custom_fsdp"] = shim


def initialize_megatron_and_load_hf_with_mbridge(
    hf_config, hf_model_dir, return_bridge=False
):
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from shardstream._utils import device as device_util

    if device_util.get_device_type() == "cuda":
        device_env = os.environ.get("DEVICE")
        if device_env is not None:
            try:
                device_id = int(device_env)
                print(
                    f"Setting torch device to {device_id} based on DEVICE={device_env}"
                )
                device_util.set_device(device_id)
            except Exception as e:
                print(f"Warning: Failed to set device from DEVICE={device_env}: {e}")
    if not dist.is_initialized():
        from shardstream._utils.common import get_free_port

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(get_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            create_gloo_process_groups=False,
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        seed = int(os.environ.get("SHARDSTREAM_MBRIDGE_SEED", "42"))
        model_parallel_cuda_manual_seed(seed)
    _ensure_mbridge_custom_fsdp_shim()
    try:
        from mbridge import AutoBridge
    except ModuleNotFoundError as exc:
        if exc.name != "mbridge":
            raise
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(hf_model_dir)
        provider = bridge.to_megatron_provider(load_weights=False)
        parallel_overrides = {
            "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
            "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
            "context_parallel_size": mpu.get_context_parallel_world_size(),
            "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
            "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
            "sequence_parallel": mpu.get_tensor_model_parallel_world_size() > 1,
        }
        for name, value in parallel_overrides.items():
            if hasattr(provider, name):
                setattr(provider, name, value)
        provider.finalize()
        model = provider.provide_distributed_model(
            wrap_with_ddp=False, fp16=provider.fp16, bf16=provider.bf16
        )
        bridge.load_hf_weights(model, hf_model_dir)
    else:
        bridge = AutoBridge.from_pretrained(hf_model_dir)
        model = bridge.get_model()
        bridge.load_weights(model, hf_model_dir)
    if return_bridge:
        return (model, bridge)
    return model

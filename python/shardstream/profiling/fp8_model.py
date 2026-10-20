"""Independent checkpoint oracle for opt-in native FP8 model benchmarks.

Checkpoint reads, reference quantization, clearing and comparisons are outside
the timed publication. Expected bytes are computed from BF16 checkpoint data,
never copied from the receiver. Scales are checked against an independent amax
reference before using their FP32 rounding to define the byte oracle.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch

from shardstream.integrations.conversion.base import normalize_scale_inv_name
from shardstream.profiling.model import ModelWeightProfile, version_marker_factor


class FP8ModelWeightProfile(ModelWeightProfile):
    def __init__(self, parameters, references, marker_mode="scale", marker_period=None,
                 block_shape=(128, 128)):
        if marker_mode not in ("constant", "scale"):
            raise ValueError("Unknown final norm marker mode")
        version_marker_factor(0, marker_period)
        self.parameters = dict(parameters)
        self.references = dict(references)
        self.marker_mode, self.marker_period = marker_mode, marker_period
        self.block_shape = block_shape
        self.compute = None
        self.norm_name = "model.norm.weight"
        self.encoded = {}
        self.encoding_scales = {}
        self.quantized = {}
        expected_names = set(self.references)
        for name, actual in self.parameters.items():
            if actual.dtype != torch.float8_e4m3fn:
                continue
            br, bc = block_shape
            if actual.ndim != 2 or actual.shape[0] % br or actual.shape[1] % bc:
                raise ValueError(f"FP8 reference requires complete blocks: {name}")
            source = self.references[name]
            if source.dtype != torch.bfloat16 or source.shape != actual.shape:
                raise ValueError(f"No matching BF16 checkpoint reference: {name}")
            scale_name = name + "_scale_inv"
            if scale_name not in self.parameters:
                raise ValueError(f"No FP8 block scales: {name}")
            blocks = source.float().reshape(source.shape[0] // br, br,
                                           source.shape[1] // bc, bc)
            scales = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
            if (self.parameters[scale_name].dtype != torch.float32
                    or self.parameters[scale_name].shape != scales.shape):
                raise ValueError(f"Invalid FP8 scale storage: {scale_name}")
            self.references[scale_name] = scales
            expected_names.add(scale_name)
            self.quantized[name] = scale_name
        if not self.quantized or expected_names != set(self.parameters):
            raise ValueError("Checkpoint references must cover every FP8, scale and mixed parameter")
        if self.norm_name not in self.references or self.parameters[self.norm_name].dtype not in (
            torch.bfloat16, torch.float32
        ):
            raise ValueError("No unquantized final norm version marker")
        for name, actual in self.parameters.items():
            if actual.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float32):
                raise ValueError(f"Unsupported benchmark parameter dtype: {name}")
            if actual.shape != self.references[name].shape:
                raise ValueError(f"Checkpoint reference shape mismatch: {name}")
        self.pointers = {name: p.data_ptr() for name, p in self.parameters.items()}

    @classmethod
    def from_vllm_checkpoint(cls, model, context, checkpoint, marker_mode, marker_period):
        """Build a checkpoint oracle without a ShardStream reader or transport.

        The native baseline is currently qualified for Qwen3-MoE TP1/2 with
        EP8. Expanded expert names already identify complete HF tensors; dense
        parameters use the same vLLM shard coordinates as the model catalog.
        """
        from safetensors import safe_open
        from shardstream.integrations.models.registry import get_infer_weights_converter
        from shardstream.integrations.sharding.vllm import (
            get_vllm_rank_info, get_vllm_sharding_strategy,
        )
        from shardstream.metadata.sharding import ShardingType

        config = context["infer_engine_config"]
        rank = get_vllm_rank_info(context, 0)
        architecture = model.config.architectures[0]
        if architecture != "Qwen3MoeForCausalLM" or rank.tp_size not in (1, 2) or rank.ep_size != 8:
            raise ValueError("Native FP8 checkpoint oracle requires Qwen3-MoE TP1/2 EP8")
        converter = get_infer_weights_converter(
            "vllm", architecture, hf_config=model.config,
            infer_engine_config=config, rank_info=rank,
        )
        strategy = get_vllm_sharding_strategy(
            architecture, config, rank, hf_config=model.config)
        parameters = {key: value for name, parameter in model.named_parameters()
                      for key, value in converter.convert_param(name, parameter)}
        checkpoint = Path(checkpoint)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        metadata = []
        with ExitStack() as stack:
            files = {}
            for key, actual in parameters.items():
                name, is_scale = normalize_scale_inv_name(key)
                offsets = [0] * actual.ndim
                if not is_scale:
                    filename = index[name]
                    if filename not in files:
                        files[filename] = stack.enter_context(safe_open(
                            checkpoint / filename, framework="pt", device="cpu"))
                    shape = tuple(files[filename].get_slice(name).get_shape())
                    if tuple(actual.shape) != shape:
                        kind, dim, count = strategy.get_sharding_strategy(name, rank_info=rank)
                        if kind == ShardingType.TP_SHARDING:
                            physical, world = rank.tp_rank, rank.tp_size
                        elif kind == ShardingType.DP_TP_SHARDING:
                            physical, world = rank.attn_tp_rank, rank.attn_tp_size
                        else:
                            raise ValueError(f"Unsupported native FP8 checkpoint shard: {key}")
                        offsets[dim] = (physical // (world // count)) * actual.shape[dim]
                        if any(actual.shape[d] != shape[d] for d in range(actual.ndim) if d != dim):
                            raise ValueError(f"Native FP8 checkpoint shard shape mismatch: {key}")
                metadata.append(SimpleNamespace(name=key, shards=[SimpleNamespace(
                    shape=tuple(actual.shape), global_offset=tuple(offsets))]))
        profile = cls.from_checkpoint(SimpleNamespace(
            parameters=parameters, current_worker_parameters_meta=metadata,
        ), checkpoint, marker_mode, marker_period)
        profile.native_model = model
        profile.native_pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
        return profile

    def command(self, operation, version):
        if hasattr(self, "native_model") and self.native_pointers != {
            name: p.data_ptr() for name, p in self.native_model.named_parameters()
        }:
            raise AssertionError("A native FP8 model parameter was replaced")
        return super().command(operation, version)

    @classmethod
    def from_checkpoint(cls, reader, checkpoint, marker_mode, marker_period):
        from safetensors import safe_open

        checkpoint = Path(checkpoint)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        parameters = dict(reader.parameters)
        metadata = {m.name: m for m in reader.current_worker_parameters_meta if m.shards}
        if set(metadata) != set(parameters):
            raise ValueError("FP8 checkpoint oracle must cover the complete local catalog")
        references = {}
        with ExitStack() as stack:
            files = {}
            for key, meta in metadata.items():
                name, is_scale = normalize_scale_inv_name(key)
                if is_scale:
                    continue
                if len(meta.shards) != 1:
                    raise ValueError(f"Expected one canonical shard: {key}")
                shard = meta.shards[0]
                filename = index[name]
                if filename not in files:
                    files[filename] = stack.enter_context(safe_open(
                        checkpoint / filename, framework="pt", device="cpu"))
                data = files[filename].get_slice(name)
                shape = data.get_shape()
                offsets = shard.global_offset or (0,) * len(shape)
                slices = tuple(slice(offset, min(offset + size, limit))
                               for offset, size, limit in zip(offsets, shard.shape, shape))
                actual = parameters[key]
                dtype = torch.bfloat16 if actual.dtype == torch.float8_e4m3fn else actual.dtype
                loaded = data[slices].to(device=actual.device, dtype=dtype)
                if tuple(loaded.shape) != tuple(shard.shape):
                    if name not in ("model.embed_tokens.weight", "lm_head.weight"):
                        raise ValueError(f"Unexpected checkpoint padding: {name}")
                    expected = torch.zeros(tuple(shard.shape), device=actual.device, dtype=dtype)
                    expected[tuple(slice(0, size) for size in loaded.shape)].copy_(loaded)
                    loaded = expected
                references[key] = loaded
        profile = cls(parameters, references, marker_mode, marker_period)
        torch.cuda.synchronize()
        return profile

    def _verify_parameters(self, version):
        br, bc = self.block_shape
        for name, scale_name in self.quantized.items():
            actual_scale = self.parameters[scale_name]
            torch.testing.assert_close(actual_scale, self.references[scale_name],
                                       rtol=5e-7, atol=0, msg=scale_name)
            if name not in self.encoded or not torch.equal(actual_scale, self.encoding_scales[name]):
                source = self.references[name]
                blocks = source.float().reshape(source.shape[0] // br, br,
                                               source.shape[1] // bc, bc)
                # Scalar CUDA division can round differently from torch's FP32
                # reciprocal. Round the reciprocal via FP64 independently.
                inverse = (1.0 / actual_scale.double()).float()
                self.encoded[name] = (blocks * inverse[:, None, :, None]).reshape(
                    source.shape).to(torch.float8_e4m3fn).view(torch.uint8)
                self.encoding_scales[name] = actual_scale.detach().clone()
            if not torch.equal(self.parameters[name].view(torch.uint8), self.encoded[name]):
                raise AssertionError(f"FP8 byte mismatch at version {version}: {name}")
        for name, actual in self.parameters.items():
            if name in self.quantized or name in self.quantized.values():
                continue
            expected = self.references[name]
            if name == self.norm_name:
                factor = version_marker_factor(version - 1, self.marker_period)
                expected = (expected * factor if self.marker_mode == "scale"
                            else torch.full_like(expected, factor))
            if not torch.equal(actual, expected):
                raise AssertionError(f"Mixed parameter mismatch at version {version}: {name}")
        return dict(full_checkpoint_reference=True, fp8_bytes_exact=True,
                    scale_rtol=5e-7, scale_atol=0,
                    fp8_elements=sum(self.parameters[n].numel() for n in self.quantized),
                    scale_elements=sum(self.parameters[n].numel() for n in self.quantized.values()),
                    bf16_fp32_elements=sum(p.numel() for n, p in self.parameters.items()
                                           if n not in self.quantized and n not in self.quantized.values()))

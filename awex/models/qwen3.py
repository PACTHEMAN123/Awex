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

"""Registration and copy-only device layouts for dense Qwen3.

Dense and MoE Qwen3 share the layout both inference engines expose: canonical
``self_attn.{q,k,v,o}_proj`` with per-head ``self_attn.{q,k}_norm``, GQA head
grouping in the fused Megatron ``linear_qkv``, and gate/up projections that
the inference engines serve fused while the train side reports them split.
The dense model has no expert parameters. Its device backend additionally
lowers copy-only conversions to stable source spans during initialization.
"""

from math import prod
from typing import Dict, Tuple

import torch

from awex.models.qwen3_moe import (
    SGlangToHFWeightConverterQwen3Moe,
    _build_mcore_converter_qwen3_moe,
)
from awex.transfer.tensor_layout import StaticTensorLayout


def _config_int(config, name: str) -> int:
    value = config.get(name) if isinstance(config, dict) else getattr(config, name)
    return int(value)


def _head_dim(config) -> int:
    value = (
        config.get("head_dim")
        if isinstance(config, dict)
        else getattr(config, "head_dim", None)
    )
    if value:
        return int(value)
    return _config_int(config, "hidden_size") // _config_int(
        config, "num_attention_heads"
    )


def build_qwen3_dense_qkv_layouts(
    parameter: torch.Tensor, hf_config
) -> Dict[str, StaticTensorLayout]:
    """Describe canonical Q/K/V tensors as views into Megatron fused QKV."""

    if not parameter.is_contiguous():
        raise ValueError("Qwen3 dense fused QKV parameter must be contiguous")
    num_heads = _config_int(hf_config, "num_attention_heads")
    num_kv_heads = _config_int(hf_config, "num_key_value_heads")
    if num_heads % num_kv_heads:
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"num_key_value_heads ({num_kv_heads})"
        )
    head_dim = _head_dim(hf_config)
    q_per_group = num_heads // num_kv_heads
    q_rows = q_per_group * head_dim
    group_rows = q_rows + 2 * head_dim
    if parameter.shape[0] % group_rows:
        raise ValueError(
            "Unexpected Qwen3 dense fused QKV rows: "
            f"rows={parameter.shape[0]} group_rows={group_rows}"
        )
    num_groups = parameter.shape[0] // group_rows
    tail_shape = tuple(parameter.shape[1:])

    q_spans = tuple(
        parameter.narrow(0, group * group_rows, q_rows) for group in range(num_groups)
    )
    k_spans = tuple(
        parameter.narrow(0, group * group_rows + q_rows, head_dim)
        for group in range(num_groups)
    )
    v_spans = tuple(
        parameter.narrow(0, group * group_rows + q_rows + head_dim, head_dim)
        for group in range(num_groups)
    )
    return {
        "q": StaticTensorLayout((num_groups * q_rows, *tail_shape), q_spans),
        "k": StaticTensorLayout((num_groups * head_dim, *tail_shape), k_spans),
        "v": StaticTensorLayout((num_groups * head_dim, *tail_shape), v_spans),
    }


def qwen3_dense_span_numels(
    parameter_name: str, shape: Tuple[int, ...], hf_config
) -> Tuple[int, ...]:
    """Return source span sizes for a canonical Q/K/V local shard."""

    projections = {
        ".self_attn.q_proj.": _config_int(hf_config, "num_attention_heads")
        // _config_int(hf_config, "num_key_value_heads")
        * _head_dim(hf_config),
        ".self_attn.k_proj.": _head_dim(hf_config),
        ".self_attn.v_proj.": _head_dim(hf_config),
    }
    block_rows = next(
        (rows for marker, rows in projections.items() if marker in parameter_name),
        None,
    )
    if block_rows is None:
        return ()
    shape = tuple(int(dim) for dim in shape)
    if not shape or shape[0] % block_rows:
        raise ValueError(
            "Canonical Qwen3 dense QKV shard does not align to GQA groups: "
            f"name={parameter_name} shape={shape} block_rows={block_rows}"
        )
    block_numel = block_rows * prod(shape[1:])
    return (block_numel,) * (shape[0] // block_rows)


def annotate_qwen3_dense_transfer_plan(plan, hf_config) -> int:
    """Attach deterministic source span boundaries to a local transfer plan."""

    annotated = 0
    for operations in plan.operations.values():
        for operation in operations:
            span_numels = qwen3_dense_span_numels(
                operation.send_shard_meta.name,
                tuple(operation.send_shard_meta.shape),
                hf_config,
            )
            if span_numels:
                operation.send_tensor_span_numels = span_numels
                annotated += 1
    return annotated


def _build_mcore_converter_qwen3():
    base_converter = _build_mcore_converter_qwen3_moe()

    class McoreToHFWeightConverterQwen3(base_converter):
        def convert_param_to_device_layout(
            self, name: str, parameter: torch.Tensor, vp_stage: int = None
        ):
            canonical_name = self._canonicalize_source_name(name, vp_stage)
            is_qkv_parameter = (
                "self_attention.linear_qkv.weight" in canonical_name
                or "self_attention.linear_qkv.bias" in canonical_name
            )
            if not is_qkv_parameter:
                return self.convert_param(name, parameter, vp_stage=vp_stage)

            layer_number, remaining_name = canonical_name.replace(
                "decoder.layers.", "", 1
            ).split(".", 1)
            if remaining_name not in {
                "self_attention.linear_qkv.weight",
                "self_attention.linear_qkv.bias",
            }:
                raise ValueError(f"Unexpected Qwen3 dense QKV name: {canonical_name}")
            suffix = "weight" if canonical_name.endswith("weight") else "bias"
            layouts = build_qwen3_dense_qkv_layouts(parameter, self.hf_config)
            return [
                (
                    f"model.layers.{layer_number}.self_attn.{projection}_proj.{suffix}",
                    layouts[projection],
                )
                for projection in ("q", "k", "v")
            ]

    return McoreToHFWeightConverterQwen3


CONFIG = {
    "model_name": "Qwen3ForCausalLM",
    "mcore_converter": _build_mcore_converter_qwen3,
    "sglang_converter": SGlangToHFWeightConverterQwen3Moe,
    "vllm_converter": SGlangToHFWeightConverterQwen3Moe,
}

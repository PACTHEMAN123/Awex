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

from typing import List, Tuple

import torch

from awex import logging
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.transfer.tensor_layout import (
    StaticTensorLayout,
    make_blockwise_fp8_layouts,
    make_blockwise_fp8_row_layouts,
)

logger = logging.getLogger(__name__)


class SGlangToHFWeightConverterQwen3Moe(SGlangToHFWeightConverter):
    """SGLang/vLLM -> HF converter for Qwen3-MoE.

    Splits the SGLang fused qkv_proj into canonical q/k/v projections
    (GQA-aware split in the base class) so that inference-side weight
    metadata matches the per-parameter HF names emitted by the Megatron
    train-side converter. Expert parameters (experts.w13_weight /
    experts.w2_weight) are expanded per expert by the base class.
    """

    def _fuse_qkv(self, name: str) -> bool:
        # Train side reports canonical self_attn.{q,k,v}_proj names, so the
        # inference side must unfuse qkv_proj for transfer-plan matching.
        return False

    @staticmethod
    def _block_count(size: int) -> int:
        return (int(size) + 127) // 128

    def _config_int(self, name: str, default=None) -> int:
        value = getattr(self.model_config, name, default)
        if value is None:
            raise ValueError(f"Qwen config is missing {name}")
        return int(value)

    def _head_dim(self) -> int:
        head_dim = getattr(self.model_config, "head_dim", None)
        if head_dim:
            return int(head_dim)
        return self._config_int("hidden_size") // self._config_int(
            "num_attention_heads"
        )

    @staticmethod
    def _orient_block_scale(
        name: str,
        parameter: torch.Tensor,
        output_blocks: int,
        input_blocks: int,
    ) -> torch.Tensor:
        """Expose vLLM input-major FP8 scales in HF output-major order."""

        expected = (int(output_blocks), int(input_blocks))
        transposed = (expected[1], expected[0])
        shape = tuple(int(dim) for dim in parameter.shape)
        if shape == expected:
            return parameter
        if shape == transposed:
            return parameter.transpose(0, 1)
        raise ValueError(
            f"Unexpected 128x128 block scale shape for {name}: got {shape}, "
            f"expected {expected} or input-major {transposed}"
        )

    def _local_attention_rows(self) -> Tuple[int, int]:
        num_heads = self._config_int("num_attention_heads")
        num_kv_heads = self._config_int("num_key_value_heads", num_heads)
        if num_heads % self.tp_size or num_kv_heads % self.tp_size:
            raise ValueError(
                "Qwen attention heads must be divisible by inference TP size: "
                f"heads={num_heads}, kv_heads={num_kv_heads}, tp={self.tp_size}"
            )
        head_dim = self._head_dim()
        return (
            num_heads // self.tp_size * head_dim,
            num_kv_heads // self.tp_size * head_dim,
        )

    def _mlp_intermediate_size(self, name: str) -> int:
        if ".experts" in name:
            return self._config_int("moe_intermediate_size")
        if "shared_expert" in name:
            shared_size = getattr(
                self.model_config, "shared_expert_intermediate_size", None
            )
            if shared_size:
                return int(shared_size)
        return self._config_int("intermediate_size")

    def _local_mlp_intermediate_size(self, name: str) -> int:
        intermediate_size = self._mlp_intermediate_size(name)
        if intermediate_size % self.tp_size:
            raise ValueError(
                f"Qwen intermediate size {intermediate_size} for {name} must be "
                f"divisible by inference TP size {self.tp_size}"
            )
        return intermediate_size // self.tp_size

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if not name.endswith("_scale_inv"):
            return super()._convert_attention_param(name, parameter, layer_number)

        base_name = name[: -len("_scale_inv")]
        hidden_blocks = self._block_count(self._config_int("hidden_size"))
        q_rows, kv_rows = self._local_attention_rows()
        if "qkv_proj" in base_name or "query_key_value" in base_name:
            q_blocks = self._block_count(q_rows)
            kv_blocks = self._block_count(kv_rows)
            scale = self._orient_block_scale(
                name, parameter, q_blocks + 2 * kv_blocks, hidden_blocks
            )
            q_name = base_name.replace("qkv_proj", "q_proj").replace(
                "query_key_value", "q_proj"
            )
            k_name = base_name.replace("qkv_proj", "k_proj").replace(
                "query_key_value", "k_proj"
            )
            v_name = base_name.replace("qkv_proj", "v_proj").replace(
                "query_key_value", "v_proj"
            )
            return [
                (
                    f"{q_name}_scale_inv",
                    scale.narrow(0, 0, q_blocks),
                ),
                (
                    f"{k_name}_scale_inv",
                    scale.narrow(0, q_blocks, kv_blocks),
                ),
                (
                    f"{v_name}_scale_inv",
                    scale.narrow(0, q_blocks + kv_blocks, kv_blocks),
                ),
            ]
        if "o_proj" in base_name or "dense" in base_name:
            scale = self._orient_block_scale(
                name,
                parameter,
                hidden_blocks,
                self._block_count(q_rows),
            )
            return [(name, scale)]
        return super()._convert_attention_param(name, parameter, layer_number)

    def _convert_mlp_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if not name.endswith("_scale_inv"):
            return super()._convert_mlp_param(name, parameter, layer_number)

        base_name = name[: -len("_scale_inv")]
        hidden_blocks = self._block_count(self._config_int("hidden_size"))
        intermediate_blocks = self._block_count(
            self._local_mlp_intermediate_size(base_name)
        )
        if "gate_up_proj" in base_name or "w13_weight" in base_name:
            scale = self._orient_block_scale(
                name, parameter, 2 * intermediate_blocks, hidden_blocks
            )
            gate_name = base_name.replace("gate_up_proj", "gate_proj").replace(
                "w13_weight", "gate_proj.weight"
            )
            up_name = base_name.replace("gate_up_proj", "up_proj").replace(
                "w13_weight", "up_proj.weight"
            )
            return [
                (f"{gate_name}_scale_inv", scale.narrow(0, 0, intermediate_blocks)),
                (
                    f"{up_name}_scale_inv",
                    scale.narrow(0, intermediate_blocks, intermediate_blocks),
                ),
            ]
        if "down_proj" in base_name or "w2_weight" in base_name:
            scale = self._orient_block_scale(
                name, parameter, hidden_blocks, intermediate_blocks
            )
            converted_name = base_name.replace("w2_weight", "down_proj.weight")
            return [(f"{converted_name}_scale_inv", scale)]
        return super()._convert_mlp_param(name, parameter, layer_number)

    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        if name.endswith(".mlp.gate.weight_scale_inv"):
            parameter = self._orient_block_scale(
                name,
                parameter,
                self._block_count(self._config_int("num_experts")),
                self._block_count(self._config_int("hidden_size")),
            )
        return super().convert_param(name, parameter)

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        # Qwen3 uses self_attn.{q,k}_norm; the base class only recognizes
        # the bailing-style {query,key}_layernorm names.
        if "q_norm" in name or "k_norm" in name:
            return [(name, parameter)]
        return super()._convert_layer_norm_param(name, parameter, layer_number)


def _build_mcore_converter_qwen3_moe():
    # Lazily import Megatron converter to avoid MindSpeed patching in
    # vLLM-only paths (same pattern as ling.py).
    from awex.converter.mcore_converter import McoreToHFWeightConverter

    class McoreToHFWeightConverterQwen3Moe(McoreToHFWeightConverter):
        """Stock HF/sglang qwen3_moe serves canonical attention names
        (self_attn.{q,k,v,o}_proj + self_attn.{q,k}_norm), so keep them
        verbatim instead of the bailing-flavored renames, and split the
        Megatron fused linear_qkv with GQA-aware group strides — the base
        equal-thirds split only holds when q/k/v head counts match.
        """

        def __init__(self, hf_config, rank_info, infer_conf, tf_config):
            super().__init__(hf_config, rank_info, infer_conf, tf_config)
            infer_hf_config = infer_conf.get("hf_config", {})
            quantization_config = self._read_cfg_value(
                infer_hf_config, "quantization_config", {}
            ) or {}
            quant_method = self._read_cfg_value(
                quantization_config, "quant_method", None
            )
            block_size = self._read_cfg_value(
                quantization_config, "weight_block_size", None
            )
            activation_scheme = self._read_cfg_value(
                quantization_config, "activation_scheme", "dynamic"
            )
            self.blockwise_fp8 = bool(
                quant_method
                and "fp8" in str(quant_method).lower()
                and tuple(block_size or ()) == (128, 128)
            )
            if self.blockwise_fp8 and activation_scheme != "dynamic":
                raise ValueError(
                    "Qwen 128 x 128 FP8 requires dynamic activation scaling"
                )
            if self.blockwise_fp8:
                logger.info(
                    "Qwen converter enabled 128 x 128 block-wise FP8 outputs"
                )

        def _uses_blockwise_fp8(self, name: str, parameter) -> bool:
            shape = tuple(int(dim) for dim in parameter.shape)
            is_attention_weight = name.endswith(
                (
                    ".self_attn.q_proj.weight",
                    ".self_attn.k_proj.weight",
                    ".self_attn.v_proj.weight",
                    ".self_attn.o_proj.weight",
                )
            )
            is_mlp_weight = ".mlp." in name and name.endswith(
                (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
            )
            is_router_weight = name.endswith(".mlp.gate.weight")
            return (
                self.blockwise_fp8
                and len(shape) == 2
                and (is_attention_weight or is_mlp_weight or is_router_weight)
            )

        def _apply_blockwise_fp8(self, converted):
            outputs = []
            index = 0
            while index < len(converted):
                name, parameter = converted[index]
                if not self._uses_blockwise_fp8(name, parameter):
                    outputs.append((name, parameter))
                    index += 1
                    continue
                if index + 1 < len(converted):
                    next_name, next_parameter = converted[index + 1]
                    gate_suffix = ".gate_proj.weight"
                    up_suffix = ".up_proj.weight"
                    can_share_fc1 = (
                        name.endswith(gate_suffix)
                        and next_name == name[: -len(gate_suffix)] + up_suffix
                        and isinstance(parameter, torch.Tensor)
                        and isinstance(next_parameter, torch.Tensor)
                        and parameter.is_contiguous()
                        and next_parameter.is_contiguous()
                        and parameter.shape == next_parameter.shape
                        and parameter.untyped_storage().data_ptr()
                        == next_parameter.untyped_storage().data_ptr()
                        and parameter.data_ptr()
                        + parameter.numel() * parameter.element_size()
                        == next_parameter.data_ptr()
                    )
                    if can_share_fc1:
                        rows, cols = (int(dim) for dim in parameter.shape)
                        source = StaticTensorLayout(
                            (rows * 2, cols), (parameter, next_parameter)
                        )
                        (gate_weight, gate_scale), (up_weight, up_scale) = (
                            make_blockwise_fp8_row_layouts(
                                source, ((0, rows), (rows, rows * 2))
                            )
                        )
                        outputs.extend(
                            (
                                (name, gate_weight),
                                (f"{name}_scale_inv", gate_scale),
                                (next_name, up_weight),
                                (f"{next_name}_scale_inv", up_scale),
                            )
                        )
                        index += 2
                        continue
                source = parameter
                if isinstance(parameter, torch.Tensor):
                    source = StaticTensorLayout(
                        tuple(int(dim) for dim in parameter.shape), (parameter,)
                    )
                weight, scale = make_blockwise_fp8_layouts(source)
                outputs.append((name, weight))
                outputs.append((f"{name}_scale_inv", scale))
                index += 1
            return outputs

        def _fuse_qkv(self, name: str) -> bool:
            return False

        @staticmethod
        def _normalize_attn_name(name: str) -> str:
            return name

        def _gqa_head_dim(self) -> int:
            head_dim = getattr(self.hf_config, "head_dim", None)
            if head_dim:
                return int(head_dim)
            return int(self.hf_config.hidden_size // self.hf_config.num_attention_heads)

        def _split_gqa_qkv(
            self, parameter: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hf = self.hf_config
            head_dim = self._gqa_head_dim()
            attn_tp = max(1, int(getattr(self.rank_info, "attn_tp_size", 1)))
            if hf.num_key_value_heads % attn_tp != 0:
                raise ValueError(
                    f"num_key_value_heads ({hf.num_key_value_heads}) must be "
                    f"divisible by attn_tp_size ({attn_tp})"
                )
            if hf.num_attention_heads % hf.num_key_value_heads != 0:
                raise ValueError(
                    f"num_attention_heads ({hf.num_attention_heads}) must be "
                    f"divisible by num_key_value_heads "
                    f"({hf.num_key_value_heads})"
                )
            num_groups = hf.num_key_value_heads // attn_tp
            q_per_group = hf.num_attention_heads // hf.num_key_value_heads
            group_rows = (q_per_group + 2) * head_dim
            expected_rows = num_groups * group_rows
            if parameter.shape[0] != expected_rows:
                raise ValueError(
                    "Unexpected linear_qkv rows for GQA split: "
                    f"got {parameter.shape[0]}, expected {expected_rows} "
                    f"(kv_heads={hf.num_key_value_heads}, attn_tp={attn_tp}, "
                    f"q_per_group={q_per_group}, head_dim={head_dim})"
                )
            blocks = parameter.reshape(num_groups, group_rows, *parameter.shape[1:])
            q_rows = q_per_group * head_dim
            q = blocks[:, :q_rows].reshape(num_groups * q_rows, *parameter.shape[1:])
            k = blocks[:, q_rows : q_rows + head_dim].reshape(
                num_groups * head_dim, *parameter.shape[1:]
            )
            v = blocks[:, q_rows + head_dim :].reshape(
                num_groups * head_dim, *parameter.shape[1:]
            )
            return q.contiguous(), k.contiguous(), v.contiguous()

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
                raise ValueError(f"Unexpected Qwen3-MoE QKV name: {canonical_name}")
            suffix = "weight" if canonical_name.endswith("weight") else "bias"
            # Import lazily because qwen3.py reuses this converter factory.
            from awex.models.qwen3 import build_qwen3_dense_qkv_layouts

            layouts = build_qwen3_dense_qkv_layouts(parameter, self.hf_config)
            converted = [
                (
                    f"model.layers.{layer_number}.self_attn.{projection}_proj.{suffix}",
                    layouts[projection],
                )
                for projection in ("q", "k", "v")
            ]
            return self._apply_blockwise_fp8(converted)

        def convert_param(
            self, name: str, parameter: torch.Tensor, vp_stage: int = None
        ):
            canonical_name = self._canonicalize_source_name(name, vp_stage)
            if self.blockwise_fp8 and canonical_name.endswith(
                "self_attention.linear_qkv.weight"
            ):
                layer_number, _ = canonical_name.replace(
                    "decoder.layers.", "", 1
                ).split(".", 1)
                from awex.models.qwen3 import build_qwen3_dense_qkv_layouts

                layouts = build_qwen3_dense_qkv_layouts(parameter, self.hf_config)
                return self._apply_blockwise_fp8(
                    [
                        (
                            f"model.layers.{layer_number}.self_attn."
                            f"{projection}_proj.weight",
                            layouts[projection],
                        )
                        for projection in ("q", "k", "v")
                    ]
                )
            return self._apply_blockwise_fp8(
                super().convert_param(name, parameter, vp_stage=vp_stage)
            )

        def _convert_attention_param(
            self, name: str, parameter: torch.Tensor, layer_number: str
        ) -> List[Tuple[str, torch.Tensor]]:
            if "self_attention.linear_qkv.weight" in name or (
                "self_attention.linear_qkv.bias" in name
            ):
                suffix = "weight" if name.endswith("weight") else "bias"
                q, k, v = self._split_gqa_qkv(parameter)
                return [
                    (f"self_attn.q_proj.{suffix}", q),
                    (f"self_attn.k_proj.{suffix}", k),
                    (f"self_attn.v_proj.{suffix}", v),
                ]
            return super()._convert_attention_param(name, parameter, layer_number)

    return McoreToHFWeightConverterQwen3Moe


CONFIG = {
    "model_name": "Qwen3MoeForCausalLM",
    "mcore_converter": _build_mcore_converter_qwen3_moe,
    "sglang_converter": SGlangToHFWeightConverterQwen3Moe,
    "vllm_converter": SGlangToHFWeightConverterQwen3Moe,
}

"""Canonical writable vLLM tensor views for the five-model comparison.

These model adapters expose native Awex metadata. Transport, transfer planning,
NCCL packing and publication scheduling remain the upstream Awex implementations.
"""
from typing import List, Tuple
import torch
from awex.converter.canonical_vllm import FusedWeightConverter
from awex.converter.weights_converter import append_scale_inv, normalize_scale_inv_name
from awex.sharding.param_sharding import ShardingStrategy
from awex.models.qwen3_moe import _build_mcore_converter_qwen3_moe

class Qwen3ShardingStrategy(ShardingStrategy):
    """Represent replicated GQA KV heads as logical TP replicas."""

    def __init__(self, *args, hf_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(hf_config, dict):
            num_kv_heads = hf_config.get("num_key_value_heads")
        else:
            num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads else None

    def get_sharding_strategy(self, parameter_name, **kwargs):
        (sharding_type, sharding_dim, num_shards) = super().get_sharding_strategy(
            parameter_name, **kwargs
        )
        is_kv_projection = (
            ".self_attn.k_proj." in parameter_name
            or ".self_attn.v_proj." in parameter_name
        )
        if (
            self.engine_name == "vllm"
            and is_kv_projection
            and self.num_kv_heads
            and (num_shards > self.num_kv_heads)
        ):
            return (sharding_type, sharding_dim, self.num_kv_heads)
        return (sharding_type, sharding_dim, num_shards)


class Qwen3FusedWeightConverter(FusedWeightConverter):
    """vLLM -> HF converter for Qwen3-MoE.

    Splits the vLLM fused qkv_proj into canonical q/k/v projections
    (GQA-aware split in the base class) so that inference-side weight
    metadata matches the per-parameter HF names emitted by the Megatron
    train-side converter. Expert parameters (experts.w13_weight /
    experts.w2_weight) are expanded per expert by the base class.
    """

    def _fuse_qkv(self, name: str) -> bool:
        return False

    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        name = name.replace(".experts.routed_experts.", ".experts.")
        (base_name, has_scale_inv) = normalize_scale_inv_name(name)
        return [
            (append_scale_inv(converted_name, has_scale_inv), tensor)
            for (converted_name, tensor) in super().convert_param(base_name, parameter)
        ]

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if "q_norm" in name or "k_norm" in name:
            return [(name, parameter)]
        return super()._convert_layer_norm_param(name, parameter, layer_number)



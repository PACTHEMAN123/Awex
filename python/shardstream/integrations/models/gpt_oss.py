"""BF16 GPT-OSS layouts, including interleaved expert rows and biases."""

import re
from types import SimpleNamespace

import torch

from shardstream.integrations.models.qwen3_moe import (
    Qwen3FusedWeightConverter,
    Qwen3ShardingStrategy,
    _build_mcore_converter_qwen3_moe,
)
from shardstream.metadata.sharding import ShardingType


class GPTOSSShardingStrategy(Qwen3ShardingStrategy):
    def get_sharding_strategy(self, parameter_name, **kwargs):
        if ".mlp.router." in parameter_name or parameter_name.endswith("o_proj.bias"):
            return ShardingType.NO_SHARDING, 0, 1
        if ".experts." in parameter_name:
            if parameter_name.endswith("down_proj.bias"):
                if self.ep_size > 1:
                    return ShardingType.EP_SHARDING, 0, self.ep_size
                return ShardingType.NO_SHARDING, 0, 1
            return self.get_expert_sharding_strategy(parameter_name, **kwargs)
        if parameter_name.endswith("self_attn.sinks"):
            return self.get_attention_sharding_strategy(parameter_name, **kwargs)
        return super().get_sharding_strategy(parameter_name, **kwargs)


class GPTOSSVLLMWeightConverter(Qwen3FusedWeightConverter):
    def convert_param(self, name, parameter):
        name = name.replace("model.embedding.", "model.embed_tokens.")
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".experts.routed_experts.", ".experts.")
        if ".mlp.router." in name or name.endswith("self_attn.sinks"):
            return [(name, parameter)]
        if ".experts." not in name:
            return super().convert_param(name, parameter)
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in {"w13_weight", "w13_bias", "w2_weight", "w2_bias"}:
            raise NotImplementedError(
                f"GPT-OSS transfer requires BF16 expert weights: {name}"
            )
        if not parameter.is_floating_point():
            raise ValueError("GPT-OSS transfer requires a dequantized BF16 checkpoint")
        if suffix == "w2_bias" and self.ep_size == 1 and self.tp_rank != 0:
            # vLLM adds the down-projection bias before TP reduction, so only
            # TP rank zero owns a nonzero bias. Other ranks must stay zero.
            if torch.count_nonzero(parameter).item():
                raise ValueError("GPT-OSS down bias must be zero outside TP rank zero")
            return []
        converted = []
        for local_id, expert in enumerate(parameter):
            expert_id = local_id + self.ep_rank * parameter.shape[0]
            prefix = name.rsplit(".", 1)[0] + f".{expert_id}."
            if suffix.startswith("w13"):
                if expert.shape[0] % 2:
                    raise ValueError("GPT-OSS gate/up row count must be even")
                kind = "weight" if suffix.endswith("weight") else "bias"
                # Writable strided views preserve vLLM's alternating G/U rows.
                converted.extend(
                    [
                        (prefix + f"gate_proj.{kind}", expert[::2]),
                        (prefix + f"up_proj.{kind}", expert[1::2]),
                    ]
                )
            else:
                kind = "weight" if suffix.endswith("weight") else "bias"
                converted.append((prefix + f"down_proj.{kind}", expert))
        return converted


def _build_mcore_converter_gpt_oss():
    base = _build_mcore_converter_qwen3_moe()

    class GPTOSSMcoreWeightConverter(base):
        def __init__(self, hf_config, rank_info, infer_conf, tf_config):
            # The base expert converter uses the generic num_experts field.
            config = SimpleNamespace(**vars(hf_config))
            config.num_experts = hf_config.num_local_experts
            super().__init__(config, rank_info, infer_conf, tf_config)

        def _convert_attention_param(self, name, parameter, layer_number):
            if name == "self_attention.core_attention.softmax_offset":
                return [("self_attn.sinks", parameter)]
            return super()._convert_attention_param(name, parameter, layer_number)

        def _convert_mlp_param(self, name, parameter, layer_number):
            if name in {"mlp.router.weight", "mlp.router.bias"}:
                return [(name, parameter)]
            if ".experts." in name:
                local = re.search(r"local_experts\.(\d+)\.", name)
                numbered = re.search(r"\.(?:weight|bias)(\d+)$", name)
                match = local or numbered
                if match is None:
                    raise NotImplementedError(
                        f"Unsupported GPT-OSS expert parameter: {name}"
                    )
                expert_id = int(match[1]) + self.rank_info.ep_rank * (
                    self.hf_config.num_experts // self.rank_info.ep_size
                )
                return [
                    (f"mlp.experts.{expert_id}.{target}", tensor)
                    for target, tensor in self._convert_linear(name, parameter)
                ]
            return super()._convert_mlp_param(name, parameter, layer_number)

        def _convert_gate(self, name, parameter):
            return "mlp.router.weight", parameter

    return GPTOSSMcoreWeightConverter


CONFIG = {
    "model_name": "GptOssForCausalLM",
    "sharding_strategy": GPTOSSShardingStrategy,
    "mcore_converter": _build_mcore_converter_gpt_oss,
    "vllm_converter": GPTOSSVLLMWeightConverter,
}

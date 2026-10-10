"""GLM-4.7-Flash MLA, expert and trained MTP weights for vLLM."""

import re
from types import SimpleNamespace

import torch

from shardstream.integrations.models.qwen3_moe import (
    Qwen3FusedWeightConverter,
    _build_mcore_converter_qwen3_moe,
)
from shardstream.metadata.sharding import ShardingStrategy, ShardingType


class GLM47FlashShardingStrategy(ShardingStrategy):
    def __init__(self, *args, hf_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_config = hf_config

    def get_sharding_strategy(self, parameter_name, **kwargs):
        if ".shared_experts." in parameter_name:
            return self.get_mlp_sharding_strategy(parameter_name, **kwargs)
        if ".experts." in parameter_name:
            return self.get_expert_sharding_strategy(parameter_name, **kwargs)
        if parameter_name.endswith(("q_a_proj.weight", "kv_a_proj_with_mqa.weight")):
            if self.engine_name == "vllm":
                return ShardingType.NO_SHARDING, 0, 1
            config = self.hf_config
            get = (
                config.get
                if isinstance(config, dict)
                else lambda key: getattr(config, key)
            )
            expected = (
                get("q_lora_rank")
                if parameter_name.endswith("q_a_proj.weight")
                else (get("kv_lora_rank") + get("qk_rope_head_dim"))
            )
            rows = kwargs["param_meta"]["shape"][0]
            if rows == expected:
                return ShardingType.NO_SHARDING, 0, 1
            if rows * self.tp_size != expected:
                raise ValueError(
                    f"Unexpected MLA down-projection shape: {parameter_name}"
                )
            return ShardingType.TP_SHARDING, 0, self.tp_size
        if parameter_name.endswith("eh_proj.weight"):
            return self.get_attention_sharding_strategy(parameter_name, **kwargs)
        if ".mlp.gate." in parameter_name:
            return ShardingType.NO_SHARDING, 0, 1
        return super().get_sharding_strategy(parameter_name, **kwargs)


class GLM47FlashVLLMWeightConverter(Qwen3FusedWeightConverter):
    def __init__(self, model_config, infer_engine_config, rank_info):
        config = SimpleNamespace(**vars(model_config))
        config.num_experts = model_config.n_routed_experts
        # MLA has no GQA KV-head field; this base field is unused by this adapter.
        config.num_key_value_heads = getattr(model_config, "num_key_value_heads", 1)
        super().__init__(config, infer_engine_config, rank_info)

    def convert_param(self, name, parameter):
        name = name.replace(".mtp_block.", ".")
        name = name.replace(".experts.routed_experts.", ".experts.")
        if ".mla_attn." in name:
            # W_UK/W_UV are derived from kv_b_proj, rebuilt after every update.
            return []
        if name.endswith("self_attn.fused_qkv_a_proj.weight"):
            q_rows = self.model_config.q_lora_rank
            kv_rows = (
                self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim
            )
            if parameter.shape[0] != q_rows + kv_rows:
                raise ValueError("Unexpected GLM fused MLA down-projection shape")
            return [
                (name.replace("fused_qkv_a_proj", "q_a_proj"), parameter[:q_rows]),
                (
                    name.replace("fused_qkv_a_proj", "kv_a_proj_with_mqa"),
                    parameter[q_rows:],
                ),
            ]
        if (
            ".self_attn." in name
            or ".mlp.gate." in name
            or name.endswith(
                (
                    "enorm.weight",
                    "hnorm.weight",
                    "eh_proj.weight",
                    "shared_head.norm.weight",
                    "shared_head.head.weight",
                    "embed_tokens.weight",
                )
            )
        ):
            return [(name, parameter)]
        return super().convert_param(name, parameter)

    def post_update(self, model):
        # vLLM uses prefer_copy=True when rebuilding MLA absorbed weights,
        # preserving buffers already referenced by CUDA graphs.
        activation_dtype = next(model.parameters()).dtype
        for module in model.modules():
            if module.__class__.__name__ == "MLAAttention":
                module.process_weights_after_loading(activation_dtype)


class GLM47FlashTransferModel(torch.nn.Module):
    """Present the main model and its MTP draft as one transfer target."""

    def __init__(self, model, draft, num_hidden_layers):
        super().__init__()
        self.target = model
        self.draft = draft
        self.config = getattr(model, "config", None)
        self.num_hidden_layers = num_hidden_layers

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        for role, model in [("target", self.target), ("draft", self.draft)]:
            for name, parameter in model.named_parameters(recurse=recurse):
                identity = (
                    parameter.data_ptr(),
                    tuple(parameter.shape),
                    parameter.dtype,
                )
                if remove_duplicate and identity in seen:
                    continue
                seen.add(identity)
                if role == "draft":
                    if name == "model.embed_tokens.weight":
                        name = (
                            f"model.layers.{self.num_hidden_layers}.embed_tokens.weight"
                        )
                    elif name == "lm_head.weight":
                        name = f"model.layers.{self.num_hidden_layers}.shared_head.head.weight"
                yield (prefix + "." + name if prefix else name), parameter


def _build_mcore_converter_glm47_flash():
    base = _build_mcore_converter_qwen3_moe()

    class GLM47FlashMcoreWeightConverter(base):
        _mla_names = {
            "linear_q_down_proj.weight": "q_a_proj.weight",
            "linear_q_up_proj.weight": "q_b_proj.weight",
            "linear_q_up_proj.layer_norm_weight": "q_a_layernorm.weight",
            "q_layernorm.weight": "q_a_layernorm.weight",
            "linear_kv_down_proj.weight": "kv_a_proj_with_mqa.weight",
            "linear_kv_up_proj.weight": "kv_b_proj.weight",
            "linear_kv_up_proj.layer_norm_weight": "kv_a_layernorm.weight",
            "kv_layernorm.weight": "kv_a_layernorm.weight",
            "linear_proj.weight": "o_proj.weight",
        }

        def __init__(self, hf_config, rank_info, infer_conf, tf_config):
            config = SimpleNamespace(**vars(hf_config))
            config.num_experts = hf_config.n_routed_experts
            super().__init__(
                config, rank_info, {**infer_conf, "router_dtype": "fp32"}, tf_config
            )

        def _convert_attention_param(self, name, parameter, layer_number):
            suffix = name.removeprefix("self_attention.")
            if suffix in self._mla_names:
                return [("self_attn." + self._mla_names[suffix], parameter)]
            return super()._convert_attention_param(name, parameter, layer_number)

        def _convert_expert_bias_param(self, name, parameter, layer_number):
            return "mlp.gate.e_score_correction_bias", parameter.float()

        def convert_param_to_device_layout(self, name, parameter, vp_stage=None):
            return self.convert_param(name, parameter, vp_stage)

        def convert_param(self, name, parameter, vp_stage=None):
            name = name.replace("module.", "")
            if name.startswith("mtp.layers."):
                _, _, local_layer, suffix = name.split(".", 3)
                layer = self.hf_config.num_hidden_layers + int(local_layer)
                prefix = f"model.layers.{layer}."
                direct = {
                    "enorm.weight": "enorm.weight",
                    "hnorm.weight": "hnorm.weight",
                    "eh_proj.weight": "eh_proj.weight",
                    "final_layernorm.weight": "shared_head.norm.weight",
                }
                if suffix in direct:
                    return [(prefix + direct[suffix], parameter)]
                suffix = re.sub(r"^(?:mtp_model_layer|transformer_layer)\.", "", suffix)
                # This synthetic decoder id is already global; bypass PP remapping.
                previous = self._pp_stage_layer_id_map
                try:
                    self._pp_stage_layer_id_map = {}
                    return super().convert_param(
                        f"decoder.layers.{layer}.{suffix}", parameter
                    )
                finally:
                    self._pp_stage_layer_id_map = previous
            converted = super().convert_param(name, parameter, vp_stage)
            aliases = []
            for target, tensor in converted:
                if target in {"model.embed_tokens.weight", "lm_head.weight"}:
                    for layer in range(
                        getattr(self.hf_config, "num_nextn_predict_layers", 0)
                    ):
                        suffix = (
                            "embed_tokens.weight"
                            if target.startswith("model.")
                            else "shared_head.head.weight"
                        )
                        aliases.append(
                            (
                                f"model.layers.{self.hf_config.num_hidden_layers + layer}.{suffix}",
                                tensor,
                            )
                        )
            return converted + aliases

    return GLM47FlashMcoreWeightConverter


CONFIG = {
    "model_name": "Glm4MoeLiteForCausalLM",
    "sharding_strategy": GLM47FlashShardingStrategy,
    "mcore_converter": _build_mcore_converter_glm47_flash,
    "vllm_converter": GLM47FlashVLLMWeightConverter,
}

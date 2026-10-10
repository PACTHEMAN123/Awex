"""Qwen2.5-VL language TP and replicated Megatron vision parameters."""

from shardstream.integrations.models.qwen3_moe import (
    Qwen3FusedWeightConverter,
    Qwen3ShardingStrategy,
    _build_mcore_converter_qwen3_moe,
)
from shardstream.metadata.sharding import ShardingType


class Qwen25VLShardingStrategy(Qwen3ShardingStrategy):
    def get_sharding_strategy(self, parameter_name, **kwargs):
        if parameter_name.startswith("model.visual."):
            # Megatron Bridge's Qwen2.5-VL provider uses a Transformers vision
            # tower replicated across training ranks. The selected vLLM TP1
            # recipe also has a complete vision tower on every rollout rank.
            if self.engine_name == "mcore" or self.tp_size == 1:
                return ShardingType.NO_SHARDING, 0, 1
            raise NotImplementedError("Qwen2.5-VL recipe requires vLLM vision TP1")
        return super().get_sharding_strategy(parameter_name, **kwargs)


class Qwen25VLVLLMWeightConverter(Qwen3FusedWeightConverter):
    def __init__(self, model_config, infer_engine_config, rank_info):
        super().__init__(
            getattr(model_config, "text_config", model_config),
            infer_engine_config,
            rank_info,
        )

    def convert_param(self, name, parameter):
        name = name.replace("language_model.model.", "model.")
        name = name.replace("language_model.lm_head.", "lm_head.")
        name = name.replace("model.language_model.", "model.")
        if name.startswith("visual."):
            name = "model." + name
        if name.startswith("model.visual."):
            name = name.replace(".attn.qkv_proj.", ".attn.qkv.")
            if ".mlp.gate_up_proj." in name:
                return self._split_gate_up(name, parameter)
            return [(name, parameter)]
        return super().convert_param(name, parameter)


def _build_mcore_converter_qwen25_vl():
    base = _build_mcore_converter_qwen3_moe()

    class Qwen25VLMcoreWeightConverter(base):
        def __init__(self, hf_config, rank_info, infer_conf, tf_config):
            super().__init__(
                getattr(hf_config, "text_config", hf_config),
                rank_info,
                infer_conf,
                tf_config,
            )

        def convert_param(self, name, parameter, vp_stage=None):
            name = name.replace("module.", "")
            if name.startswith("visual."):
                return [("model." + name, parameter)]
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            return super().convert_param(name, parameter, vp_stage=vp_stage)

        def convert_param_to_device_layout(self, name, parameter, vp_stage=None):
            # Strip the VLM prefix before the base Qwen GQA span builder.
            name = name.replace("module.", "")
            if name.startswith("visual."):
                return [("model." + name, parameter)]
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            return super().convert_param_to_device_layout(name, parameter, vp_stage)

    return Qwen25VLMcoreWeightConverter


CONFIG = {
    "model_name": "Qwen2_5_VLForConditionalGeneration",
    "sharding_strategy": Qwen25VLShardingStrategy,
    "mcore_converter": _build_mcore_converter_qwen25_vl,
    "vllm_converter": Qwen25VLVLLMWeightConverter,
}

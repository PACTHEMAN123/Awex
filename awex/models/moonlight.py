"""Moonlight's DeepSeek-V3 MLA layout with an uncompressed query projection."""

from awex.models.glm4_moe_lite import (
    GLM47FlashShardingStrategy,
    GLM47FlashVLLMWeightConverter,
    _build_mcore_converter_glm47_flash,
)


class MoonlightShardingStrategy(GLM47FlashShardingStrategy):
    """Use replicated MLA down projections and TP-sharded shared experts."""


class MoonlightVLLMWeightConverter(GLM47FlashVLLMWeightConverter):
    """Retain writable MLA targets and rebuild absorbed weights after updates."""


def _build_mcore_converter_moonlight():
    base = _build_mcore_converter_glm47_flash()

    class MoonlightMcoreWeightConverter(base):
        # Moonlight has q_lora_rank=None: Megatron emits linear_q_proj instead
        # of the Q-LoRA down/up pair used by GLM. The remaining MLA, shared
        # expert, routed expert and correction-bias mappings are identical.
        _mla_names = {**base._mla_names, "linear_q_proj.weight": "q_proj.weight"}

    return MoonlightMcoreWeightConverter


CONFIG = {
    "model_name": "DeepseekV3ForCausalLM",
    "sharding_strategy": MoonlightShardingStrategy,
    "mcore_converter": _build_mcore_converter_moonlight,
    "vllm_converter": MoonlightVLLMWeightConverter,
}

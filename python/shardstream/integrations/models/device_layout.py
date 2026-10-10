"""Models whose fixed transfer plans bind original parameter storage."""

STATIC_DEVICE_LAYOUT_ARCHITECTURES = {
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "GptOssForCausalLM",
    "Glm4MoeLiteForCausalLM",
}


def annotate_device_transfer_plan(plan, architecture, hf_config):
    if architecture.startswith("Qwen3_5"):
        from .qwen3_5 import annotate_qwen35_transfer_plan

        return annotate_qwen35_transfer_plan(plan, hf_config)
    if architecture == "Glm4MoeLiteForCausalLM":
        return 0
    from .qwen3 import annotate_qwen3_dense_transfer_plan

    return annotate_qwen3_dense_transfer_plan(
        plan,
        hf_config.get("text_config", hf_config)
        if isinstance(hf_config, dict)
        else getattr(hf_config, "text_config", hf_config),
    )

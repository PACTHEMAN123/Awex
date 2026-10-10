"""Writable vLLM targets and recipe-specific Megatron name contracts."""

from types import SimpleNamespace as NS

import pytest
import torch
from shardstream.integrations.models.glm4_moe_lite import (
    GLM47FlashShardingStrategy,
    GLM47FlashTransferModel,
    GLM47FlashVLLMWeightConverter,
    _build_mcore_converter_glm47_flash,
)
from shardstream.integrations.models.gpt_oss import (
    GPTOSSShardingStrategy,
    GPTOSSVLLMWeightConverter,
    _build_mcore_converter_gpt_oss,
)
from shardstream.integrations.models.qwen2_5_vl import (
    Qwen25VLVLLMWeightConverter,
    _build_mcore_converter_qwen25_vl,
)
from shardstream.integrations.models.qwen3_5 import (
    Qwen3_5VLLMWeightConverter,
    _Qwen3_5Layout,
)
from shardstream.integrations.models.registry import get_infer_weights_converter
from shardstream.metadata.sharding import ShardingType


def config():
    return NS(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        hidden_size=8,
        num_local_experts=8,
        num_experts=8,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )


def infer(converter, tp=1, tp_rank=0, ep=1, ep_rank=0):
    return converter(
        config(), NS(tp_size=tp, ep_size=ep), NS(tp_rank=tp_rank, ep_rank=ep_rank)
    )


def mcore(factory, ep=1, ep_rank=0):
    return factory()(
        config(),
        NS(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            ep_size=ep,
            ep_rank=ep_rank,
            pp_size=1,
            pp_rank=0,
        ),
        {"infer_atten_tp_size": 1},
        NS(),
    )


@pytest.mark.parametrize(
    "architecture,converter",
    [
        ("Qwen3_5ForConditionalGeneration", Qwen3_5VLLMWeightConverter),
        ("Qwen2_5_VLForConditionalGeneration", Qwen25VLVLLMWeightConverter),
        ("GptOssForCausalLM", GPTOSSVLLMWeightConverter),
    ],
)
def test_registry_selects_vllm_model_converter(architecture, converter):
    assert isinstance(
        get_infer_weights_converter(
            "vllm",
            architecture,
            config(),
            NS(tp_rank=0, ep_rank=0),
            NS(tp_size=1, ep_size=1),
        ),
        converter,
    )


def test_gated_attention_reorders_queries_and_gates_per_head():
    # Each Megatron group is Q0,Q1,G0,G1,K,V (two rows per head).
    source = torch.arange(24).reshape(24, 1)
    result = _Qwen3_5Layout.pack_output_gated_qkv(source, config(), 2)
    expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11]
    assert result[:, 0].tolist() == expected + [x + 12 for x in expected]


def test_gdn_rank_local_categories_reshard_without_interleaving_b_a():
    # Q,K each have 2 local rows; V,Z each have 4; B,A each have 2.
    source = torch.arange(32).reshape(32, 1)
    qkvz, ba = _Qwen3_5Layout.pack_gdn_input(source, config(), 2, 1)
    expected = [
        0,
        1,
        16,
        17,
        2,
        3,
        18,
        19,
        4,
        5,
        6,
        7,
        20,
        21,
        22,
        23,
        8,
        9,
        10,
        11,
        24,
        25,
        26,
        27,
    ]
    assert qkvz[:, 0].tolist() == expected
    assert ba[:, 0].tolist() == [12, 13, 28, 29, 14, 15, 30, 31]


@pytest.mark.parametrize(
    "name",
    [
        "language_model.model.layers.3.self_attn.qkv_proj.weight",
        "language_model.model.layers.3.linear_attn.in_proj_qkvz.weight",
        "language_model.model.layers.3.linear_attn.in_proj_ba.weight",
        "visual.blocks.0.attn.qkv.weight",
    ],
)
def test_qwen35_targets_keep_writable_native_storage(name):
    tensor = torch.zeros(8, 4)
    [(target, view)] = infer(Qwen3_5VLLMWeightConverter).convert_param(name, tensor)
    assert target.startswith("model.")
    assert view.data_ptr() == tensor.data_ptr()
    view.fill_(5)
    assert torch.equal(tensor, torch.full_like(tensor, 5))


def test_qwen25_vision_gated_mlp_matches_replicated_train_names():
    train = mcore(_build_mcore_converter_qwen25_vl)
    gate, up = torch.ones(4, 8), torch.full((4, 8), 2.0)
    fused = torch.cat([gate, up])
    targets = dict(
        infer(Qwen25VLVLLMWeightConverter).convert_param(
            "visual.blocks.0.mlp.gate_up_proj.weight", fused
        )
    )
    for projection, expected in [("gate", gate), ("up", up)]:
        name = f"visual.blocks.0.mlp.{projection}_proj.weight"
        [(canonical, value)] = train.convert_param(name, expected)
        assert torch.equal(targets[canonical], value)
    targets["model.visual.blocks.0.mlp.up_proj.weight"].fill_(7)
    assert torch.equal(fused[4:], torch.full_like(up, 7))


@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_gpt_oss_interleaved_expert_targets_match_mcore_halves(suffix):
    shape = (2, 6, 8) if suffix == "weight" else (2, 6)
    native = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    converter = infer(GPTOSSVLLMWeightConverter)
    targets = dict(
        converter.convert_param(f"model.layers.0.mlp.experts.w13_{suffix}", native)
    )
    train = mcore(_build_mcore_converter_gpt_oss)
    for expert_id in range(2):
        expected_gate, expected_up = (
            native[expert_id, ::2].clone(),
            native[expert_id, 1::2].clone(),
        )
        converted = dict(
            train.convert_param(
                f"decoder.layers.0.mlp.experts.linear_fc1.{suffix}{expert_id}",
                torch.cat([expected_gate, expected_up]),
            )
        )
        for name, expected in converted.items():
            assert torch.equal(targets[name], expected)
        targets[f"model.layers.0.mlp.experts.{expert_id}.up_proj.{suffix}"].fill_(99)
        assert torch.all(native[expert_id, 1::2] == 99)
        assert torch.equal(native[expert_id, ::2], expected_gate)


def test_gpt_oss_bias_ownership_keeps_nonzero_bias_on_tp_zero():
    native = torch.ones(2, 8)
    assert (
        len(
            infer(GPTOSSVLLMWeightConverter, tp=4).convert_param(
                "model.layers.0.mlp.experts.w2_bias", native
            )
        )
        == 2
    )
    assert (
        infer(GPTOSSVLLMWeightConverter, tp=4, tp_rank=1).convert_param(
            "model.layers.0.mlp.experts.w2_bias", torch.zeros_like(native)
        )
        == []
    )
    with pytest.raises(ValueError, match="outside TP rank zero"):
        infer(GPTOSSVLLMWeightConverter, tp=4, tp_rank=1).convert_param(
            "model.layers.0.mlp.experts.w2_bias", native
        )


def test_gpt_oss_expert_parallel_offset_includes_biases():
    converted = dict(
        infer(GPTOSSVLLMWeightConverter, ep=4, ep_rank=2).convert_param(
            "model.layers.0.mlp.experts.routed_experts.w13_bias", torch.zeros(2, 6)
        )
    )
    assert set(converted) == {
        f"model.layers.0.mlp.experts.{i}.{p}_proj.bias"
        for i in [4, 5]
        for p in ["gate", "up"]
    }


def test_gpt_oss_down_bias_is_replicated_in_shape_but_expert_owned():
    rank = NS(tp_size=1, attn_tp_size=1)
    strategy = GPTOSSShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=1,
        tp_size=1,
        ep_size=8,
        ep_tp_size=1,
        rank_info=rank,
    )
    assert strategy.get_sharding_strategy(
        "model.layers.0.mlp.experts.4.down_proj.bias"
    ) == (ShardingType.EP_SHARDING, 0, 8)


def test_gpt_oss_sinks_and_router_bias_names_are_preserved():
    train = mcore(_build_mcore_converter_gpt_oss)
    native = torch.zeros(4)
    for source, target in [
        ("self_attention.core_attention.softmax_offset", "self_attn.sinks"),
        ("mlp.router.bias", "mlp.router.bias"),
        ("mlp.router.weight", "mlp.router.weight"),
    ]:
        [(name, value)] = train.convert_param("decoder.layers.0." + source, native)
        assert name == "model.layers.0." + target
        assert value.data_ptr() == native.data_ptr()


def glm_config():
    return NS(
        **vars(config()),
        n_routed_experts=8,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        num_hidden_layers=47,
        num_nextn_predict_layers=1,
    )


def glm_train():
    return _build_mcore_converter_glm47_flash()(
        glm_config(),
        NS(
            tp_size=2,
            attn_tp_size=2,
            attn_tp_rank=0,
            ep_size=8,
            ep_rank=3,
            pp_size=2,
            pp_rank=1,
        ),
        {"infer_atten_tp_size": 1, "train_pp_stage_layer_id_map": {"1:0": {0: 24}}},
        NS(),
    )


def glm_infer():
    return GLM47FlashVLLMWeightConverter(
        glm_config(), NS(tp_size=1, ep_size=8), NS(tp_rank=0, ep_rank=3)
    )


def test_glm_mla_fused_target_views_match_two_training_projections():
    fused = torch.arange(80).reshape(10, 8).float()
    converted = dict(
        glm_infer().convert_param(
            "model.layers.24.self_attn.fused_qkv_a_proj.weight", fused
        )
    )
    train = glm_train()
    for source, rows in [
        ("linear_q_down_proj", slice(0, 4)),
        ("linear_kv_down_proj", slice(4, 10)),
    ]:
        [(name, value)] = train.convert_param(
            f"decoder.layers.0.self_attention.{source}.weight", fused[rows]
        )
        assert torch.equal(converted[name], value)
        converted[name].fill_(7)
        assert torch.all(fused[rows] == 7)


@pytest.mark.parametrize("suffix", ["enorm.weight", "hnorm.weight", "eh_proj.weight"])
def test_glm_trained_mtp_is_not_filtered(suffix):
    tensor = torch.ones(4, 8)
    [(name, value)] = glm_train().convert_param("mtp.layers.0." + suffix, tensor)
    assert name == "model.layers.47." + suffix
    assert (
        dict(glm_infer().convert_param(name, value))[name].data_ptr()
        == tensor.data_ptr()
    )


def test_glm_mtp_decoder_id_bypasses_main_pipeline_layer_remapping():
    tensor = torch.ones(4, 8)
    train = glm_train()
    [(name, _)] = train.convert_param(
        "mtp.layers.0.mtp_model_layer.self_attention.linear_q_up_proj.weight", tensor
    )
    assert name == "model.layers.47.self_attn.q_b_proj.weight"
    [(target, _)] = glm_infer().convert_param(
        "model.layers.47.mtp_block.self_attn.q_b_proj.weight", tensor
    )
    assert target == name
    assert train._pp_stage_layer_id_map == {(1, 0): {0: 24}}


def test_glm_mtp_shared_head_norm_and_router_fp32():
    train = glm_train()
    [(name, _)] = train.convert_param(
        "mtp.layers.0.final_layernorm.weight", torch.ones(8)
    )
    assert name == "model.layers.47.shared_head.norm.weight"
    assert glm_infer().convert_param(name, torch.ones(8))[0][0] == name
    for source, target in [
        ("router.weight", "gate.weight"),
        ("router.expert_bias", "gate.e_score_correction_bias"),
    ]:
        [(name, tensor)] = train.convert_param(
            "decoder.layers.0.mlp." + source, torch.ones(8, dtype=torch.bfloat16)
        )
        assert name == "model.layers.24.mlp." + target
        assert tensor.dtype == torch.float32


def test_glm_mla_derived_weights_are_rebuilt_instead_of_transferred():
    assert (
        glm_infer().convert_param(
            "model.layers.0.self_attn.mla_attn.mla_attn.W_UV", torch.ones(2, 4, 2)
        )
        == []
    )

    class MLAAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.derived = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

        def process_weights_after_loading(self, dtype):
            assert dtype == torch.bfloat16
            with torch.no_grad():
                self.derived.copy_(self.raw * 3)

    module = MLAAttention()
    pointer = module.derived.data_ptr()
    glm_infer().post_update(module)
    assert module.derived.tolist() == [3, 3]
    assert module.derived.data_ptr() == pointer


def test_glm_transfer_model_includes_draft_but_deduplicates_shared_heads():
    main, draft = torch.nn.Module(), torch.nn.Module()
    main.lm_head = torch.nn.Linear(8, 8, bias=False)
    draft.lm_head = main.lm_head
    draft.eh_proj = torch.nn.Linear(16, 8, bias=False)
    combined = GLM47FlashTransferModel(main, draft, 47)
    names = dict(combined.named_parameters())
    assert set(names) == {"lm_head.weight", "eh_proj.weight"}
    assert len(list(combined.parameters())) == 2


@pytest.mark.parametrize(
    "rows,expected", [(4, ShardingType.NO_SHARDING), (2, ShardingType.TP_SHARDING)]
)
def test_glm_mla_replicated_and_column_parallel_down_projections(rows, expected):
    strategy = GLM47FlashShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=8,
        ep_tp_size=1,
        rank_info=NS(tp_size=2, attn_tp_size=2),
        hf_config=glm_config(),
    )
    kind, dimension, _ = strategy.get_sharding_strategy(
        "model.layers.0.self_attn.q_a_proj.weight", param_meta={"shape": (rows, 8)}
    )
    assert (kind, dimension) == (expected, 0)

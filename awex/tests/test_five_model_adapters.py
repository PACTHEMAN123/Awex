"""Five-model adapters: writable storage, expert offsets, router dtype and MLA/MTP."""
from types import SimpleNamespace as NS
import pytest
import torch
from awex.models.glm4_moe_lite import (GLM47FlashShardingStrategy, GLM47FlashVLLMWeightConverter, GLM47FlashTransferModel, _build_mcore_converter_glm47_flash)
from awex.models.gpt_oss import (GPTOSSShardingStrategy, GPTOSSVLLMWeightConverter, _build_mcore_converter_gpt_oss)
from awex.models.moonlight import (MoonlightShardingStrategy, MoonlightVLLMWeightConverter, _build_mcore_converter_moonlight)
from awex.sharding.param_sharding import ShardingType
from awex.meta.train_meta_resolver import _build_pp_stage_layer_id_map, _canonicalize_pp_layer_names_in_global_meta

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

def test_gpt_oss_actual_vllm_attention_and_embedding_names():
    converter = infer(GPTOSSVLLMWeightConverter)
    tensor = torch.arange(16 * 8).reshape(16, 8).float()
    converted = dict(
        converter.convert_param("model.layers.0.attn.qkv_proj.weight", tensor)
    )
    assert set(converted) == {
        f"model.layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v")
    }
    [(name, value)] = converter.convert_param("model.embedding.weight", tensor)
    assert name == "model.embed_tokens.weight"
    assert value.data_ptr() == tensor.data_ptr()
    [(name, value)] = converter.convert_param("model.layers.0.attn.sinks", tensor)
    assert name == "model.layers.0.self_attn.sinks"

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

def moonlight_config():
    cfg = glm_config()
    cfg.q_lora_rank = None
    cfg.num_hidden_layers = 27
    cfg.num_nextn_predict_layers = 0
    return cfg

def test_moonlight_uncompressed_query_and_shared_experts_match_inference_storage():
    cfg = moonlight_config()
    train = _build_mcore_converter_moonlight()(
        cfg,
        NS(tp_size=2, attn_tp_size=2, attn_tp_rank=0, ep_size=8,
           ep_rank=3, pp_size=1, pp_rank=0),
        {"infer_atten_tp_size": 2},
        NS(),
    )
    converter = MoonlightVLLMWeightConverter(
        cfg, NS(tp_size=2, ep_size=1), NS(tp_rank=0, ep_rank=0)
    )
    query = torch.arange(32).reshape(4, 8).float()
    [(name, value)] = train.convert_param(
        "decoder.layers.1.self_attention.linear_q_proj.weight", query
    )
    assert name == "model.layers.1.self_attn.q_proj.weight"
    assert dict(converter.convert_param(name, value))[name].data_ptr() == query.data_ptr()
    fused = torch.arange(64).reshape(8, 8).float()
    outputs = dict(train.convert_param(
        "decoder.layers.1.mlp.shared_experts.linear_fc1.weight", fused
    ))
    assert set(outputs) == {
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
    }
    assert torch.equal(outputs["model.layers.1.mlp.shared_experts.gate_proj.weight"], fused[:4])
    assert torch.equal(outputs["model.layers.1.mlp.shared_experts.up_proj.weight"], fused[4:])
    assert all(value.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
               for value in outputs.values())

def test_moonlight_replicated_mla_projection_and_shared_expert_tp_are_distinct():
    strategy = MoonlightShardingStrategy(
        engine_name="vllm", enable_dp_attention=False, enable_dp_lm_head=False,
        moe_dense_tp_size=2, tp_size=2, ep_size=1, ep_tp_size=1,
        rank_info=NS(tp_size=2, attn_tp_size=2), hf_config=moonlight_config(),
    )
    assert strategy.get_sharding_strategy(
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        param_meta={"shape": [6, 8]},
    ) == (ShardingType.NO_SHARDING, 0, 1)
    assert strategy.get_sharding_strategy(
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        param_meta={"shape": [4, 8]},
    ) == (ShardingType.TP_SHARDING, 0, 2)

@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize(
    "source_norm,target_norm",
    [
        ("input_layernorm", "input_layernorm"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
    ],
)
def test_glm_standalone_norms_keep_storage_and_resolve_main_or_mtp_pp_ids(
    mtp, source_norm, target_norm
):
    train = glm_train()
    prefix = "mtp.layers.0.mtp_model_layer" if mtp else "decoder.layers.0"
    tensor = torch.arange(8).float()
    [(name, view)] = train.convert_param_to_device_layout(
        f"{prefix}.{source_norm}.weight", tensor
    )
    assert name == f"model.layers.{47 if mtp else 24}.{target_norm}.weight"
    assert view.data_ptr() == tensor.data_ptr()

def test_glm_global_mtp_and_alias_ids_do_not_shift_main_pipeline_layers():
    train = glm_train()
    metadata = []
    for rank in (0, 1):
        params = [{"name": f"model.layers.{i}.input_layernorm.weight"} for i in (0, 1)]
        name = "model.layers.47.embed_tokens.weight"
        params.append(
            {"name": name, "layer_id_is_global": train.is_global_layer_name(name)}
        )
        metadata.append(
            {
                "rank_info": NS(pp_size=2, pp_rank=rank, global_rank=rank),
                "params_meta": params,
            }
        )
    mapping = _build_pp_stage_layer_id_map(metadata)
    assert mapping == {(0, 0): {0: 0, 1: 1}, (1, 0): {0: 2, 1: 3}}
    _canonicalize_pp_layer_names_in_global_meta(metadata, mapping)
    assert (
        metadata[1]["params_meta"][0]["name"] == "model.layers.2.input_layernorm.weight"
    )
    assert all(m["params_meta"][-1]["name"] == name for m in metadata)

@pytest.mark.parametrize("distinct", [False, True])
def test_glm_mtp_aliases_follow_actual_distinct_inference_targets(distinct):
    train = glm_train()
    alias = "model.layers.47.embed_tokens.weight"
    train._mtp_alias_targets = [alias] if distinct else []
    parameter = torch.ones(8, 8)
    names = dict(
        train.convert_param_to_device_layout(
            "embedding.word_embeddings.weight", parameter
        )
    )
    assert (alias in names) == distinct
    assert names["model.embed_tokens.weight"].data_ptr() == parameter.data_ptr()

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


def test_moonlight_gate_stays_bf16_while_correction_bias_uses_fp32():
    train = _build_mcore_converter_moonlight()(
        moonlight_config(),
        NS(tp_size=1, attn_tp_size=1, attn_tp_rank=0, ep_size=1,
           ep_rank=0, pp_size=1, pp_rank=0),
        {"infer_atten_tp_size": 1},
        NS(),
    )
    gate = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    [(name, value)] = train.convert_param("decoder.layers.1.mlp.router.weight", gate)
    assert name == "model.layers.1.mlp.gate.weight"
    assert value.dtype == torch.bfloat16
    assert value.data_ptr() == gate.data_ptr()
    bias = torch.tensor([0.5, -0.25], dtype=torch.float32)
    [(name, value)] = train.convert_param(
        "decoder.layers.1.mlp.router.expert_bias", bias
    )
    assert name == "model.layers.1.mlp.gate.e_score_correction_bias"
    assert value.dtype == torch.float32
    assert torch.equal(value, bias)

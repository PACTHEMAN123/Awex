from dataclasses import replace

import pytest
from shardstream_benchmarks.recipes import load_recipes, rollout_host_counts


@pytest.mark.parametrize("instances,expected", [(3, [1, 1, 1]), (4, [2, 1, 1]),
                                                (5, [2, 2, 1]), (6, [2, 2, 2])])
def test_complete_tp4_instances_balance_across_three_hosts(instances, expected):
    base = load_recipes()["qwen3-30b-a3b"]
    recipe = replace(base, training={**base.training, "world_size": 8},
                     rollout={**base.rollout, "tp": 4, "dp": 1, "ep": 1,
                              "instances": instances, "world_size": instances * 4})
    assert rollout_host_counts(instances, 3) == expected
    recipe.validate_placement(["train0"], ["infer0", "infer1", "infer2"])
    assert sum(expected) * 4 == recipe.rollout["world_size"]


def test_rollout_host_assignment_rejects_empty_hosts_and_overbooking():
    with pytest.raises(ValueError, match="complete instance"):
        rollout_host_counts(2, 3)
    base = load_recipes()["qwen3-30b-a3b"]
    recipe = replace(base, rollout={**base.rollout, "tp": 4, "dp": 1,
                                   "instances": 7, "world_size": 28})
    with pytest.raises(ValueError, match="rollout placement"):
        recipe.validate_placement(["train0", "train1"], ["infer0", "infer1", "infer2"])


def test_recipe_gpu_counts_and_cross_node_placement():
    for recipe in load_recipes().values():
        hosts = (
            ["train0", "train1"] if recipe.training["world_size"] > 8 else ["train0"]
        )
        rollout_hosts = [
            f"infer{i}" for i in range((recipe.rollout["world_size"] + 7) // 8)
        ]
        recipe.validate_placement(hosts, rollout_hosts)


def test_expanded_layouts_fit_budget_and_preserve_model_semantics():
    import json
    from pathlib import Path

    repository = Path(__file__).resolve().parents[1]
    layouts = json.loads((repository / "benchmarks/recipes/layouts.json").read_text())["layouts"]
    for entry in layouts:
        base = load_recipes()[entry["model"]]
        recipe = base.with_layout(entry["id"])
        assert recipe.training["world_size"] + recipe.rollout["world_size"] <= 32
        assert recipe.provider_overrides == base.provider_overrides
        assert recipe.fp32_parameter_suffixes == base.fp32_parameter_suffixes
        recipe.validate_placement(
            ["train0", "train1"] if recipe.training["world_size"] == 16 else ["train0"],
            ["infer0", "infer1"] if recipe.rollout["world_size"] == 16 else
            (["infer0", "infer1", "infer2"] if recipe.rollout["world_size"] == 24 else ["infer0"]),
        )


def test_layout_cannot_be_applied_to_another_model_or_overbooked():
    recipes = load_recipes()
    with pytest.raises(ValueError, match="Unknown layout"):
        recipes["glm-4.7-flash"].with_layout("qwen8-tp2-pp2-r4tp4")
    recipe = recipes["qwen3-8b"].with_layout("qwen8-tp2-pp2-r2tp8")
    with pytest.raises(ValueError, match="GPU budget"):
        recipe.with_rollout_replicas(3)


def test_qwen8_tp2_supplement_uses_two_trainers_and_one_full_rollout_node():
    recipe = load_recipes()["qwen3-8b"].with_layout("qwen8-tp2-pp2-r4tp2")
    assert recipe.training["world_size"] == 16
    assert (recipe.training["tp"], recipe.training["pp"]) == (2, 2)
    assert recipe.rollout["tp"] == 2
    assert recipe.rollout["instances"] == 4
    assert recipe.rollout["world_size"] == 8
    recipe.validate_placement(["train0", "train1"], ["infer0"])
    command = recipe.rollout_arguments("/models/qwen8", "infer0", 8000)
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    with pytest.raises(ValueError):
        recipe.validate_placement(["train0"], ["infer0"])


@pytest.mark.parametrize("suffix,tp,ep,instances", [("r2tp8ep8", 8, 8, 2), ("r4tp4", 4, 1, 4)])
def test_qwen30_replicated_training_dp2_layouts(suffix, tp, ep, instances):
    recipe = load_recipes()["qwen3-30b-a3b"].with_layout(
        "qwen30-tp4-cp2-ep8-" + suffix
    )
    train = recipe.training
    assert (train["tp"], train["cp"], train["ep"], train["etp"]) == (4, 2, 8, 1)
    assert train["world_size"] // (train["tp"] * train["pp"] * train["cp"]) == 2
    assert train["world_size"] // (train["etp"] * train["pp"] * train["ep"]) == 2
    assert train["world_size"] // 2 == train["tp"] * train["cp"] == 8
    rollout = recipe.rollout
    assert (rollout["tp"], rollout["ep"], rollout["instances"]) == (tp, ep, instances)
    assert rollout["world_size"] == 16 and instances // 2 == 8 // tp
    recipe.validate_placement(["train0", "train1"], ["infer0", "infer1"])


def test_glm_vllm_dp_expert_group_uses_eight_rollout_gpus():
    recipe = load_recipes()["glm-4.7-flash"]
    assert recipe.rollout["attention_dp"] == 8
    assert recipe.rollout["engine"] == "vllm"
    assert recipe.rollout["tp"] == 1
    assert recipe.rollout["dp"] == recipe.rollout["ep"] == 8
    assert recipe.training["world_size"] + recipe.rollout["world_size"] == 24


def test_glm_upstream_eight_gpu_actor_cannot_satisfy_expert_pipeline_topology():
    recipe = load_recipes()["glm-4.7-flash"]
    bad = replace(recipe, training={**recipe.training, "world_size": 8})
    from pathlib import Path

    with pytest.raises(RuntimeError, match="required multiple=16"):
        bad.validate(Path(__file__).resolve().parents[1])


def test_same_host_is_rejected_even_when_different_gpus_would_be_available():
    recipe = load_recipes()["qwen3.5-9b"]
    with pytest.raises(ValueError, match="different physical hosts"):
        recipe.validate_placement(["node0"], ["node0"])


def test_glm_requires_two_training_nodes():
    recipe = load_recipes()["glm-4.7-flash"]
    with pytest.raises(ValueError, match="does not fit"):
        recipe.validate_placement(["train0"], ["infer0"])


def test_glm_launch_keeps_dp_experts_and_trained_mtp():
    import json

    recipe = load_recipes()["glm-4.7-flash"]
    command = recipe.rollout_arguments("/models/glm", "node0", 8000)
    assert command[command.index("--data-parallel-size") + 1] == "8"
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert "--enable-expert-parallel" in command
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative == {"method": "mtp", "num_speculative_tokens": 3}
    assert recipe.provider_overrides == {
        "mtp_num_layers": 1,
        "num_layers_in_last_pipeline_stage": 23,
    }


def test_qwen35_recipe_does_not_enable_mtp_in_either_engine():
    recipe = load_recipes()["qwen3.5-9b"]
    assert recipe.provider_overrides["mtp_num_layers"] == 0
    assert "--speculative-config" not in recipe.rollout_arguments(
        "/models/qwen", "node0", 8000
    )


@pytest.mark.parametrize("model,replicas", [("qwen3.5-9b", 4), ("qwen2.5-vl-7b", 8)])
def test_replica_scaling_retains_recipe_axes_and_pinned_defaults(model, replicas):
    pinned = load_recipes()[model]
    scaled = pinned.with_rollout_replicas(replicas)
    assert scaled.training == pinned.training
    assert scaled.source == pinned.source
    assert scaled.rollout["world_size"] == 8
    assert scaled.rollout["instances"] == replicas
    for axis in ("tp", "dp", "ep", "attention_dp"):
        assert scaled.rollout[axis] == pinned.rollout[axis]
    assert pinned.rollout["world_size"] == 4
    scaled.validate_placement(["train0"], ["infer0"])
    with pytest.raises(ValueError, match="does not fit"):
        scaled.with_rollout_replicas(replicas * 2).validate_placement(
            ["train0"], ["infer0"]
        )


def test_replica_override_rejects_zero_and_combined_gpu_budget_overflow():
    recipe = load_recipes()["qwen3.5-9b"]
    with pytest.raises(ValueError, match="positive"):
        recipe.with_rollout_replicas(0)
    with pytest.raises(ValueError, match="GPU budget"):
        recipe.with_rollout_replicas(15)


def test_rollout_process_uses_only_its_reserved_ray_gpus(tmp_path, monkeypatch):
    from shardstream_benchmarks.recipe_runner import NodeProcesses

    launched = {}

    class FakeProcess:
        pid = 1234

        def __init__(self, command, **kwargs):
            launched.update(kwargs)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setattr(
        "shardstream_benchmarks.recipe_runner.subprocess.Popen", FakeProcess
    )
    actor = NodeProcesses(tmp_path, tmp_path, tmp_path, tmp_path / "run")
    actor.launch("rollout", ["-c", "pass"], {"SHARDSTREAM_RECIPE_GPU_INDICES": "2,3"})
    assert launched["env"]["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert launched["start_new_session"] is True
    rollout_sync = launched["env"]["SHARDSTREAM_PROFILE_SYNC_START"]
    actor.launch("training", ["-c", "pass"], {})
    # Both roles must enter the same collective before starting the transfer.
    assert launched["env"]["SHARDSTREAM_PROFILE_SYNC_START"] == rollout_sync == "1"


def test_fp32_state_survives_checkpoint_loading_into_bf16_model():
    import torch
    from shardstream_benchmarks.model_loader import _preserve_fp32_parameters

    model = torch.nn.Module()
    model.self_attention = torch.nn.Module()
    model.self_attention.A_log = torch.nn.Parameter(
        torch.zeros(3, dtype=torch.bfloat16)
    )
    model.weight = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    checkpoint = torch.tensor([0.1234567, 1.234567, 2.345678], dtype=torch.float32)
    # Match both bare modules and wrappers, as providers return wrapped models.
    _preserve_fp32_parameters([model], ("self_attention.A_log",))
    with torch.no_grad():
        model.self_attention.A_log.copy_(checkpoint)
    assert torch.equal(model.self_attention.A_log, checkpoint)
    assert model.weight.dtype == torch.bfloat16


def test_moonlight_checkpoint_gate_is_not_promoted_by_glm_recipe():
    recipes = load_recipes()
    name = "decoder.layers.1.mlp.router.weight"
    assert name.endswith(recipes["glm-4.7-flash"].fp32_parameter_suffixes)
    assert not name.endswith(recipes["moonlight-16b-a3b"].fp32_parameter_suffixes)
    assert "decoder.layers.1.mlp.router.expert_bias".endswith(
        recipes["moonlight-16b-a3b"].fp32_parameter_suffixes
    )


def test_moonlight_query_spec_keeps_kv_norm_and_original_layer_spec():
    pytest.importorskip("megatron.core")
    from types import SimpleNamespace as NS

    from megatron.core.transformer.mla_qk_norm_config import get_backend
    from shardstream_benchmarks.model_loader import _uncompressed_query_layer_spec

    class Attention:
        def _resolve_qk_norm_config(self, submodules):
            return {"linear_q_proj": "fused-query", "linear_kv_up_proj": "fused-kv"}

    original = NS(layer_specs=[NS(submodules=NS(
        self_attention=NS(module=Attention), input_layernorm="input-norm"))])
    config = NS(q_lora_rank=None, transformer_impl="transformer_engine")
    result = _uncompressed_query_layer_spec(lambda config: original, config)
    attention = result.layer_specs[0].submodules.self_attention.module()
    attention.config = config
    resolved = attention._resolve_qk_norm_config(None)
    assert resolved["linear_q_proj"] is get_backend(config.transformer_impl).column_parallel_linear()
    assert resolved["linear_kv_up_proj"] == "fused-kv"
    assert result.layer_specs[0].submodules.input_layernorm == "input-norm"
    assert original.layer_specs[0].submodules.self_attention.module is Attention
    attention.config = NS(q_lora_rank=128)
    assert attention._resolve_qk_norm_config(None)["linear_q_proj"] == "fused-query"

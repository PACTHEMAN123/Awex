from dataclasses import replace

import pytest
from shardstream_benchmarks.recipes import load_recipes


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


def test_moonlight_bridge_maps_fused_input_norm_without_changing_other_models():
    pytest.importorskip("megatron.bridge")
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
    from transformers import DeepseekV3Config, Qwen3Config
    from shardstream_benchmarks.model_loader import _provider_auto_bridge_class

    config = DeepseekV3Config(q_lora_rank=None)
    config.architectures = ["DeepseekV3ForCausalLM"]
    bridge = _provider_auto_bridge_class(config).from_hf_config(config)
    registry = bridge._model_bridge.mapping_registry()
    mapping = registry.megatron_to_hf_lookup(
        "decoder.layers.3.self_attention.linear_q_proj.layer_norm_weight"
    )
    assert isinstance(mapping, ReplicatedMapping)
    assert mapping.hf_param == "model.layers.3.input_layernorm.weight"
    assert registry.megatron_to_hf_lookup(
        "decoder.layers.3.self_attention.linear_q_proj.weight"
    ).hf_param == "model.layers.3.self_attn.q_proj.weight"
    assert _provider_auto_bridge_class(Qwen3Config()) is AutoBridge
    assert _provider_auto_bridge_class(DeepseekV3Config(q_lora_rank=128)) is AutoBridge

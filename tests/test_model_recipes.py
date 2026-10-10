from dataclasses import replace

import pytest
from shardstream_benchmarks.recipes import load_recipes


def test_recipe_gpu_counts_and_cross_node_placement():
    for recipe in load_recipes().values():
        hosts = (
            ["train0", "train1"] if recipe.training["world_size"] > 8 else ["train0"]
        )
        recipe.validate_placement(hosts, ["infer0"])


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

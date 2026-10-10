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

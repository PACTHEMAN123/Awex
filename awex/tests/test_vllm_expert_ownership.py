"""Exercise the worker's actual rank-reporting method and GPT expert conversion."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from awex.models.gpt_oss import GPTOSSVLLMWeightConverter
from awex.sharding.vllm_sharding import get_vllm_rank_info


def rank_context(tp, rank, enable_ep):
    path = Path(__file__).parents[1] / "vllm_plugin.py"
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_awex_rank_info"
    )
    namespace = {"InferenceConfig": object, "os": os}
    for name, size, value in [
        ("tp", tp, rank),
        ("ep", tp, rank),
        ("dp", 1, 0),
        ("pp", 1, 0),
    ]:
        group = SimpleNamespace(world_size=size, rank_in_group=value)
        namespace[f"get_{name}_group"] = lambda group=group: group
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    parallel = SimpleNamespace(
        enable_expert_parallel=enable_ep,
        world_size=tp,
        rank=rank,
        world_size_across_dp=tp,
        prefill_context_parallel_size=1,
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            vllm_config=SimpleNamespace(parallel_config=parallel)
        ),
        local_rank=rank,
    )
    return namespace["_awex_rank_info"](worker)


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("enable_ep", [False, True])
def test_ep_group_only_assigns_experts_when_expert_parallel_is_enabled(tp, enable_ep):
    total_experts = 32
    all_names = []
    for rank in range(tp):
        context = rank_context(tp, rank, enable_ep)
        assert (context["ep_rank"], context["ep_size"]) == (
            (rank, tp) if enable_ep else (0, 1)
        )
        assert context["tp_rank"] == context["global_rank"] == rank
        info = get_vllm_rank_info(context, engine_rank=0)
        cfg = SimpleNamespace(tp_size=tp, ep_size=tp if enable_ep else 1)
        hf = SimpleNamespace(num_attention_heads=8, num_key_value_heads=4)
        converter = GPTOSSVLLMWeightConverter(hf, cfg, info)
        count = total_experts // tp if enable_ep else total_experts
        params = converter.convert_param(
            "model.layers.0.mlp.experts.w13_weight", torch.zeros(count, 6, 4)
        )
        names = {name for name, _ in params}
        start = rank * count if enable_ep else 0
        expected = {
            f"model.layers.0.mlp.experts.{expert}.{kind}_proj.weight"
            for expert in range(start, start + count)
            for kind in ("gate", "up")
        }
        assert names == expected
        all_names.append(names)
    assert set.union(*all_names) == {
        f"model.layers.0.mlp.experts.{expert}.{kind}_proj.weight"
        for expert in range(total_experts)
        for kind in ("gate", "up")
    }
    if not enable_ep:
        assert all(names == all_names[0] for names in all_names)

"""Native FP8 refits must consume reusable NCCL bucket views synchronously."""
import sys
from types import SimpleNamespace

import torch

from shardstream.integrations.publication.verl_native import _load_fp8_weights


def test_fp8_native_loader_consumes_views_before_bucket_reuse(monkeypatch):
    events = []
    loaded = []
    parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    class Model:
        def named_parameters(self):
            return [('weight', parameter)]

        def parameters(self):
            return [parameter]

        def load_weights(self, weights):
            for name, tensor in weights:
                events.append(('load', name))
                loaded.append(tensor.clone())

    def quant_weights(weights, model, config, dtype):
        assert events == ['prepare']
        for name, weight in weights:
            yield name, weight * 2

    state = object()

    def prepare(model):
        events.append('prepare')
        return state

    def finish(model, actual_state):
        assert actual_state is state
        events.append('finish')

    monkeypatch.setitem(sys.modules, 'verl.utils.vllm.vllm_quant_utils', SimpleNamespace(
        prepare_quanted_weights_for_loading=prepare,
        process_quanted_weights_after_loading=finish,
        quant_weights=quant_weights,
    ))
    buffer = torch.zeros(4)

    def weights():
        for version in range(4):
            buffer.fill_(version)
            yield str(version), buffer

    runner = SimpleNamespace(model=Model(), vllm_config=SimpleNamespace(
        quant_config=object(), model_config=SimpleNamespace(dtype=torch.bfloat16)))
    _load_fp8_weights(runner, weights())
    assert [tensor.tolist() for tensor in loaded] == [[2 * v] * 4 for v in range(4)]
    assert events == ['prepare', *[('load', str(v)) for v in range(4)], 'finish']


def test_native_fp8_oracle_uses_dense_tp_slice_and_complete_ep_expert(tmp_path, monkeypatch):
    import json
    import pytest
    from shardstream.profiling.fp8_model import FP8ModelWeightProfile

    safetensors = pytest.importorskip('safetensors.torch')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    dense_name = 'model.layers.0.self_attn.o_proj.weight'
    expert_name = 'model.layers.0.mlp.experts.7.down_proj.weight'
    dense = torch.arange(128 * 256).reshape(128, 256).bfloat16() / 1024
    expert = torch.linspace(-1, 2, 128 * 128).reshape(128, 128).bfloat16()
    checkpoint = {dense_name: dense, expert_name: expert,
                  'model.norm.weight': torch.ones(128, dtype=torch.bfloat16)}
    safetensors.save_file(checkpoint, tmp_path/'weights.safetensors')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {key: 'weights.safetensors' for key in checkpoint}}))
    native = {
        dense_name: torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
        dense_name + '_scale_inv': torch.zeros(1, 1),
        'model.layers.0.mlp.experts.w2_weight': torch.zeros((1, 128, 128), dtype=torch.float8_e4m3fn),
        'model.layers.0.mlp.experts.w2_weight_scale_inv': torch.zeros(1, 1, 1),
        'model.norm.weight': torch.zeros(128, dtype=torch.bfloat16),
    }
    model = SimpleNamespace(named_parameters=lambda: native.items(), config=SimpleNamespace(
        architectures=['Qwen3MoeForCausalLM'], num_attention_heads=4,
        num_key_value_heads=2, num_experts=8))
    config = SimpleNamespace(tp_size=2, ep_size=8, enable_dp_attention=False,
                             enable_dp_lm_head=False, moe_dense_tp_size=2, device_backend='cuda')
    context = dict(tp_rank=1, tp_size=2, pp_rank=0, pp_size=1, dp_rank=3, dp_size=4,
                   ep_rank=7, ep_size=8, ep_tp_rank=0, ep_tp_size=1,
                   attn_tp_rank=1, attn_tp_size=2, attn_dp_rank=3,
                   world_size=8, global_rank=7, local_rank=7, infer_engine_config=config)
    oracle = FP8ModelWeightProfile.from_vllm_checkpoint(model, context, tmp_path, 'scale', 35)
    assert torch.equal(oracle.references[dense_name], dense[:, 128:])
    assert torch.equal(oracle.references[expert_name], expert)
    # Replacing the live model tensor must fail even if old canonical views survive.
    native[dense_name] = native[dense_name].clone()
    with pytest.raises(AssertionError, match='native FP8 model parameter was replaced'):
        oracle.command('clear', 0)

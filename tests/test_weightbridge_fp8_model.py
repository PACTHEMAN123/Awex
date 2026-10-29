import sys
from types import SimpleNamespace

import pytest
import torch

from shardstream.integrations.publication import weightbridge_fp8_model as wb


@pytest.mark.parametrize('name, expected', [
    ('model.layers.0.self_attn.q_proj.weight', True),
    ('model.layers.0.mlp.experts.17.down_proj.weight', True),
    ('model.layers.0.self_attn.q_norm.weight', False),
    ('model.layers.0.mlp.gate.weight', False),
    ('model.embed_tokens.weight', False), ('lm_head.weight', False),
    ('model.layers.0.self_attn.q_proj.weight_scale_inv', False),
])
def test_qwen3_quantization_policy(name, expected):
    assert wb.is_quantized(name) == expected


def test_catalog_adds_real_scale_shapes_and_preserves_mixed_weights(monkeypatch):
    name = 'model.layers.0.self_attn.q_proj.weight'
    value = torch.ones(256, 128, dtype=torch.bfloat16)
    norm = torch.ones(128)
    calls = []
    def quantize(x):
        assert x is value
        calls.append(1)
        return x.to(torch.float8_e4m3fn), torch.ones(2, 1)
    monkeypatch.setattr(wb, 'quantize', quantize)
    factory = lambda: norm
    factories, shapes = wb.quantized_catalog({name: lambda: value, 'norm': factory},
                                           {name: (256, 128), 'norm': (128,)})
    assert not calls
    assert factories['norm'] is factory
    assert factories[name]().dtype == torch.uint8
    assert factories[name + '_scale_inv']().dtype == torch.float32
    assert shapes[name + '_scale_inv'] == (2, 1)


def test_unsupported_catalog_fails_without_silent_local_quantization():
    name = 'model.layers.0.self_attn.q_proj.weight'
    with pytest.raises(ValueError, match='physical tensor'):
        wb.quantized_catalog({name: lambda: None}, {name: (129, 128)})


def test_vllm_loader_reinterprets_fp8_bytes_without_requantizing(monkeypatch):
    events = []
    module = SimpleNamespace(
        prepare_quanted_weights_for_loading=lambda model: (events.append('prepare') or {'opaque': True}),
        process_quanted_weights_after_loading=lambda model, state: events.append(('process', state)),
    )
    monkeypatch.setitem(sys.modules, 'verl.utils.vllm.vllm_quant_utils', module)
    name = 'model.layers.0.self_attn.q_proj.weight'
    raw = torch.arange(256, dtype=torch.uint8).view(16, 16)
    received = {}
    model = SimpleNamespace(named_parameters=lambda: [], parameters=lambda: [],
                            load_weights=lambda weights: received.update(weights))
    wb.load_native_fp8(SimpleNamespace(model=model), {name: lambda: raw,
        name + '_scale_inv': lambda: torch.ones(1, 1), 'model.norm.weight': lambda: torch.ones(16)})
    assert received[name].dtype == torch.float8_e4m3fn
    assert torch.equal(received[name].view(torch.uint8), raw)
    assert received[name].data_ptr() == raw.data_ptr()
    assert events == ['prepare', ('process', {'opaque': True})]

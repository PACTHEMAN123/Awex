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

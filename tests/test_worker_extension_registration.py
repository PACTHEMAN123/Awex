import runpy
import sys
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_spawned_worker_registers_selected_baseline(monkeypatch, enabled):
    calls = []
    for name, method in (
        ("shardstream.integrations.vllm.plugin", "_patch_shardstream_worker"),
        ("awex.vllm_plugin", "_patch_awex_worker"),
    ):
        module = types.ModuleType(name)
        setattr(module, method, lambda method=method: calls.append(method))
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("VERL_VLLM_FP8_QUANT_ENABLED", raising=False)
    monkeypatch.delenv("SHARDSTREAM_BENCH_ENABLE_AWEX", raising=False)
    if enabled is not None:
        monkeypatch.setenv("SHARDSTREAM_BENCH_ENABLE_AWEX", enabled)
    runpy.run_path(str(Path(__file__).parents[1] / "python/shardstream/integrations/vllm/worker_extension.py"))
    assert calls == ["_patch_shardstream_worker"] + (["_patch_awex_worker"] if enabled == "1" else [])

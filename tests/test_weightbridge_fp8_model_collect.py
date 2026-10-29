import importlib.util
import io
from pathlib import Path
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def collector(monkeypatch, tmp_path):
    ray = ModuleType('ray')
    ray.remote = lambda **kwargs: lambda function: function
    ray.util = SimpleNamespace(get_node_ip_address=lambda: 'test-node')
    strategies = ModuleType('ray.util.scheduling_strategies')
    strategies.NodeAffinitySchedulingStrategy = object
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setitem(sys.modules, 'ray.util.scheduling_strategies', strategies)
    path = (Path(__file__).parents[1] / 'benchmarks/experiments/'
            'weightbridge_quantized_20261029/model_run.py')
    spec = importlib.util.spec_from_file_location('wb_model_collect_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'BASE', tmp_path)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: '')
    return module


def test_lossless_xz_collection_round_trip(collector, tmp_path):
    run = tmp_path / 'model-runs/example'
    run.mkdir(parents=True)
    payload = b'full model evidence\n' * 1000
    (run / 'training.log').write_bytes(payload)
    row = collector.collect(str(run))
    assert row['ip'] == 'test-node' and row['gpu_processes'] == ''
    assert row['data'].startswith(b'\xfd7zXZ\x00')
    with tarfile.open(fileobj=io.BytesIO(row['data'])) as archive:
        assert archive.getnames() == ['training.log']
        assert archive.extractfile('training.log').read() == payload


def test_collection_rejects_unrelated_directory(collector, tmp_path):
    with pytest.raises(AssertionError):
        collector.collect(str(tmp_path / 'unrelated'))

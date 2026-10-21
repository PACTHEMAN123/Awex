import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def timeline(monkeypatch):
    path = Path(__file__).parents[1] / "python/shardstream/profiling/timeline.py"
    spec = importlib.util.spec_from_file_location("timeline_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def start(label):
        calls.append(("open", label))
        return len(calls)
    cuda = SimpleNamespace(
        nvtx=SimpleNamespace(range_start=start, range_end=lambda token: calls.append(("close", token))),
        profiler=SimpleNamespace(start=lambda: calls.append("start"), stop=lambda: calls.append("stop")),
        current_device=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("SHARDSTREAM_TIMELINE", "1")
    return module, calls


def test_capture_boundaries(timeline, monkeypatch):
    module, calls = timeline
    for version in range(-1, 6):
        with module.publication("training", version, 0):
            with module.phase("work"):
                pass
    assert calls.count("start") == calls.count("stop") == 1
    assert sum(isinstance(c, tuple) and c[0] == "open" for c in calls) == 6
    assert not module._active
    calls.clear()
    monkeypatch.setenv("SHARDSTREAM_TIMELINE", "0")
    with module.publication("training", 2, 0):
        with module.phase("work"):
            pass
    assert not calls


def test_iterator_closes_before_yield(timeline):
    module, calls = timeline
    module._active = True
    iterator = module.iter_phases(iter([1, 2]), "export")
    assert next(iterator) == 1
    assert calls[-1][0] == "close"
    assert list(iterator) == [2]
    assert len(calls) == 6


def test_async_wrapper_balanced_and_idempotent(timeline):
    module, calls = timeline
    module._active = True
    async def fail():
        raise ValueError("expected")
    owner = SimpleNamespace(fail=fail)
    module.wrap(owner, "fail", "wait")
    wrapped = owner.fail
    module.wrap(owner, "fail", "wait")
    assert owner.fail is wrapped
    with pytest.raises(ValueError):
        asyncio.run(owner.fail())
    assert len(calls) == 2 and calls[-1][0] == "close"


def test_publication_exception_restores_state(timeline):
    module, calls = timeline
    with pytest.raises(RuntimeError):
        with module.publication("rollout", 4, 1):
            raise RuntimeError("expected")
    assert not module._active
    assert calls[-1] == "stop"

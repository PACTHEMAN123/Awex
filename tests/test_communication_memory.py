import sys
from types import SimpleNamespace

import pytest

from shardstream.profiling import communication_memory as memory


def test_memory_profiling_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SHARDSTREAM_COMM_MEMORY_PROFILE", raising=False)
    with pytest.raises(RuntimeError, match="not enabled"):
        memory.memory_task()
    monkeypatch.setenv("SHARDSTREAM_COMM_MEMORY_PROFILE", "1")
    assert memory.enabled()


def test_native_sampler_stops_and_retains_peak(monkeypatch):
    values = iter([100, 200])
    calls = []
    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=next(values))))
    monkeypatch.setattr(memory.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(memory.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-test"))
    monkeypatch.setattr(memory.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(memory, "snapshot", lambda: {"allocated_bytes": 10})
    sampler = memory.MemorySampler(interval=100).start()
    result = sampler.stop()
    assert not sampler.thread.is_alive()
    assert result["sampled_device_peak_bytes"] == 200
    assert result["sample_count"] == 2
    assert calls == ["init", "shutdown"]

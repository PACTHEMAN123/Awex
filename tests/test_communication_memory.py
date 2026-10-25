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
    host_values = iter([{"host_rss_bytes": 500}, {"host_rss_bytes": 400}])
    monkeypatch.setattr(memory, "host_memory", lambda: next(host_values))
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
    assert result["sampled_host_rss_peak_bytes"] == 500
    assert result["sampled_host_rss_after_bytes"] == 400
    assert calls == ["init", "shutdown"]


def test_host_memory_reports_bytes_without_equating_locked_memory_to_rss(monkeypatch, tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tworker\nVmRSS:\t200 kB\nVmHWM:\t300 kB\nVmLck:\t4 kB\n")
    monkeypatch.setattr(memory, "Path", lambda _: status)
    assert memory.host_memory() == {"host_rss_bytes": 204800,
                                    "host_rss_high_water_bytes": 307200,
                                    "host_locked_bytes": 4096}

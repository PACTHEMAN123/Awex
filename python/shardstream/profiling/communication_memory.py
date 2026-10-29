"""Opt-in device-wide memory sampling for exclusive-GPU publication benchmarks."""

import os
from pathlib import Path
import threading
import time

import torch


def enabled():
    return os.environ.get("SHARDSTREAM_COMM_MEMORY_PROFILE") == "1"


def host_memory():
    """Process RSS and Linux locked memory, in bytes (VmLck is not all CUDA pinning)."""
    names = {"VmRSS": "host_rss_bytes", "VmHWM": "host_rss_high_water_bytes",
             "VmLck": "host_locked_bytes"}
    status = Path("/proc/self/status")
    if not status.exists():
        return {}
    result = {}
    for line in status.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in names:
            amount, unit = value.split()
            if unit != "kB":
                raise ValueError(f"Unexpected process memory unit: {unit}")
            result[names[key]] = int(amount) * 1024
    return result


def snapshot(empty_cache=False):
    import pynvml

    torch.cuda.synchronize()
    if empty_cache:
        torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    pynvml.nvmlInit()
    try:
        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
        nvml_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()
    return dict(pid=os.getpid(), time_ns=time.time_ns(),
                nvml_used_bytes=nvml_used,
                device_used_bytes=total - free,
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), **host_memory())


class MemorySampler:
    """NVML captures native allocations omitted by the PyTorch allocator.

    Samples are a lower bound on the true peak. Call only with one model worker
    per physical GPU; device-wide readings include CUDA context/library memory.
    """

    def __init__(self, interval=0.01):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []
        self.host_samples = []
        self.error = None

    def start(self):
        self.before = snapshot()
        torch.cuda.reset_peak_memory_stats()
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        self.samples.append((time.time_ns(), self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used))
        self.host_samples.append(host_memory())

    def run(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except Exception as exc:
            self.error = repr(exc)

    def stop(self):
        after = snapshot()
        self.stop_event.set()
        self.thread.join()
        self.sample()
        self.nvml.nvmlShutdown()
        if self.error:
            raise RuntimeError(self.error)
        host = {}
        if all("host_rss_bytes" in sample for sample in self.host_samples):
            host = dict(sampled_host_rss_before_bytes=self.host_samples[0]["host_rss_bytes"],
                        sampled_host_rss_after_bytes=self.host_samples[-1]["host_rss_bytes"],
                        sampled_host_rss_peak_bytes=max(s["host_rss_bytes"] for s in self.host_samples))
        return dict(before=self.before, after=after, **host,
                    sampled_device_before_bytes=self.samples[0][1],
                    sampled_device_after_bytes=self.samples[-1][1],
                    sampled_device_peak_bytes=max(value for _, value in self.samples),
                    sample_count=len(self.samples), sample_interval_ms=self.interval * 1000,
                    sample_first_ns=self.samples[0][0], sample_last_ns=self.samples[-1][0])


def memory_task(empty_cache=False, **kwargs):
    if not enabled():
        raise RuntimeError("Communication memory profiling is not enabled")
    return snapshot(empty_cache=empty_cache)

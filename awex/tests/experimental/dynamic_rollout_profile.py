"""Bounded BF16 computation and wall-clock spans for the elastic mock.

GPU event durations measure device work. Span endpoints bound submission and
completion on the host; they are deliberately not presented as kernel timestamps.
The computation reads a private snapshot of the last published model weights.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def span(events: list[dict], name: str):
    started = time.time_ns()
    try:
        yield
    finally:
        events.append({"name": name, "start_ns": started, "end_ns": time.time_ns()})


class MockCompute:
    """Continuously complete bounded GEMM batches on a separate CUDA stream."""

    def __init__(self, weight: torch.Tensor, batch: int, repeats: int):
        self.weight = weight.detach().clone().to(torch.bfloat16)
        self.inputs = torch.ones(
            (batch, weight.shape[1]), device=weight.device, dtype=torch.bfloat16
        )
        self.output = torch.empty(
            (batch, weight.shape[0]), device=weight.device, dtype=torch.bfloat16
        )
        self.repeats = repeats
        self.events = []
        self.error = None
        self.stopping = threading.Event()
        self.ready = threading.Event()
        if weight.is_cuda:
            torch.cuda.synchronize(weight.device)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout: float):
        self.thread.start()
        if not self.ready.wait(timeout):
            raise TimeoutError("Background compute did not complete its first batch")
        if self.error:
            raise RuntimeError(self.error)

    def _run(self):
        try:
            cuda = self.weight.is_cuda
            if cuda:
                torch.cuda.set_device(self.weight.device)
                stream = torch.cuda.Stream(device=self.weight.device)
            with torch.cuda.stream(stream) if cuda else nullcontext():
                while not self.stopping.is_set():
                    start_ns = time.time_ns()
                    if cuda:
                        begin, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        begin.record(stream)
                    for _ in range(self.repeats):
                        torch.mm(self.inputs, self.weight.T, out=self.output)
                    if cuda:
                        end.record(stream)
                        end.synchronize()
                    self.events.append(
                        {
                            "name": "BF16 GEMM submit-to-complete",
                            "start_ns": start_ns,
                            "end_ns": time.time_ns(),
                            "gpu_ms": begin.elapsed_time(end) if cuda else None,
                            "gemms": self.repeats,
                        }
                    )
                    self.ready.set()
        except Exception as exc:
            self.error = repr(exc)
            self.ready.set()

    def stop(self, timeout: float) -> list[dict]:
        self.stopping.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("Background compute did not stop before publication")
        if self.error:
            raise RuntimeError(self.error)
        if not torch.isfinite(self.output).all().item():
            raise AssertionError("Background BF16 GEMM produced nonfinite output")
        return self.events

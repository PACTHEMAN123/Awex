"""Opt-in NVTX ranges; no synchronization or tensor retention is added."""

from contextlib import contextmanager
from functools import wraps
import inspect
import json
import os

_active = False


def enabled():
    return os.environ.get("SHARDSTREAM_TIMELINE") == "1"


@contextmanager
def phase(label):
    if not _active:
        yield
        return
    import torch

    # Process ranges, unlike push/pop stacks, may overlap across async tasks.
    handle = torch.cuda.nvtx.range_start(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_end(handle)


@contextmanager
def publication(role, version, rank):
    global _active
    if not enabled():
        yield
        return
    import torch

    first = int(os.environ.get("SHARDSTREAM_TIMELINE_START", "2"))
    last = int(os.environ.get("SHARDSTREAM_TIMELINE_STOP", "4"))
    if version == first:
        print("TIMELINE_WORKER " + json.dumps(dict(
            role=role, rank=rank, pid=os.getpid(), device=torch.cuda.current_device(),
            start_version=first, stop_version=last)), flush=True)
        torch.cuda.profiler.start()
    previous = _active
    _active = first <= version <= last
    try:
        with phase(f"publication/{role}/rank={rank}/version={version}"):
            yield
    finally:
        _active = previous
        if version == last:
            torch.cuda.profiler.stop()


def iter_phases(iterator, label):
    iterator = iter(iterator)
    while True:
        try:
            with phase(label):
                value = next(iterator)
        except StopIteration:
            return
        yield value
        del value


def wrap(owner, name, label):
    original = getattr(owner, name)
    if getattr(original, "_timeline_wrapped", False):
        return
    if inspect.iscoroutinefunction(original):
        @wraps(original)
        async def decorated(*args, **kwargs):
            with phase(label):
                return await original(*args, **kwargs)
    else:
        @wraps(original)
        def decorated(*args, **kwargs):
            with phase(label):
                return original(*args, **kwargs)
    decorated._timeline_wrapped = True
    setattr(owner, name, decorated)


def instrument_verl(module, backend):
    if not enabled():
        return
    if backend == "nccl":
        wrap(module.BroadcastOperation, "_run", "nccl/metadata_and_enqueue")
        wrap(module.BroadcastOperation, "wait_for_complete", "nccl/enqueue_wait")
    elif backend == "nixl":
        for name, label in (("read_metadata", "metadata_wait"),
                            ("begin_read", "read_submit"),
                            ("wait_for_complete", "read_completion_wait")):
            wrap(module.ReadOperation, name, "nixl/" + label)
        wrap(module.ReadableOperation, "__init__", "nixl/advertise")
        wrap(module.ReadableOperation, "wait_for_complete", "nixl/downstream_ack_wait")


def instrument_weightbridge():
    if not enabled():
        return
    from wbridge.backend import gantt

    if getattr(gantt.span, "_timeline_wrapped", False):
        return
    original = gantt.span

    @contextmanager
    def span(role, rank, wt, op, rnd):
        with phase(f"wb/{role}/rank={rank}/{op}/round={rnd}"), original(role, rank, wt, op, rnd):
            yield

    span._timeline_wrapped = True
    gantt.span = span
    # router can have been imported transitively before this hook.
    import sys
    router = sys.modules.get("wbridge.backend.router")
    if router is not None:
        router.gantt_span = span

"""Apply an explicit recipe rank placement before initializing NCCL."""

import json
import os
from pathlib import Path


def configure_rank_affinity(local_rank, *, physical_gpu=None):
    configured = os.environ.get("SHARDSTREAM_RANK_AFFINITY")
    if not configured:
        return
    if physical_gpu is None:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical_gpu = devices[int(local_rank)]
    physical_gpu = str(physical_gpu)
    entry = json.loads(configured)[physical_gpu]
    # Bind every existing thread, including pools created while importing
    # torch. New threads inherit the affinity; model allocations follow the
    # local CPU domain's first-touch policy. This does not force membind.
    cpus = set(entry["cpus"])
    for thread in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(thread.name), cpus)
        except ProcessLookupError:
            pass
    os.environ["NCCL_IB_HCA"] = entry["hca"]
    os.environ["SHARDSTREAM_HCA_POLICY"] = "topology"
    print(
        "SHARDSTREAM_RANK_AFFINITY "
        + json.dumps(
            {
                "pid": os.getpid(),
                "physical_gpu": physical_gpu,
                "hca": entry["hca"],
                "numa": entry["numa"],
                "cpus": sorted(os.sched_getaffinity(0)),
                "before_nccl_init": True,
            }
        ),
        flush=True,
    )

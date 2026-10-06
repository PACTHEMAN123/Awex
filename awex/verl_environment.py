"""Environment normalization for veRL's Ray worker layout."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


def configure_device_v2_ray_locality(
    env: MutableMapping[str, str] | None = None,
    *,
    worker_local_rank: int | None = None,
) -> None:
    """Map a single-GPU Ray actor back to its node-local physical GPU rank."""

    env = os.environ if env is None else env
    policy = env.get("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced")
    if policy.strip().lower() != "balanced":
        return
    if "AWEX_NODE_LOCAL_RANK_OFFSET" in env:
        return

    visible_devices = [
        device.strip()
        for device in env.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    if not visible_devices or not all(device.isdigit() for device in visible_devices):
        return

    local_world_size = env.get("RAY_LOCAL_WORLD_SIZE") or env.get("LOCAL_WORLD_SIZE")
    try:
        if worker_local_rank is None:
            if len(visible_devices) != 1:
                return
            physical_rank = int(visible_devices[0])
        else:
            worker_local_rank = int(worker_local_rank)
            if not 0 <= worker_local_rank < len(visible_devices):
                return
            physical_rank = int(visible_devices[worker_local_rank])
            env.setdefault("LOCAL_RANK", str(worker_local_rank))
        local_rank = int(env.get("LOCAL_RANK", "0"))
        world_size = int(local_world_size or "")
    except (TypeError, ValueError):
        return
    if world_size <= 1 or not 0 <= physical_rank < world_size:
        return

    env["AWEX_NODE_LOCAL_RANK_OFFSET"] = str(physical_rank - local_rank)
    env["AWEX_NODE_LOCAL_WORLD_SIZE"] = str(world_size)

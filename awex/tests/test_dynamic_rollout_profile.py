"""Reject apparent overlap caused by clock skew or work crossing prepare bounds."""

from copy import deepcopy

import pytest

from awex.tests.experimental.dynamic_rollout_profile_report import audit, chrome_trace


def sample_profile():
    keys = ["training//0", "rollout/engine-0/0", "rollout/engine-1/0"]
    metrics = {
        "plan_cache_hit": True,
        **dict.fromkeys(
            (
                "transport_init_time_ms",
                "plan_initialization_time_ms",
                "metadata_upload_time_ms",
                "host_lowering_time_ms",
                "build_batch_time_ms",
            ),
            0,
        ),
    }
    ranks = {
        key: {
            "pid": i,
            "pointers": {"weight": i},
            "hostname": key,
            "epoch": 0,
            "verified": True,
            "transfer_ms": 1.0,
            "metrics": metrics,
            "start_ns": 0,
            "phases": [
                {
                    "name": "device_communicator_fifo_cache",
                    "start_ns": 100,
                    "end_ns": 200,
                }
            ],
            "preparation_metrics": metrics,
        }
        for i, key in enumerate(keys)
    }
    publications = [
        {
            "event": "publication",
            "epoch": 0,
            "version": i,
            "source_payload_bytes": 1024,
            "engine_ids": ["engine-0", "engine-1"],
            "ranks": deepcopy(ranks),
        }
        for i in range(20)
    ]
    compute = {
        key: {
            "compute_events": []
            if key == keys[-1]
            else [
                {
                    "name": "compute",
                    "start_ns": 110,
                    "end_ns": 190,
                    "gpu_ms": 0.00004,
                    "gemms": 1,
                },
                {
                    "name": "compute",
                    "start_ns": 90,
                    "end_ns": 120,
                    "gpu_ms": 0.00001,
                    "gemms": 1,
                },
            ]
        }
        for key in keys
    }
    join = {
        "event": "join",
        "epoch": 1,
        "engine_ids": ["engine-0", "engine-1"],
        "serving_engine_ids": ["engine-0"],
        "start_ns": 0,
        "launch_start_ns": 10,
        "prepare_start_ns": 100,
        "ready_ns": 200,
        "join_ms": 0.0002,
        "launch_and_startup_ms": 0.00009,
        "prepare_ms": 0.0001,
        "ranks": ranks,
        "compute_ranks": compute,
    }
    return {
        "passed": True,
        "cuda_kernel_validated": True,
        "profile_compute": True,
        "fp8": False,
        "tp": 1,
        "updates": 20,
        "records": [{"event": "initial_prepare", "ranks": ranks}, *publications, join],
        "clock_samples": {
            key: {"offset_ns": 5000, "uncertainty_ns": 1000} for key in keys
        },
    }


def test_overlap_uses_same_worker_bounds_and_excludes_crossing_batches():
    profile = sample_profile()
    result = audit(profile)
    assert result["cache_hits"] == 60
    assert (
        result["joins"][0]["compute_by_rank"]["training//0"][
            "batches_inside_device_prepare"
        ]
        == 1
    )
    trace = chrome_trace(profile)
    compute = next(e for e in trace["traceEvents"] if e["name"] == "compute")
    assert compute["ts"] == 5.11
    assert compute["args"]["cuda_event_ms"] == 0.00004


def test_host_overlap_without_completed_gpu_work_is_rejected():
    profile = sample_profile()
    for rank in profile["records"][-1]["compute_ranks"].values():
        for event in rank["compute_events"]:
            event["gpu_ms"] = None
    with pytest.raises(AssertionError):
        audit(profile)


def test_cache_miss_after_join_is_rejected():
    profile = sample_profile()
    profile["records"][1]["ranks"]["training//0"]["metrics"]["plan_cache_hit"] = False
    with pytest.raises(AssertionError):
        audit(profile)

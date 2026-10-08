"""Require exact model checks, warm caches, and completed same-worker CUDA work."""

from copy import deepcopy

import pytest

from awex.tests.experimental.model_weight_profile_report import audit


def model_profile():
    metrics = {
        "plan_cache_hit": True,
        "kernel_transfer_time_ms": 10.0,
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
    records = []
    engines = 1
    for version in range(-1, 20):
        ranks = {
            str(i): [
                {
                    "pid": i + 10,
                    "pointers": {"weight": i + 100},
                    "verified": True,
                    "version": version,
                    "model_bytes": 1024,
                    "parameter_count": 1,
                    "transfer_metrics": deepcopy(metrics),
                }
            ]
            for i in range(engines)
        }
        records.append(
            {
                "event": "publication",
                "version": version,
                "num_engines": engines,
                "ranks": ranks,
                "end_to_end_update_ms": 12.0,
            }
        )
        if version not in (5, 12):
            continue
        epoch, target = (1, 2) if version == 5 else (2, 4)
        event = {"start_ns": 110, "end_ns": 190, "gpu_ms": 0.00001, "gemms": 256}

        def prepared(pid, pointer, epoch=epoch):
            return {
                "pid": pid,
                "pointers": {"0/weight": pointer},
                "epoch": epoch,
                "preparation_metrics": {
                    "device_plan_ready": True,
                    "kernel_launched": False,
                },
                "phases": [
                    {
                        "name": "device_communicator_fifo_cache",
                        "start_ns": 100,
                        "end_ns": 200,
                    }
                ],
            }

        records.append(
            {
                "event": "join",
                "epoch": epoch,
                "num_engines": target,
                "join_ms": 100,
                "launch_and_startup_ms": 90,
                "prepare_ms": 10,
                "training_ranks": [
                    {"prepare": prepared(1, 2), "compute_events": [deepcopy(event)]}
                ],
                "inference_ranks": {
                    str(i): [prepared(i + 10, i + 100)] for i in range(target)
                },
                "inference_compute": {
                    str(i): [
                        {"compute_events": [deepcopy(event)] if i < engines else []}
                    ]
                    for i in range(target)
                },
            }
        )
        engines = target
    records[0]["ranks"]["0"][0]["transfer_metrics"]["plan_cache_hit"] = False
    return {
        "passed": True,
        "real_model_weights": True,
        "dtype": "bfloat16",
        "train_world_size": 1,
        "inference_tp_size": 1,
        "records": records,
    }


def test_initial_static_miss_is_allowed_and_post_join_snapshots_are_audited():
    result = audit(model_profile())
    assert result["publications"] == 21
    assert result["model_bytes_per_engine"] == 1024
    assert result["joins"][0]["first_version"] == 6
    assert len(result["joins"][1]["overlap_by_worker"]) == 3


def test_first_post_join_cache_miss_is_rejected():
    profile = model_profile()
    next(r for r in profile["records"] if r.get("version") == 6)["ranks"]["1"][0][
        "transfer_metrics"
    ]["plan_cache_hit"] = False
    with pytest.raises(AssertionError):
        audit(profile)


def test_compute_crossing_prepare_bound_is_not_overlap_evidence():
    profile = model_profile()
    next(r for r in profile["records"] if r["event"] == "join")["training_ranks"][0][
        "compute_events"
    ][0]["end_ns"] = 210
    with pytest.raises(AssertionError, match="No completed CUDA work"):
        audit(profile)


def test_model_worker_storage_replacement_is_rejected():
    profile = model_profile()
    next(r for r in profile["records"] if r.get("version") == 6)["ranks"]["0"][0][
        "pointers"
    ]["weight"] += 1
    with pytest.raises(AssertionError):
        audit(profile)

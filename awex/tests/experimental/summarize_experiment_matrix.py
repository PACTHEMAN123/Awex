#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from awex.tests.experimental.analyze_profile_run import (
    load_profile_records,
    stats,
    summarize_profiles,
)


EXPERIMENTS = ("A1", "A2", "A3", "A4")
BACKENDS = ("verl-nccl-bucket", "awex-nccl", "awex-nccl-device-v2-gin")


def run_prefix(experiment, backend):
    if experiment == "A1" and backend == "awex-nccl-device-v2-gin":
        return "A1-device-v2-gin-budget"
    return f"{experiment}-{backend}"


def range_summary(values):
    if not values:
        return None
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def summarize_gpu_node(path):
    by_timestamp = defaultdict(list)
    by_gpu_memory = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            sample = {
                "memory": float(row["memory_used_mib"]),
                "gpu_util": float(row["gpu_util_percent"]),
                "memory_util": float(row["memory_util_percent"]),
            }
            by_timestamp[int(row["timestamp_ns"])].append(sample)
            by_gpu_memory[int(row["gpu_index"])].append(sample["memory"])

    node_memory = []
    node_gpu_util = []
    node_memory_util = []
    active_node_memory = []
    active_node_gpu_util = []
    active_node_memory_util = []
    for samples in by_timestamp.values():
        total_memory = sum(sample["memory"] for sample in samples)
        mean_gpu_util = sum(sample["gpu_util"] for sample in samples) / len(samples)
        mean_memory_util = sum(sample["memory_util"] for sample in samples) / len(samples)
        node_memory.append(total_memory)
        node_gpu_util.append(mean_gpu_util)
        node_memory_util.append(mean_memory_util)
        if total_memory >= 1024 * len(samples):
            active_node_memory.append(total_memory)
            active_node_gpu_util.append(mean_gpu_util)
            active_node_memory_util.append(mean_memory_util)

    per_gpu_peaks = [max(values) for values in by_gpu_memory.values() if values]
    per_gpu_active_p50 = []
    for values in by_gpu_memory.values():
        active = [value for value in values if value >= 1024]
        metric = stats(active)
        if metric is not None:
            per_gpu_active_p50.append(metric["p50"])
    return {
        "sample_count": len(by_timestamp),
        "node_memory_total_mib": stats(node_memory),
        "active_node_memory_total_mib": stats(active_node_memory),
        "active_node_gpu_util_percent": stats(active_node_gpu_util),
        "active_node_memory_util_percent": stats(active_node_memory_util),
        "per_gpu_peak_memory_mib": range_summary(per_gpu_peaks),
        "per_gpu_active_p50_memory_mib": range_summary(per_gpu_active_p50),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--node-role", choices=("train-r0", "train-r1", "infer"), required=True
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    runs = []
    for experiment in EXPERIMENTS:
        for backend in BACKENDS:
            prefix = run_prefix(experiment, backend)
            log_path = args.log_dir / f"{prefix}-{args.node_role}.log"
            gpu_path = args.log_dir / f"{prefix}-{args.node_role}-gpu.csv"
            if not log_path.exists() or not gpu_path.exists():
                continue
            profile = summarize_profiles(load_profile_records([log_path]))
            runs.append(
                {
                    "experiment": experiment,
                    "backend": backend,
                    "node_role": args.node_role,
                    "complete": profile["measure_steps"] == [1, 2, 3, 4, 5],
                    "profile": profile,
                    "gpu": summarize_gpu_node(gpu_path),
                }
            )

    result = {"node_role": args.node_role, "runs": runs}
    rendered = json.dumps(result, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

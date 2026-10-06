#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROFILE_MARKER = "AWEX_PROFILE "
LATENCY_METRICS = (
    "end_to_end_update_time_ms",
    "kernel_transfer_time_ms",
    "backend_execute_time_ms",
    "transport_total_time_ms",
    "total_transfer_time_ms",
    "sync_start_barrier_time_ms",
    "completion_barrier_time_ms",
    "reader_wait_time_ms",
    "reader_copyback_total_time_ms",
    "worker_update_time_ms",
    "update_body_time_ms",
    "flush_cache_time_ms",
    "device_input_wait_time_ms",
    "device_output_wait_time_ms",
    "device_copy_time_ms",
    "device_post_time_ms",
    "device_final_wait_time_ms",
    "device_flush_time_ms",
)
SHAPE_FIELDS = (
    "payload_bytes",
    "active_peer_count",
    "lsa_peer_count",
    "gin_peer_count",
    "channel_count",
    "network_channels_per_peer",
    "network_channel_budget",
    "active_peer_channel_limits",
    "active_peer_payload_bytes",
    "work_count",
    "fragment_count",
    "chunk_count",
    "batch_count",
    "registered_window_bytes",
    "payload_buffer_bytes",
)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def load_profile_records(paths):
    records = []
    decoder = json.JSONDecoder()
    for path in paths:
        content = path.read_text(errors="replace")
        offset = 0
        while True:
            marker_at = content.find(PROFILE_MARKER, offset)
            if marker_at < 0:
                break
            payload_at = marker_at + len(PROFILE_MARKER)
            try:
                record, payload_end = decoder.raw_decode(content, payload_at)
            except json.JSONDecodeError:
                offset = payload_at
                continue
            record["source_log"] = str(path)
            records.append(record)
            offset = payload_end
    return records


def reduce_metric(records, metric, roles=None, reduction=max):
    by_step = defaultdict(list)
    for record in records:
        if record.get("phase") != "measure" or metric not in record:
            continue
        if roles is not None and record.get("role") not in roles:
            continue
        by_step[int(record["step_id"])].append(float(record[metric]))
    return {
        step: reduction(values)
        for step, values in sorted(by_step.items())
        if values
    }


def summarize_profiles(records):
    measure_records = [record for record in records if record.get("phase") == "measure"]
    summary = {
        "record_count": len(records),
        "measure_record_count": len(measure_records),
        "measure_steps": sorted({int(record["step_id"]) for record in measure_records}),
        "metrics": {},
        "rank_shapes": [],
    }
    for metric in LATENCY_METRICS:
        roles = {"driver"} if metric == "end_to_end_update_time_ms" else None
        per_step = reduce_metric(measure_records, metric, roles=roles)
        if per_step:
            summary["metrics"][metric] = stats(list(per_step.values()))

    critical_path = reduce_metric(
        measure_records,
        "backend_execute_time_ms",
        roles={"writer", "reader"},
    )
    if critical_path:
        summary["transfer_critical_path_ms"] = stats(list(critical_path.values()))

    writer_payload = defaultdict(float)
    reader_payload = defaultdict(float)
    for record in measure_records:
        if record.get("event") != "weight_transfer" or "payload_bytes" not in record:
            continue
        target = writer_payload if record.get("role") == "writer" else reader_payload
        target[int(record["step_id"])] += float(record["payload_bytes"])
    logical_payload = writer_payload or reader_payload
    if logical_payload:
        summary["logical_payload_bytes"] = stats(list(logical_payload.values()))
    if logical_payload and critical_path:
        throughputs = [
            logical_payload[step] / (critical_path[step] * 1_000_000.0)
            for step in sorted(set(logical_payload) & set(critical_path))
        ]
        summary["logical_throughput_gbps"] = stats(throughputs)

    seen_shapes = set()
    for record in measure_records:
        if record.get("event") != "weight_transfer":
            continue
        key = (record.get("role"), int(record.get("rank", -1)))
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        shape = {"role": key[0], "rank": key[1]}
        for field in SHAPE_FIELDS:
            if field in record:
                shape[field] = record[field]
        summary["rank_shapes"].append(shape)
    summary["rank_shapes"].sort(key=lambda item: (str(item["role"]), item["rank"]))
    return summary


def summarize_gpu_csv(paths):
    samples = defaultdict(lambda: {"memory": [], "gpu_util": [], "memory_util": []})
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["hostname"], int(row["gpu_index"]))
                samples[key]["memory"].append(float(row["memory_used_mib"]))
                samples[key]["gpu_util"].append(float(row["gpu_util_percent"]))
                samples[key]["memory_util"].append(float(row["memory_util_percent"]))
    result = []
    for (hostname, gpu_index), values in sorted(samples.items()):
        result.append(
            {
                "hostname": hostname,
                "gpu_index": gpu_index,
                "memory_used_mib": stats(values["memory"]),
                "gpu_util_percent": stats(values["gpu_util"]),
                "memory_util_percent": stats(values["memory_util"]),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-log", action="append", type=Path, default=[])
    parser.add_argument("--gpu-csv", action="append", type=Path, default=[])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = {}
    if args.profile_log:
        result["profile"] = summarize_profiles(load_profile_records(args.profile_log))
    if args.gpu_csv:
        result["gpus"] = summarize_gpu_csv(args.gpu_csv)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

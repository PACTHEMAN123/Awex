# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


PROFILE_MARKER = "AWEX_PROFILE "
METRICS = (
    ("writer", "convert_time_ms", "ms", "max"),
    ("writer", "build_batch_time_ms", "ms", "max"),
    ("writer", "region_metadata_time_ms", "ms", "max"),
    ("writer", "transport_init_time_ms", "ms", "max"),
    ("writer", "host_descriptor_time_ms", "ms", "max"),
    ("writer", "buffer_allocation_time_ms", "ms", "max"),
    ("writer", "metadata_upload_time_ms", "ms", "max"),
    ("writer", "kernel_transfer_time_ms", "ms", "max"),
    ("writer", "device_copy_span_time_ms", "ms", "max"),
    ("writer", "sender_publish_time_ms", "ms", "max"),
    ("writer", "sender_reader_ack_wait_time_ms", "ms", "max"),
    ("writer", "control_download_time_ms", "ms", "max"),
    ("writer", "buffer_cleanup_time_ms", "ms", "max"),
    ("writer", "transport_total_time_ms", "ms", "max"),
    ("writer", "total_transfer_time_ms", "ms", "max"),
    ("writer", "sync_start_barrier_time_ms", "ms", "max"),
    ("writer", "completion_barrier_time_ms", "ms", "max"),
    ("writer", "resource_cleanup_time_ms", "ms", "max"),
    ("writer", "gc_collect_time_ms", "ms", "max"),
    ("reader", "build_batch_time_ms", "ms", "max"),
    ("reader", "metadata_upload_time_ms", "ms", "max"),
    ("reader", "kernel_transfer_time_ms", "ms", "max"),
    ("reader", "reader_first_ready_time_ms", "ms", "max"),
    ("reader", "reader_wait_time_ms", "ms", "max"),
    ("reader", "reader_copyback_time_ms", "ms", "max"),
    ("reader", "reader_copyback_tail_time_ms", "ms", "max"),
    ("reader", "python_copyback_time_ms", "ms", "max"),
    ("reader", "reader_copyback_total_time_ms", "ms", "max"),
    ("reader", "transport_total_time_ms", "ms", "max"),
    ("reader", "total_transfer_time_ms", "ms", "max"),
    ("reader", "sync_start_barrier_time_ms", "ms", "max"),
    ("reader", "completion_barrier_time_ms", "ms", "max"),
    ("reader", "resource_cleanup_time_ms", "ms", "max"),
    ("reader", "gc_collect_time_ms", "ms", "max"),
    ("reader_worker", "pre_update_device_sync_time_ms", "ms", "max"),
    ("reader_worker", "update_body_time_ms", "ms", "max"),
    ("reader_worker", "flush_cache_call_time_ms", "ms", "max"),
    ("reader_worker", "post_flush_device_sync_time_ms", "ms", "max"),
    ("reader_worker", "flush_cache_time_ms", "ms", "max"),
    ("reader_worker", "worker_update_time_ms", "ms", "max"),
    ("driver", "end_to_end_update_time_ms", "ms", "max"),
    ("writer", "effective_gbps", "GB/s", "min"),
    ("reader", "effective_gbps", "GB/s", "min"),
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_records(path: Path) -> list[dict]:
    records = []
    content = path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        marker_at = content.find(PROFILE_MARKER, search_from)
        if marker_at < 0:
            break
        payload_at = marker_at + len(PROFILE_MARKER)
        while payload_at < len(content) and content[payload_at].isspace():
            payload_at += 1
        try:
            record, payload_end = decoder.raw_decode(content, payload_at)
        except json.JSONDecodeError:
            search_from = payload_at
            continue
        if record.get("phase") == "measure":
            records.append(record)
        search_from = payload_end
    return records


def summarize(records: list[dict]) -> list[dict]:
    result = []
    for role, metric, unit, reduction in METRICS:
        by_step = defaultdict(list)
        for record in records:
            if record.get("role") != role or metric not in record:
                continue
            by_step[int(record["step_id"])].append(float(record[metric]))
        values = []
        for step_values in by_step.values():
            values.append(
                min(step_values) if reduction == "min" else max(step_values)
            )
        if values:
            result.append(
                {
                    "role": role,
                    "metric": metric,
                    "unit": unit,
                    "samples": len(values),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return result


def render_markdown(summaries: dict[str, list[dict]]) -> str:
    lines = [
        "| Backend | Role | Metric | N | p50 | p95 | Min | Max | Unit |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for backend, metrics in summaries.items():
        for item in metrics:
            lines.append(
                f"| {backend} | {item['role']} | {item['metric']} | "
                f"{item['samples']} | "
                f"{item['p50']:.4f} | {item['p95']:.4f} | "
                f"{item['min']:.4f} | {item['max']:.4f} | {item['unit']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AWEX_PROFILE JSON logs")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        metavar="BACKEND=PATH",
        help="Profile log to read; may be provided more than once.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    summaries = {}
    for value in args.log:
        backend, separator, raw_path = value.partition("=")
        if not separator or not backend or not raw_path:
            parser.error(f"invalid --log value: {value!r}")
        summaries[backend] = summarize(load_records(Path(raw_path)))

    rendered = render_markdown(summaries)
    print(rendered, end="")
    if args.json_out:
        args.json_out.write_text(json.dumps(summaries, indent=2) + "\n")
    if args.markdown_out:
        args.markdown_out.write_text(rendered)


if __name__ == "__main__":
    main()

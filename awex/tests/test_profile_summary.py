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

import json

from awex.tests.summarize_weights_exchange_profile import (
    PROFILE_MARKER,
    load_records,
    percentile,
    summarize,
    summarize_logs,
)


def test_percentile_interpolates_values():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0], 0.95) == 2.9


def test_summarize_uses_per_step_critical_rank():
    records = [
        {
            "role": "writer",
            "step_id": 0,
            "kernel_transfer_time_ms": 10.0,
            "effective_gbps": 50.0,
        },
        {
            "role": "writer",
            "step_id": 0,
            "kernel_transfer_time_ms": 12.0,
            "effective_gbps": 40.0,
        },
        {
            "role": "writer",
            "step_id": 1,
            "kernel_transfer_time_ms": 20.0,
            "effective_gbps": 30.0,
        },
    ]

    result = {item["metric"]: item for item in summarize(records)}

    assert result["kernel_transfer_time_ms"]["samples"] == 2
    assert result["kernel_transfer_time_ms"]["p50"] == 16.0
    assert result["effective_gbps"]["p50"] == 35.0


def test_summarize_reports_cross_role_backend_critical_path():
    records = [
        {
            "role": "writer",
            "step_id": 0,
            "backend_execute_time_ms": 8.0,
            "backend_effective_gbps": 90.0,
        },
        {
            "role": "reader",
            "step_id": 0,
            "backend_execute_time_ms": 10.0,
            "backend_effective_gbps": 75.0,
        },
    ]

    result = {
        (item["role"], item["metric"]): item for item in summarize(records)
    }

    assert result[("transfer_critical_path", "backend_execute_time_ms")][
        "p50"
    ] == 10.0
    assert result[("transfer_critical_path", "backend_effective_gbps")][
        "p50"
    ] == 75.0


def test_load_records_recovers_multiple_process_records_on_one_line(tmp_path):
    records = [
        {"role": "writer", "phase": "measure", "step_id": 1},
        {"role": "reader", "phase": "measure", "step_id": 1},
    ]
    log = tmp_path / "profile.log"
    log.write_text(
        "prefix "
        + PROFILE_MARKER
        + json.dumps(records[0])
        + " colored-prefix "
        + PROFILE_MARKER
        + json.dumps(records[1])
        + "\n"
    )

    assert load_records(log) == records


def test_summarize_logs_merges_same_backend_and_can_hide_non_transfer_metrics(
    tmp_path,
):
    writer_log = tmp_path / "writer.log"
    reader_log = tmp_path / "reader.log"
    writer_log.write_text(
        PROFILE_MARKER
        + json.dumps(
            {
                "role": "writer",
                "phase": "measure",
                "step_id": 0,
                "backend_execute_time_ms": 8.0,
                "gc_collect_time_ms": 100.0,
            }
        )
    )
    reader_log.write_text(
        PROFILE_MARKER
        + json.dumps(
            {
                "role": "reader",
                "phase": "measure",
                "step_id": 0,
                "backend_execute_time_ms": 10.0,
                "completion_barrier_time_ms": 50.0,
            }
        )
    )

    summaries = summarize_logs(
        [("nccl", writer_log), ("nccl", reader_log)],
        transfer_only=True,
    )

    assert summaries == {
        "nccl": [
            {
                "role": "transfer_critical_path",
                "metric": "backend_execute_time_ms",
                "unit": "ms",
                "samples": 1,
                "p50": 10.0,
                "p95": 10.0,
                "min": 10.0,
                "max": 10.0,
            }
        ]
    }

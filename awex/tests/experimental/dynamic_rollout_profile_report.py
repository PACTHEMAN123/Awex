"""Audit mock join profiles and export a Chrome trace, timelines and latency plots.

Run directly with Python (matplotlib is needed only for figures). Input is the
JSON emitted by nccl_device_v2_dynamic_e2e --profile-compute. Wall spans bound
GPU submission/completion; CUDA-event durations are exported as span arguments.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (position - lower)


def audit(summary: dict, warmup: int = 5) -> dict:
    assert summary["passed"] and summary["cuda_kernel_validated"]
    assert summary["profile_compute"] and not summary["fp8"]
    publications = [r for r in summary["records"] if r["event"] == "publication"]
    assert [r["version"] for r in publications] == list(range(summary["updates"]))
    ranks = [r for publication in publications for r in publication["ranks"].values()]
    for rank in ranks:
        assert rank["verified"] and rank["metrics"]["plan_cache_hit"]
        assert all(
            rank["metrics"][key] == 0
            for key in (
                "transport_init_time_ms",
                "plan_initialization_time_ms",
                "metadata_upload_time_ms",
                "host_lowering_time_ms",
                "build_batch_time_ms",
            )
        )
    identities = {}
    for publication in publications:
        for key, rank in publication["ranks"].items():
            identity = (rank["pid"], rank["pointers"], rank["hostname"])
            assert identities.setdefault(key, identity) == identity
    cohorts = []
    for epoch in sorted({r["epoch"] for r in publications}):
        cohort = [r for r in publications if r["epoch"] == epoch]
        values = [
            max(rank["transfer_ms"] for rank in r["ranks"].values()) for r in cohort
        ]
        steady = values[warmup:]
        assert len(steady) >= 10, "Need at least ten steady publications per cohort"
        median = statistics.median(steady)
        cohorts.append(
            {
                "epoch": epoch,
                "engines": len(cohort[0]["engine_ids"]),
                "samples": len(values),
                "excluded_warmup": warmup,
                "steady_samples": len(steady),
                "first_ms": values[0],
                "first_to_steady_ratio": values[0] / median,
                "steady_median_ms": median,
                "steady_p95_ms": percentile(steady, 0.95),
                "steady_min_ms": min(steady),
                "steady_max_ms": max(steady),
                "steady_cv": statistics.pstdev(steady) / statistics.mean(steady),
                "values_ms": values,
            }
        )
    joins = []
    for join in (r for r in summary["records"] if r["event"] == "join"):
        overlap = {}
        for key, compute in join["compute_ranks"].items():
            events = compute["compute_events"]
            if not events:
                continue  # Newly launched ranks do not have old model snapshots.
            phases = join["ranks"][key]["phases"]
            phase = next(
                p for p in phases if p["name"] == "device_communicator_fifo_cache"
            )
            # Compare clocks within one worker. Cross-host clock calibration is
            # only for drawing; it cannot turn apparent overlap into evidence.
            completed_inside = [
                e
                for e in events
                if e["start_ns"] >= phase["start_ns"] and e["end_ns"] <= phase["end_ns"]
            ]
            assert completed_inside, (
                f"No completed compute batch inside device prepare on {key}"
            )
            assert all(
                e["gpu_ms"] is not None and e["gpu_ms"] > 0 for e in completed_inside
            )
            overlap[key] = {
                "batches_inside_device_prepare": len(completed_inside),
                "gemms_inside_device_prepare": sum(
                    e["gemms"] for e in completed_inside
                ),
                "cuda_stream_elapsed_ms_inside_device_prepare": sum(
                    e["gpu_ms"] for e in completed_inside
                ),
                "device_prepare_ms": (phase["end_ns"] - phase["start_ns"]) / 1e6,
                "max_batch_wall_ms": max(
                    (e["end_ns"] - e["start_ns"]) / 1e6 for e in events
                ),
            }
        assert len(overlap) == (len(join["serving_engine_ids"]) + 1) * summary["tp"]
        metrics = [r["preparation_metrics"] for r in join["ranks"].values()]
        joins.append(
            {
                "epoch": join["epoch"],
                "engines": len(join["engine_ids"]),
                "join_ms": join["join_ms"],
                "launch_and_startup_ms": join["launch_and_startup_ms"],
                "prepare_ms": join["prepare_ms"],
                "compute_by_rank": overlap,
                "native_metrics_max_ms": {
                    key: max(m[key] for m in metrics)
                    for key in (
                        "transport_init_time_ms",
                        "plan_initialization_time_ms",
                        "host_lowering_time_ms",
                        "metadata_upload_time_ms",
                    )
                },
            }
        )
    return {
        "cache_hits": len(ranks),
        "publications": len(publications),
        "source_payload_bytes": publications[0]["source_payload_bytes"],
        "cohorts": cohorts,
        "joins": joins,
        "clock_samples": summary["clock_samples"],
    }


def chrome_trace(summary: dict) -> dict:
    traces = []
    joins = [r for r in summary["records"] if r["event"] == "join"]
    first = min(r["start_ns"] for r in summary["records"][0]["ranks"].values())
    clocks = summary["clock_samples"]

    def add(name, start, end, pid, tid, args=None, offset=0):
        traces.append(
            {
                "name": name,
                "ph": "X",
                "ts": (start + offset - first) / 1000,
                "dur": (end - start) / 1000,
                "pid": pid,
                "tid": tid,
                "args": args or {},
            }
        )

    for record in summary["records"]:
        for key, rank in record["ranks"].items():
            offset = clocks[key]["offset_ns"]
            for phase in rank["phases"]:
                add(
                    phase["name"],
                    phase["start_ns"],
                    phase["end_ns"],
                    key,
                    "exchange preparation/publication",
                    {
                        "epoch": rank["epoch"],
                        "version": rank.get("version"),
                        "clock_uncertainty_us": clocks[key]["uncertainty_ns"] / 1000,
                    },
                    offset,
                )
    for join in joins:
        add(
            "join total",
            join["start_ns"],
            join["ready_ns"],
            "driver",
            "join",
            {"epoch": join["epoch"]},
        )
        add(
            "worker launch + startup",
            join["launch_start_ns"],
            join["prepare_start_ns"],
            "driver",
            "stages",
        )
        add(
            "collective prepare",
            join["prepare_start_ns"],
            join["ready_ns"],
            "driver",
            "stages",
        )
        for key, rank in join["compute_ranks"].items():
            for event in rank["compute_events"]:
                add(
                    event["name"],
                    event["start_ns"],
                    event["end_ns"],
                    key,
                    "snapshot compute (separate stream)",
                    {"cuda_event_ms": event["gpu_ms"], "gemms": event["gemms"]},
                    clocks[key]["offset_ns"],
                )
    return {"traceEvents": traces, "displayTimeUnit": "ms"}


def figures(summaries: dict, audits: dict, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    colors = {
        "process_group": "#6099be",
        "plan_build": "#fdcc67",
        "host_bind_and_old_release": "#bf88b2",
        "device_communicator_fifo_cache": "#e48a4b",
        "verify_and_ready_barrier": "#bac1cd",
    }
    # Use the largest participating host count for the detailed resource view.
    name, summary = max(
        summaries.items(), key=lambda pair: len(pair[1]["agent_descriptions"])
    )
    joins = [r for r in summary["records"] if r["event"] == "join"]
    fig, axes = plt.subplots(
        len(joins), 1, figsize=(13, 5 * len(joins)), constrained_layout=True
    )
    for ax, join in zip(axes, joins):
        origin = join["start_ns"]
        ordered = sorted(
            join["ranks"], key=lambda key: (key.startswith("rollout"), key)
        )
        polygons = []
        for i, key in enumerate(ordered):
            offset = summary["clock_samples"][key]["offset_ns"]
            for phase in join["ranks"][key]["phases"]:
                x = (phase["start_ns"] + offset - origin) / 1e9
                width = (phase["end_ns"] - phase["start_ns"]) / 1e9
                ax.broken_barh(
                    [(x, width)], (i - 0.06, 0.30), facecolors=colors[phase["name"]]
                )
            for event in join["compute_ranks"][key]["compute_events"]:
                a = (event["start_ns"] + offset - origin) / 1e9
                b = (event["end_ns"] + offset - origin) / 1e9
                polygons.append(
                    [(a, i - 0.38), (b, i - 0.38), (b, i - 0.14), (a, i - 0.14)]
                )
        ax.add_collection(
            PolyCollection(polygons, facecolors="#24867b", edgecolors="none")
        )
        prepare = (join["prepare_start_ns"] - origin) / 1e9
        ready = (join["ready_ns"] - origin) / 1e9
        ax.axvline(prepare, color="#666666", ls="--", lw=0.8)
        ax.axvline(ready, color="#666666", ls="--", lw=0.8)
        ax.set_title(
            f"{name}: epoch {join['epoch']}, {len(join['serving_engine_ids'])} → {len(join['engine_ids'])} rollout engines | "
            f"startup {join['launch_and_startup_ms'] / 1000:.2f}s + prepare {join['prepare_ms'] / 1000:.2f}s",
            loc="left",
        )
        ax.set_yticks(
            range(len(ordered)),
            [
                key.replace("training//", "train/TP").replace("rollout/", "")
                for key in ordered
            ],
        )
        ax.set_ylim(len(ordered) - 0.65, -0.75)
        ax.set_xlim(0, ready * 1.04)
        ax.set_xlabel("Seconds from join request (calibrated host clocks)")
        ax.grid(axis="x", alpha=0.15)
    handles = [Patch(color=color, label=label) for label, color in colors.items()]
    handles.append(
        Patch(color="#24867b", label="BF16 GEMM submission → completion (lower lane)")
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(output / "timeline.png", dpi=180)
    fig.savefig(output / "timeline.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(
        len(audits), 1, figsize=(12, 3.1 * len(audits)), constrained_layout=True
    )
    if len(audits) == 1:
        axes = [axes]
    for ax, (case, result) in zip(axes, audits.items()):
        position = 0
        for cohort, color in zip(result["cohorts"], ("#6099be", "#24867b", "#e48a4b")):
            values = cohort["values_ms"]
            x = list(range(position, position + len(values)))
            ax.plot(
                x,
                values,
                marker=".",
                markersize=4,
                color=color,
                lw=0.8,
                label=f"{cohort['engines']} engines: steady median {cohort['steady_median_ms']:.3f}, p95 {cohort['steady_p95_ms']:.3f} ms",
            )
            ax.scatter(x[0], values[0], marker="D", color=color, s=28, zorder=3)
            ax.axvspan(
                position - 0.5,
                position + cohort["excluded_warmup"] - 0.5,
                color=color,
                alpha=0.08,
            )
            if position:
                ax.axvline(position - 0.5, color="#666666", ls="--", lw=0.8)
            position += len(values)
        ax.set_title(case, loc="left")
        ax.set_xlabel("Global weight version (join gaps omitted)")
        ax.set_ylabel("Slowest rank exchange (ms)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output / "latency.png", dpi=180)
    fig.savefig(output / "latency.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Exclude the first N samples of each cohort from steady statistics; retain them in the plot",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("Warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {path.stem: json.loads(path.read_text()) for path in args.inputs}
    assert len(summaries) == len(args.inputs), "Input stems must be unique"
    audits = {name: audit(summary, args.warmup) for name, summary in summaries.items()}
    for name, summary in summaries.items():
        (args.output / (name + ".trace.json")).write_text(
            json.dumps(chrome_trace(summary))
        )
    (args.output / "analysis.json").write_text(json.dumps(audits, indent=2) + "\n")
    figures(summaries, audits, args.output)
    print(
        json.dumps(
            {
                name: {
                    "publications": r["publications"],
                    "cache_hits": r["cache_hits"],
                    "cohorts": [
                        {k: v for k, v in c.items() if k != "values_ms"}
                        for c in r["cohorts"]
                    ],
                    "joins": [
                        {k: v for k, v in j.items() if k != "compute_by_rank"}
                        for j in r["joins"]
                    ],
                }
                for name, r in audits.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

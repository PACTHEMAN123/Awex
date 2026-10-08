"""Audit model-loaded elastic weight exchange and plot measured join timelines.

Overlap evidence compares timestamps inside the same worker. The trace keeps
workers separate: machine clocks are not calibrated. CUDA events measure stream
elapsed time; host spans bound submission/completion, not kernel occupancy.
"""

from __future__ import annotations

import argparse
import gzip
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


PREPARE_METRICS = (
    "transport_init_time_ms",
    "plan_initialization_time_ms",
    "metadata_upload_time_ms",
    "host_lowering_time_ms",
    "build_batch_time_ms",
)


def readers(publication: dict):
    for engine, ranks in publication["ranks"].items():
        for tp, rank in enumerate(ranks):
            yield f"engine-{engine}/tp-{tp}", rank


def audit(summary: dict, warmup: int = 2) -> dict:
    assert summary["passed"] and summary["real_model_weights"]
    assert summary["dtype"] == "bfloat16"
    publications = [r for r in summary["records"] if r["event"] == "publication"]
    assert [r["version"] for r in publications] == list(
        range(-1, len(publications) - 1)
    )
    identities, hits, checks = {}, 0, 0
    for publication in publications:
        ranks = list(readers(publication))
        assert len(ranks) == publication["num_engines"] * summary["inference_tp_size"]
        for key, rank in ranks:
            assert rank["verified"] and rank["version"] == publication["version"]
            assert rank["model_bytes"] > 0 and rank["parameter_count"] > 0
            identity = (rank["pid"], rank["pointers"])
            assert identities.setdefault(key, identity) == identity
            metrics = rank["transfer_metrics"]
            # Initial static publication initializes its plan on demand.
            if publication["version"] >= 0:
                assert metrics["plan_cache_hit"], (key, publication["version"])
                assert all(metrics[name] == 0 for name in PREPARE_METRICS)
                hits += 1
            checks += 1
    cohorts = []
    for engines in sorted({r["num_engines"] for r in publications}):
        cohort = [r for r in publications if r["num_engines"] == engines]
        kernels = [
            max(
                rank["transfer_metrics"]["kernel_transfer_time_ms"]
                for _, rank in readers(r)
            )
            for r in cohort
        ]
        end_to_end = [r["end_to_end_update_ms"] for r in cohort]
        steady = kernels[warmup:]
        steady_end_to_end = end_to_end[warmup:]
        assert len(steady) >= 5
        median = statistics.median(steady)
        cohorts.append(
            {
                "engines": engines,
                "versions": [r["version"] for r in cohort],
                "samples": len(kernels),
                "excluded_warmup": warmup,
                "kernel_ms": kernels,
                "end_to_end_ms": end_to_end,
                "first_kernel_ms": kernels[0],
                "first_to_steady_ratio": kernels[0] / median,
                "steady_kernel_median_ms": median,
                "steady_kernel_p95_ms": percentile(steady, 0.95),
                "steady_kernel_cv": statistics.pstdev(steady) / statistics.mean(steady),
                "first_end_to_end_ms": end_to_end[0],
                "first_end_to_end_to_steady_ratio": end_to_end[0]
                / statistics.median(steady_end_to_end),
                "steady_end_to_end_median_ms": statistics.median(steady_end_to_end),
                "steady_end_to_end_p95_ms": percentile(steady_end_to_end, 0.95),
                "steady_end_to_end_cv": statistics.pstdev(steady_end_to_end)
                / statistics.mean(steady_end_to_end),
            }
        )
    joins, training_identities = [], {}
    for join in (r for r in summary["records"] if r["event"] == "join"):
        workers = {
            f"training/{i}": (row["prepare"], row["compute_events"])
            for i, row in enumerate(join["training_ranks"])
        }
        assert len(workers) == summary["train_world_size"]
        for key, (rank, _) in workers.items():
            identity = (rank["pid"], rank["pointers"])
            assert training_identities.setdefault(key, identity) == identity
        for engine, ranks in join["inference_ranks"].items():
            for tp, rank in enumerate(ranks):
                key = f"engine-{engine}/tp-{tp}"
                workers[key] = (
                    rank,
                    join["inference_compute"][engine][tp]["compute_events"],
                )
                pointers = {
                    name.split("/", 1)[1]: p for name, p in rank["pointers"].items()
                }
                assert identities[key] == (rank["pid"], pointers)
        overlap = {}
        for key, (rank, events) in workers.items():
            assert rank["epoch"] == join["epoch"]
            assert rank["preparation_metrics"]["device_plan_ready"]
            assert rank["preparation_metrics"]["kernel_launched"] is False
            if not events:
                assert not key.startswith("training/")
                continue  # Late model workers have no previous compute workload.
            phase = next(
                p
                for p in rank["phases"]
                if p["name"] == "device_communicator_fifo_cache"
            )
            contained = [
                e
                for e in events
                if phase["start_ns"] <= e["start_ns"] and e["end_ns"] <= phase["end_ns"]
            ]
            assert contained, f"No completed CUDA work inside device prepare: {key}"
            assert all(e["gpu_ms"] > 0 for e in contained)
            overlap[key] = {
                "completed_batches": len(contained),
                "completed_gemms": sum(e["gemms"] for e in contained),
                "cuda_stream_elapsed_ms": sum(e["gpu_ms"] for e in contained),
                "device_prepare_ms": (phase["end_ns"] - phase["start_ns"]) / 1e6,
            }
        previous_engines = next(
            r["num_engines"]
            for r in reversed(summary["records"][: summary["records"].index(join)])
            if r["event"] == "publication"
        )
        assert (
            len(overlap)
            == summary["train_world_size"]
            + previous_engines * summary["inference_tp_size"]
        )
        first = next(r for r in publications if r["num_engines"] == join["num_engines"])
        joins.append(
            {
                "epoch": join["epoch"],
                "engines": join["num_engines"],
                "join_ms": join["join_ms"],
                "launch_and_startup_ms": join["launch_and_startup_ms"],
                "prepare_ms": join["prepare_ms"],
                "first_version": first["version"],
                "overlap_by_worker": overlap,
                "phase_max_ms": {
                    name: max(
                        (p["end_ns"] - p["start_ns"]) / 1e6
                        for rank, _ in workers.values()
                        for p in rank["phases"]
                        if p["name"] == name
                    )
                    for name in {
                        p["name"]
                        for rank, _ in workers.values()
                        for p in rank["phases"]
                    }
                },
            }
        )
    return {
        "passed": True,
        "full_model_parameter_checks": checks,
        "cached_reader_publications": hits,
        "publications": len(publications),
        "model_bytes_per_engine": sum(
            r["model_bytes"] for _, r in readers(publications[0])
        )
        // publications[0]["num_engines"],
        "cohorts": cohorts,
        "joins": joins,
    }


def trace(summary: dict) -> dict:
    events = []
    for join in (r for r in summary["records"] if r["event"] == "join"):
        workers = {
            f"training/{i}": (r["prepare"], r["compute_events"])
            for i, r in enumerate(join["training_ranks"])
        }
        workers.update(
            {
                f"engine-{engine}/tp-{tp}": (
                    rank,
                    join["inference_compute"][engine][tp]["compute_events"],
                )
                for engine, ranks in join["inference_ranks"].items()
                for tp, rank in enumerate(ranks)
            }
        )
        for key, (rank, compute) in workers.items():
            for tid, spans in (
                ("prepare", rank["phases"]),
                ("CUDA submit-to-complete", compute),
            ):
                for span in spans:
                    events.append(
                        {
                            "name": span["name"],
                            "ph": "X",
                            "pid": key,
                            "tid": tid,
                            "ts": span["start_ns"] / 1000,
                            "dur": (span["end_ns"] - span["start_ns"]) / 1000,
                            "args": {
                                "epoch": join["epoch"],
                                "cuda_event_ms": span.get("gpu_ms"),
                                "uncalibrated_host_clock": True,
                            },
                        }
                    )
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def figures(summary: dict, result: dict, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    joins = [r for r in summary["records"] if r["event"] == "join"]
    fig, axes = plt.subplots(len(joins), 2, figsize=(15, 4 * len(joins)), squeeze=False)
    colors = ["#6366f1", "#eab308", "#f97316", "#2563eb", "#a855f7", "#64748b"]
    phase_names = [
        p["name"] for p in joins[0]["training_ranks"][0]["prepare"]["phases"]
    ]
    for i, join in enumerate(joins):
        row = join["training_ranks"][0]
        prep, compute = row["prepare"], row["compute_events"]
        origin = join["start_ns"]
        for j, ax in enumerate(axes[i]):
            for n, phase in enumerate(prep["phases"]):
                ax.broken_barh(
                    [
                        (
                            (phase["start_ns"] - origin) / 1e9,
                            (phase["end_ns"] - phase["start_ns"]) / 1e9,
                        )
                    ],
                    (1.2, 0.55),
                    facecolors=colors[n % len(colors)],
                )
            vertices = [
                [
                    ((e["start_ns"] - origin) / 1e9, 0.2),
                    ((e["end_ns"] - origin) / 1e9, 0.2),
                    ((e["end_ns"] - origin) / 1e9, 0.75),
                    ((e["start_ns"] - origin) / 1e9, 0.75),
                ]
                for e in compute
            ]
            ax.add_collection(
                PolyCollection(vertices, facecolors="#14b8a6", edgecolors="none")
            )
            if j == 0:
                ax.broken_barh(
                    [
                        (
                            (join["launch_start_ns"] - origin) / 1e9,
                            (join["prepare_start_ns"] - join["launch_start_ns"]) / 1e9,
                        )
                    ],
                    (2.2, 0.55),
                    facecolors="#94a3b8",
                )
                ax.set_xlim(0, (join["ready_ns"] - origin) / 1e9 * 1.02)
                ax.set_title(
                    f"Join to {join['num_engines']} engines: {join['join_ms'] / 1000:.2f} s total"
                )
                ax.set_yticks(
                    [0.48, 1.48, 2.48],
                    [
                        "BF16 GEMM batches",
                        "Transfer preparation",
                        "Launch / model startup",
                    ],
                )
            else:
                start, end = prep["start_ns"], prep["end_ns"]
                padding = (end - start) * 0.03
                ax.set_xlim(
                    (start - origin - padding) / 1e9, (end - origin + padding) / 1e9
                )
                ax.set_title(
                    f"Preparation detail: {join['prepare_ms'] / 1000:.3f} s (driver worker)"
                )
                ax.set_yticks(
                    [0.48, 1.48], ["BF16 GEMM batches", "Transfer preparation"]
                )
            ax.set_ylim(0, 3 if j == 0 else 2)
            ax.set_xlabel("Seconds since join request (same training worker clock)")
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Qwen3-30B-A3B BF16: real model weights; background compute during join",
        fontsize=15,
    )
    fig.legend(
        [Patch(color=colors[i % len(colors)]) for i in range(len(phase_names))],
        [name.replace("_", " ") for name in phase_names],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.01,
        "Compute bars bound submission/completion; CUDA-event elapsed time is not kernel occupancy.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"model-join-timeline.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for cohort in result["cohorts"]:
        for ax, metric in zip(axes, ("kernel_ms", "end_to_end_ms")):
            # Show every post-initial update, including the first after each join.
            samples = [
                (version, duration)
                for version, duration in zip(cohort["versions"], cohort[metric])
                if version >= 0
            ]
            ax.plot(
                [version for version, _ in samples],
                [duration for _, duration in samples],
                "o-",
                label=f"{cohort['engines']} engines",
            )
            if cohort["versions"][0] >= 0:
                ax.axvline(cohort["versions"][0], color="#64748b", alpha=0.5, ls="--")
                ax.scatter(
                    cohort["versions"][0],
                    cohort[metric][0],
                    marker="*",
                    s=150,
                    color="#111827",
                    zorder=5,
                )
    for ax in axes:
        ax.set_xlabel("Weight version")
        ax.set_ylabel("Milliseconds")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_title("Slowest reader: device weight kernel")
    axes[1].set_title("Publication end to end (validation excluded)")
    fig.text(
        0.5,
        0.01,
        f"Stars: first update after join. Initial cold update (version -1): "
        f"{result['cohorts'][0]['first_end_to_end_ms'] / 1000:.2f} s; excluded from this view.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"model-sync-latency.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    opener = gzip.open if args.profile.suffix == ".gz" else open
    with opener(args.profile, "rt") as stream:
        summary = json.load(stream)
    result = audit(summary, args.warmup)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "model-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with gzip.open(args.output / "model-trace.json.gz", "wt") as stream:
        json.dump(trace(summary), stream)
    figures(summary, result, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cohorts", "joins")}, indent=2
        )
    )


if __name__ == "__main__":
    main()

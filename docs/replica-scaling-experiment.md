# Replica scaling experiment (2026-10-20)

Run Qwen3-30B-A3B BF16 with the current complete logical replica routing,
training source policy `stream`, source-node swizzle and NIC-spread GPU
placement. No disabled-feature controls are part of this experiment.

Each unique configuration gets three independent process launches, each with
five warmup publications and 30 measured publications. All workers receive
exact full-parameter checks outside the timed publication, version and stable
pointer checks, plus checkpoint generation checks with prefix caching disabled.

| Series | Training | Inference | Configurations |
|---|---|---|---|
| Intra-instance DP | 2 nodes, TP4/CP2/EP8/ETP1; dense/expert DP2/2 | 2 nodes; 2 instances, EP8, 8 GPUs each | DP1/TP8, DP2/TP4, DP4/TP2, DP8/TP1 |
| Independent instances | 1 node, TP4/CP2/EP8/ETP1; dense/expert DP1/1 | Up to 3 nodes; TP4/DP1/EP1 | 1, 2, 3, 4, 5, 6 instances |
| Training source DP | TP4/CP2/EP8/ETP1, 8 or 16 training GPUs; dense/expert DP1/1 or DP2/2 | Fixed 2 nodes, 2 instances, TP4/DP2/EP8 | 2 |

The sender DP2 configuration is identical to intra-instance DP2 and reuses
those three launches. Thus 11 unique configurations require 33 launches,
1,155 verified publications and 990 measured samples. No previous experiment's
launches are reused in this new matrix.

Independent instances stay within a physical node. For 1–6 instances, active
host counts are 1, 2, 3, 3, 3, 3; per-host instance counts are respectively
`[1]`, `[1,1]`, `[1,1,1]`, `[2,1,1]`, `[2,2,1]`, `[2,2,2]`.
Selected rollout host order rotates across repeats to vary which host carries
the extra instance. Record both available and active host counts. This changes
physical node count at the first three points, which must remain visible when
interpreting scaling.

The runner reserves the whole eight-GPU node for NIC-spread placement, then
selects physical GPUs within that reservation. Every TP4 instance must map its
four ranks to four distinct active RDMA ports. Both first and second TP4 groups
are audited; the second group can have longer PCIe distances on these nodes.
No GPU is shared between instances. Retain per-rank HCA/CPU affinity, topology,
physical placement, per-port TX/RX deltas, actual forwarding routes and native
payload bytes.

A global experiment lock serializes GPU ownership. Only processes and NCCL SHM
created by the exact experiment are cleanup candidates. Preserve failed attempts
separately, never turn incomplete publications into passing measurements.
Report cold update separately and aggregate steady distributions per launch;
report median launch p50/p95 with all three values/ranges rather than treating
90 updates as 90 independent process launches. The desired flat latency is a
hypothesis, not an acceptance criterion that permits changing or dropping data.

# Standalone replica routing evaluation (2026-10-19/20)

Branch: `codex/unified-replica-routing`. GPU-tested implementation:
`7a8c7555ac2c3922509439fb355c6c81ab1738be`. Subsequent changes remove two
out-of-scope benchmark layouts and document these results; routing code is unchanged.

Qwen3-30B-A3B BF16 on four eight-GPU H20-3e nodes, using Megatron training and
vLLM 0.27.0 inference. Native CUDA code and extension are unchanged. Both paths
use training source policy `stream`; the only comparison switch is replica
routing `engine` versus `logical`. Each launch publishes 35 versions, with five
warmups and 30 measured updates.

| Inference placement | Engine p50 | Logical p50 | Logical p95 | Source payload reduction |
|---|---:|---:|---:|---:|
| 4 instances, TP4 / DP1 / EP1 | 445.0 ms | 446.6 ms | 482.9 ms | 0.13% |
| 2 instances, TP2 / DP4 / EP8 | 512.6 ms | 432.8 ms | 451.6 ms | 13.28% |

These are one independent launch per mode per placement, not repeated-launch
confidence estimates. The second placement reduces p50 by 15.57%; the first
is essentially unchanged. All logical cases inject 61,064,245,248 bytes per
publication, equal to the unique canonical source bytes.

Eight logical layouts and two engine controls passed: 350 publications with
300 steady samples. Training coverage includes dense DP1/2/4, expert DP1/2,
CP1/2 and PP1/4; inference covers DP1/2/4/8 with EP8 where DP exceeds one,
TP-replicated parameters, and one/two/four independent instances. The eight
logical p50 values are 446.6, 412.6, 413.4, 432.8, 427.2, 442.0, 513.3 and
438.1 ms in matrix order. The one-training-node case assigns 30.532 GB of entry
tasks to each of the two inference nodes.

An extra EP-off inference-DP layout failed before transfer at expert metadata
validation (786432 versus 1572864 elements). Its logs are retained as a failed
attempt, excluded from the passing matrix; the other added EP-off DP layout
was not run. Both recipes were removed to keep this evaluation focused on the
requested deployment combinations. This failure does not establish the general
support range of vLLM.

Every publication clears all inference parameters outside the timer, then checks
all workers against full GPU reference parameters with `torch.equal`, verifies
version markers and stable parameter addresses. The final version restores the
checkpoint and passes before/after generation checks pinned to DP0 with prefix
caching disabled. Timed work includes publication notification and device
completion; loading, clearing, comparisons and generation are outside the timer.
All steady device-plan/cache checks passed. Four nodes each passed 237 CPU tests
at the frozen implementation revision. Final resource audit found all 32 GPUs
free and completed recorded task-owned NCCL SHM cleanup.

Full results and reproduction scripts are retained in
`results/shardstream-logical-replicas-20261019/REPORT.md` in the parent workspace,
and `/mnt/fuse/verl-e2e/runs/shardstream-logical-replicas-20261019` on the selected
head node. They include per-publication and per-rank CSVs, raw per-node logs,
SHA256 manifests, runtime versions, source bundle, launch commands and cleanup
audit. BF16 performance here does not establish FP8 or other-model performance.
See [routing design](logical-replicas.md) for activation and limitations.

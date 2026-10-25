# Figure 11 four-backend replica experiment, 2026-10-25

The frozen paper `fig:fanout` (Figure 11) measures standalone complete BF16
Qwen3-30B-A3B weight updates. This campaign runs all points freshly with
veRL NCCL, veRL NIXL, Awex Mooncake RDMA with persistent registration, and
ShardStream. No optimizer loop.

Each backend has 11 unique configurations and three independent launches per
configuration. Each launch contains five warmups and 30 measured updates.
The trainer-DP2 point reuses the identical intra-DP2 launch within this campaign;
no historical measurements are substituted. Total: 132 launches, 4,620 verified
updates, 3,960 measured samples and 68,880 rollout-worker parameter checks.

- Rollout intra-DP: two trainer hosts, TP4/CP2/EP8/ETP1 (dense/expert DP2/2);
  two rollout hosts, two eight-GPU instances, EP8, DP1/2/4/8 with TP8/4/2/1.
- Independent rollout instances: one trainer host, TP4/CP2/EP8/ETP1
  (dense/expert DP1/1); TP4/DP1/EP1 instances 1 through 6 on up to three hosts.
  Host occupancies are [1], [1,1], [1,1,1], [2,1,1], [2,2,1], [2,2,2].
  The rollout-host order rotates between launches; one instance never spans hosts.
- Trainer source DP: fixed two TP4/DP2/EP8 rollout instances on two hosts;
  compare one and two training hosts with TP4/CP2/EP8/ETP1 unchanged.

veRL NCCL and NIXL use their native checkpoint engines, 256 MiB buckets and
native loading. Awex uses official inclusionAI/Awex main commit
`bae159256a768d54465ed693d90d925025032a17`, with `comm_backend=mooncake` and
`AWEX_MOONCAKE_PROTOCOL=rdma`. Its shard planner, writer and reader remain
unchanged. The user-requested registration lifetime fix is pinned separately in
`49e919b95bcb2851ac2f4f56226f7f6b595cb775`: outgoing compact snapshots retain
their allocation and registration across updates and may be overwritten only
after every reader acknowledges completion. Receiver staging is retained per
dtype/device and grows to the largest slice. New converter/model pointers do
not invalidate snapshots. Failed or active transports cannot release/recycle
exported storage; idle successful transports support explicit close. This is
an explicitly patched Awex baseline, not unmodified upstream transport.
Benchmark native-layout adapters are ported separately
in `f7ed45cc9b95fd62b7a78177da2dfaf2ee85cf75`, covering vLLM expert names,
replicated GQA heads, MLA and MTP. An untouched-upstream full-model attempt
failed strict metadata-key validation and is archived separately. Every transport must attest RDMA, source-file hash,
GPU memory registration and successful native reads. TCP fallback is rejected.
NCCL buffer is explicitly 4 MiB, its default; this is not a Mooncake RDMA knob.
Mooncake `MC_HANDSHAKE_MAX_LENGTH=67108864` permits full-model memory-region
metadata (the first model-scale RDMA attempt exceeded the native 1 MiB default
with a 1,377,665-byte handshake). This controls metadata acceptance only; native
Mooncake transfer calls and per-peer acknowledgment ordering are preserved.
A three-host/two-reader native RDMA qualification must pass exact values,
strided destination addresses/sentinels, replaced and poisoned source storage,
four rounds with unchanged registration count/address, actual RDMA byte deltas
and explicit release. Every formal launch then audits stable registration counts
for all 35 updates. Persistent snapshot memory remains part of the memory result.
AState was investigated but not selected: the default external SDK selected an
unusable link-local GID, while its open-source UCX implementation has placeholder
hash-map creation and actual transfer functions. Failed investigations contribute
no accepted samples. All new Awex experiments default to the patched Mooncake
RDMA path; historical Awex NCCL data
retain their original provenance and are not relabelled. ShardStream uses stream source selection, source-node ring
order, logical replica routing, compact FIFO, 64 channels, depth 16 and four
GIN connections. No FIFO layout/memory/Nsight profiling is added to timed runs.
All methods use the same model checkpoint, prefix-caching-disabled before/after
generation checks, NIC-spread GPU placement and full model-weight validation.

Complete-update wall time includes export, packing/conversion, transfer,
destination writes, synchronization and control. Checkpoint validation,
per-update clearing/checking, startup and telemetry collection are outside it.
Every update must pass full-parameter, version and unchanged-address checks on
all active rollout workers; checkpoint generation must agree after restoration.
Retain cold update separately. Plot all 90 measured samples per point with
p25/p50/p75 and p5/p95, as in the paper, and report independent launch p50/p95
and ranges separately. Do not treat 90 samples as 90 independent launches.

All cases use the same four refreshed 8xH20-3e Hole nodes. A shared global
experiment lock serializes GPU ownership. Inventory all four nodes before and
after each launch; refuse occupied GPUs. Only newly created, unreferenced NCCL
SHM segments owned by this launch are cleanup candidates. Capture exact source,
native/dependency hashes, commands, GPU/HCA affinity, native transport metrics,
RDMA deltas, full node logs, exit status and SHA256 manifests. Failed attempts
remain separate and contribute no accepted samples. Download and verify results
locally after each launch. A detached controller must survive SSH disconnects.

The performance trend is a measurement outcome, never an acceptance threshold.
Do not change configuration, validation or sampling to obtain a preferred trend.

After this matrix, `run_matrix.py --suite paper-awex` runs the current paper's
remaining native Awex RDMA points: three launches each for A/B/C/D/E (Fig 9),
F/G (Fig 10, sharing fresh B), and the default RDMA memory configuration (Fig 13).
The explicit axes in `other_awex_plan.py` are checked against pinned recipes.
Memory runs use a separate suite and opt-in 10 ms NVML sampling, retaining all
per-worker samples, PyTorch live/cache peaks and device residency. Sampled NVML
peaks are lower bounds; these runs are not substituted for uninstrumented timing.
No six-setting RDMA buffer sweep is invented from the old NCCL buffer sweep.

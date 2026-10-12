# Host RMA preparation candidate

This is an unmeasured candidate, branched from the immutable r23 baseline.
The running five-model baseline matrix continues to use r23 and its original
native extension. Do not adopt this candidate until GPU comparisons pass.

All 15 primary ShardStream main formals retain preparation snapshots for all
32 ranks. FIFO registration takes approximately 1.9–3.1 seconds and GIN device
communicator creation approximately 1.9–2.4 seconds when summarizing each
model's three launches. These are stage observations, not additive global
critical-path timings.

NCCL 2.31.2 source at commit
`fd168324a3dc0c9080fd4881b6c7f4bb252a95a2` shows:

- `src/init.cc`: `numRmaCtx` defaults to 1; setting it to 0 is valid and disables
  host RMA support, independently of Device API GIN support.
- `src/dev_runtime.cc`: first window registration initializes RMA CE when host
  RMA is supported.
- `src/rma/rma_ce.cc`: RMA CE creates additional signal windows using
  `NCCL_WIN_COLL_SYMMETRIC`. Such registration initializes symmetric collective
  kernels even though ShardStream's own FIFO uses `NCCL_WIN_DEFAULT`.
- `src/dev_runtime.cc`: enabling GIN later registers preexisting memories with
  GIN, including resources introduced by host RMA.

The candidate creates only ShardStream's private communicator with
`config.numRmaCtx=0`. It preserves Device API GIN requirements, data kernels,
routes, FIFO geometry, channels, connections and model tensor bindings. No
process-global NCCL environment is changed. Preparation metrics report the
effective `host_rma_support` on NCCL 2.31+; -1 denotes an unavailable property
with older compiled headers. Environment overrides still take precedence in
NCCL, so the effective property must be checked during qualification.

Validation must compile the candidate against the pinned NCCL 2.31.2 headers
and library, then qualify with full parameter/generation checks and actual
four-HCA traffic. Run three interleaved independent baseline/candidate pairs
on Qwen3-8B, each 5 warmup plus 30 measured versions. Compare FIFO registration,
GIN creation, maximum local preparation, first complete publication and steady
p50/p95. Keep all four raw node archives and source/native SHA for every run.
If initialization benefits are reproducible with no steady or correctness
regression, extend validation to all five models. This candidate does not
establish an Elastic membership-change improvement; Elastic remains deferred.

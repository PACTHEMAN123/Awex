# Device v2: dynamic rollout joins

This feature adds rollout instances **between completed weight publications**.
It keeps old workers and model tensors alive, rebuilds the transfer group and
prepared plans, and includes the new workers in the next full weight update.
Training TP/process membership and rollout TP do not change during a run.
Removing instances, heterogeneous rollout TP, in-flight hot splicing, and
automatic rollback after a failed reconfiguration are not supported yet.

The CUDA FIFO/relay/FP8 kernels are unchanged. Ring off, naive ring, swizzle ring,
and blockwise FP8 remain independently selectable. Changing membership is an
explicit API operation; static users do not need to enable it.

## Driver protocol

`awex.transfer.rollout_membership.RolloutJoinCoordinator` owns logical engine
IDs, monotonically increasing membership epochs, and weight versions. Its
acknowledgements must come from the application's RPC/control plane.

1. Publish and wait for **every** existing rank to complete the previous update.
   Join requests may be queued while that update is running.
2. Call `prepare_join()` at the publication boundary. Start the new rollout
   processes and allocate their inference tensors.
3. Create an independent, epoch-namespaced transfer process group containing
   surviving and new participants. Rebuild plans with its new rank map. Reader
   ranks are append-only; training transfer ranks move past all readers. The
   training default process group remains unchanged.
4. On each survivor, call `transport.reconfigure(group, membership, participant,
   parameters, plan)`. On new workers, create and prepare a fresh transport with
   `membership_epoch=membership.epoch`. Then collectively call
   `transport.initialize_prepared_plan(parameters, plan, sender)` on every
   participant. This creates the native communicator, registers the sparse
   FIFO, lowers/uploads the C++ schedule, and establishes its device cache
   **without** copying/quantizing/forwarding weights or advancing a version.
   `is_device_plan_ready(...)` must return true before acknowledging preparation.
   Close/destroy caller-owned old process
   groups only after the corresponding transport has released its resources.
5. Acknowledge preparation on **all** ranks, then `commit_join()`.
6. Publish the next full snapshot and acknowledge its exact epoch/version on
   all ranks. Only after `finish_publication()` may the new cohort serve.

Participant identities are `("training", "", training_rank)` or
`("rollout", engine_id, tp_rank)`. The group rank map is defined by
`RolloutMembership.participants`; custom groups use their own `group.rank()`
and `group.size()`, not the default training rank.

Reconfiguration serializes with local send/recv calls and releases the old
device FIFO before preparing its replacement. It preserves model tensor
storage and invalidates the old epoch's plans/cache. It adds no second full
weight receive buffer to the device transport. Process groups are borrowed;
the application owns their lifecycle. Epoch/geometry are checked across all
participants when the new device communicator initializes.

`serving_engine_ids` retains the old, fully synchronized cohort while the new
membership/cache is prepared, so existing model weights can continue to serve
rollouts during preparation. Newly joined engines remain unavailable until
the next full snapshot completes. The cohort is empty during a weight
publication or after failure; the initial cohort also requires a complete first
snapshot. A rebuild failure
poisons the transport. The driver must call `coordinator.fail(reason)` and stop
serving that group, then recover explicitly with fresh transport objects.
Plan validation failures before resource release leave the old transport
intact. A distributed caller must handle deadlines/failure on **all** workers;
an in-process lock is not a distributed publication barrier.

## Model-loaded weight exchange end to end

`awex.tests.weights_exchange_multi_vllm_it` now supports joining actual loaded
vLLM models to a running Megatron weight publisher. Use non-colocated
`nccl_device_v2`, homogeneous rollout TP, and Qwen3 dense/MoE models. Existing
model objects, parameter storage, and the training default process group stay
alive. Only the transfer group and its epoch-bound plan/cache are replaced.

The training entrypoint accepts `--join-after 10 20 --target-engines 2 4`,
`--elastic-model-profile`, and `--elastic-output model-profile.json`. Provide
`--inference-endpoint ENGINE,HOST,PORT` for every final engine and start only the
initial `--num-engines` cohort before launching training. An external controller
owns late vLLM process launch and GPU placement:

1. Poll the existing metadata server for `elastic/launch/EPOCH`. Its payload
   contains the target `num_engines`.
2. Start the additional real model servers on their reserved devices, then put
   `{"started": true}` under `elastic/launched/EPOCH`. The harness waits for
   model health and initializes only the new reader adapters.
3. The harness collectively calls writers' `prepare_membership` and readers'
   `/areal_awex_prepare_membership`, with `epoch` and `num_engines`. Each worker
   returns a phase profile only after its device cache and readiness barrier
   complete. The next publication uses the prepared cohort directly.

The optional model profile clears **every real destination model parameter**
before each publication and compares every BF16 parameter with a complete
HF-loaded reference afterward. A changing final norm identifies the exact
weight version. Reference clones consume an additional full TP model shard
per inference GPU; this memory is benchmark-only. Verification and destination
clearing are excluded from publication timing. Initial static publication can
build its cache on demand; all following publications, including each first
post-join update, must hit the cache.

During joining, each training and surviving inference worker continuously
executes BF16 GEMMs on a separate CUDA stream. GEMMs use a small private
snapshot derived from its loaded model; the **weight transfer is the complete
Megatron-to-vLLM model**, not that small snapshot. Phase records and completed
compute batches support same-worker overlap checks. This proves overlap with
mock computation; it does not measure live token generation, optimizer steps,
or production scheduling. CUDA events measure elapsed stream time, and host
spans bound submission/completion, not kernel occupancy. Native preparation
releases the Python GIL during communicator/FIFO/cache setup.

If `--sync-transfer-start` is used, set `AWEX_PROFILE_SYNC_START=1` on both
training and all inference servers before starting them. A mismatched setting
would make their first collective operations disagree.

Export an audited timeline, per-update latency plot, and Chrome trace with:

```bash
python -m awex.tests.experimental.model_weight_profile_report \
  model-profile.json --output model-profile-report --warmup 2
```

The auditor rejects incomplete full-model checks, replaced worker/storage,
post-initial cache misses, preparation work inside publications, and compute
batches crossing the preparation bounds. Figures use one worker's clock;
the multi-worker trace labels host clocks as uncalibrated.

## Synthetic transport acceptance benchmark

Run from the checkout using an environment with PyTorch. This benchmark uses
small synthetic TP shards so it needs no model/vLLM dependencies. It launches
new OS processes only at the selected round boundaries, verifies changing
weights on every old/new rollout, and asserts old PIDs and tensor addresses are
unchanged. BF16 validates exact values; FP8 validates exact bytes, block scales,
and unchanged-precision norm weights.

```bash
python -m awex.tests.experimental.nccl_device_v2_dynamic_e2e \
  --backend cpu-mock --tp 2 --initial-engines 1 \
  --join-after 3 6 --target-engines 2 3 --updates 10 \
  --ring swizzle --output dynamic-bf16.json
```

After updates 3 and 6, the active cohort grows 1 → 2 → 3 engines. Each addition
receives the immediately following update (zero-based versions 3 and 6), then
every subsequent update. The acceptance commands use BF16 throughout;
`--ring off|naive|swizzle` covers all three communication modes.

**CPU mock is control-plane/plan-lowering validation only.** Its hooks mock
native communicator/cache initialization and replace the CUDA launch with Gloo
broadcast and reference quantization.
It does not exercise GIN, device FIFO forwarding, relay kernels, or fused FP8,
and its timings must not be used as device v2 performance results. For actual
model weights, use the model-loaded entrypoint above. veRL integration remains
outside these benchmarks.

## Device v2 GPU acceptance

On one host with 8 free GPUs, the same benchmark can execute the actual kernel:

```bash
python -m awex.tests.experimental.nccl_device_v2_dynamic_e2e \
  --backend device-v2 --tp 2 --initial-engines 1 \
  --join-after 3 6 --target-engines 2 3 --updates 10 \
  --rows 4096 --cols 1024 --ring swizzle --timeout 300 --output dynamic-device-bf16.json
```

Local workers bind training to GPUs 0–1 and rollout engines to GPUs 2–3,
4–5, and 6–7. Reserve these GPUs before running. Supply the existing device v2
CUDA/NCCL environment settings; `--timeout` bounds store/collective/device
operations and includes extension build time.

For multiple hosts, run the driver with `--external --backend device-v2`, a
reachable `--control-address` and an explicit `--control-port`. The driver
emits a `launch_worker` JSON event with an argv array whenever a new worker is
needed. Start those workers on the selected nodes from the same checkout and
environment; replace their `--device` with a unique **node-local** GPU index.
Use `--worker-python` to select the remote interpreter. Existing workers stay
running and consume subsequent epoch commands from the driver's TCPStore.
The driver never SSHes to unselected hosts or kills external processes.
Worker `--device` sets `LOCAL_RANK` before the transport chooses its HCA; that
physical GPU binding does not follow the moving transfer rank. Explicit
`NCCL_IB_HCA` settings are preserved. Keep existing NUMA/CPU bindings in each
node's worker launch environment.

### Automatic late launch on multiple hosts

Use node agents to make joins automatic instead of manually starting every
emitted worker. Start one bounded agent on each already selected/verified node,
then start the driver with a JSON placement map. The map covers all final
participants and assigns each a node label and unique node-local device.
Agents keep ownership of only this run's children and reap them on driver
shutdown or their control-store timeout. Driver/agent source fingerprints must
match before any worker is spawned. GPU preflight rejects duplicate physical
GPU assignments even if the same host has two node labels.

For a portable test of the launcher protocol (two agents on **one CPU host**):

```bash
python -m awex.tests.experimental.nccl_device_v2_dynamic_e2e \
  --backend cpu-mock --local-agents \
  --placements awex/tests/experimental/dynamic_rollout_placements.example.json \
  --tp 2 --join-after 3 6 --target-engines 2 4 --updates 10 \
  --ring swizzle --output dynamic-agent-mock.json
```

For real GPUs, replace `DRIVER_ADDRESS` below with the current verified driver
container address. On node-a and node-b, respectively, from the same branch and
their node-local environment, start:

```bash
python -m awex.tests.experimental.dynamic_rollout_agent \
  --node node-a --control-address DRIVER_ADDRESS --control-port 19170 --timeout 300
```

```bash
python -m awex.tests.experimental.dynamic_rollout_agent \
  --node node-b --control-address DRIVER_ADDRESS --control-port 19170 --timeout 300
```

Then on the driver:

```bash
python -m awex.tests.experimental.nccl_device_v2_dynamic_e2e \
  --backend device-v2 \
  --placements awex/tests/experimental/dynamic_rollout_placements.example.json \
  --control-address DRIVER_ADDRESS --control-port 19170 --timeout 300 \
  --tp 2 --join-after 3 6 --target-engines 2 4 --updates 10 \
  --rows 4096 --cols 1024 --ring swizzle --output dynamic-agent-gpu.json
```

The example requires node-a GPUs 0–3 and node-b GPUs 0–5 to be free. Adjust
the map to the selected deployment before running. A late engine is first
spawned at its join boundary, using the node agent's interpreter; no old worker
is restarted. Agents accept only this benchmark's worker configuration, not
arbitrary shell commands. This harness remains synthetic weight exchange, not
the full 30B vLLM/Megatron integration test.

Output records include every participant's epoch/version, PID, storage
addresses, weight checks, per-update metrics, source payload, and membership
preparation time. Preparation checks that weight/scale buffers remain unchanged.
Every publication, including the first post-join update, must hit the plan cache
and report zero communicator initialization, host batch construction, native
plan initialization, host lowering, and metadata upload time. The benchmark
fails if any of that work leaks into publication. Preparation time is recorded
separately; do not count validation/reference construction as transfer time.

## BF16 H20 acceptance (2026-10-09)

The synthetic weight-exchange harness passed all nine combinations of
single-host, two-host, and four-host placement with ring off, naive, and swizzle.
All weights were BF16. Each case ran ten changing snapshots at TP2, using a
4096 × 1024 matrix and a norm tensor per training shard. Single-host membership
grew 1 → 2 → 3 rollout instances; multi-host membership grew 1 → 2 → 4. New
processes started after rounds 3 and 6 and received the immediately following
snapshot. Existing PIDs, tensor addresses, and training weights survived both
joins. Hardware acceptance used code revision `8a34ea3`, PyTorch 2.13.0+cu132,
CUDA 13.2, and NCCL 2.30.4; Ray controlled the four selected H20 containers.

Preparation initialized communicator, sparse FIFO, native schedule, and device
cache without launching the weight kernel or changing model storage. Every
publication hit the cache and reported zero transport initialization, batch
construction, native plan initialization, host lowering, and metadata upload
time. This qualifies the explicit preparation boundary; overlapping its latency
with an actual training/rollout workload remains a subsequent integration step.

The hardware run found and fixed a pure-LSA ring allocation bug: ring works use
the network FIFO depth (16 by default), but a communicator without GIN peers
had allocated only the local depth (8). The first post-join BF16 snapshot crossed
that depth and overwrote adjacent channel slots. Symmetric registration now
includes ring depth on every host, and native preparation rejects schedules
whose depth or step size exceeds the registered geometry. The benchmark checks
`max_work_fifo_depth <= registered_fifo_depth` and
`max_work_step_bytes <= slot_bytes` before accepting preparation.

This acceptance uses real CUDA/device-v2 execution with synthetic model tensors.
Full-model vLLM/Megatron now has its separate entrypoint above; veRL dynamic
membership remains outside this harness.

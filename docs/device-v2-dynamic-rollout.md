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

Dynamic model joins collect the actual node identity of every new transfer
participant. Swizzle groups rollout engines by these epoch-specific identities,
so local relays stay on LSA even when the active cohort differs from the node's
GPU capacity or the final planned cohort. Static transport users retain the
existing environment-based topology fallback.

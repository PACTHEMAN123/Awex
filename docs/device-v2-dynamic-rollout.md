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

## Mock end-to-end acceptance benchmark

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
  --ring swizzle --fp8 --output dynamic-fp8.json
```

After updates 3 and 6, the active cohort grows 1 → 2 → 3 engines. Each addition
receives the immediately following update (zero-based versions 3 and 6), then
every subsequent update. `--ring off|naive|swizzle` and removing `--fp8` cover
the feature combinations.

**CPU mock is control-plane/plan-lowering validation only.** Its hooks mock
native communicator/cache initialization and replace the CUDA launch with Gloo
broadcast and reference quantization.
It does not exercise GIN, device FIFO forwarding, relay kernels, or fused FP8,
and its timings must not be used as device v2 performance results. Full
`weights_exchange_multi_vllm_it.py`, 30B vLLM, and veRL integration have not
been wired to dynamic membership in this first transport-focused stage.

## Device v2 GPU acceptance

On one host with 8 free GPUs, the same benchmark can execute the actual kernel:

```bash
python -m awex.tests.experimental.nccl_device_v2_dynamic_e2e \
  --backend device-v2 --tp 2 --initial-engines 1 \
  --join-after 3 6 --target-engines 2 3 --updates 10 \
  --ring swizzle --fp8 --timeout 300 --output dynamic-device-fp8.json
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
  --ring swizzle --fp8 --output dynamic-agent-mock.json
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
  --rows 4096 --cols 1024 --ring swizzle --fp8 --output dynamic-agent-gpu.json
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

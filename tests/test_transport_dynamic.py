from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from shardstream import transport as module
from shardstream.membership import RolloutMembership
from shardstream.plan import CommunicationOperation, TransferPlan
from shardstream.transport import Transport


def make_plan(rank, membership, sender=True):
    writer = membership.inference_world_size
    readers = range(membership.inference_world_size)
    shard = SimpleNamespace(name="weight", shape=(128, 128), dtype="bfloat16")
    operations = {}
    for reader in readers:
        if not sender and reader != rank:
            continue
        op = CommunicationOperation(
            writer,
            shard,
            (0, 0),
            reader,
            shard,
            (0, 0),
            (128, 128),
            (slice(None), slice(None)),
            (slice(None), slice(None)),
        )
        operations[reader if sender else writer] = [op]
    return TransferPlan(operations=operations)


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setattr(module, "_ensure_cuda_tensor", lambda *_: None)
    monkeypatch.setenv("SHARDSTREAM_FP8_BLOCKWISE", "0")
    initial = RolloutMembership(0, 1, 1, ("old",))
    transport = Transport(
        object(),
        1,
        2,
        infer_instance_world_size=1,
        num_infer_engines=1,
        ring_broadcast=True,
        ring_swizzle=True,
    )
    params = {"weight": torch.ones((128, 128), dtype=torch.bfloat16)}
    plan = make_plan(1, initial)
    transport.prepare_send(params, plan, allow_staging=False)
    destroyed = []
    transport._extension = SimpleNamespace(destroy=destroyed.append)
    transport._handle = 42
    transport._initialized = True
    next_membership = initial.grow(("new",))
    group = SimpleNamespace(rank=lambda: 2, size=lambda: 3)
    return transport, params, plan, destroyed, next_membership, group


def test_expansion_releases_old_fifo_rebuilds_ring_and_preserves_weights(prepared):
    transport, params, old_plan, destroyed, members, group = prepared
    old_pointer = params["weight"].data_ptr()
    plan = make_plan(2, members)
    transport.reconfigure(group, members, ("training", "", 0), params, plan)
    assert destroyed == [42]
    assert transport._handle is None and not transport._initialized
    assert transport.rank == 2 and transport.world_size == 3
    assert transport.membership_epoch == 1 and transport.num_infer_engines == 2
    assert params["weight"].data_ptr() == old_pointer
    assert transport._prepared_send[1] is plan
    assert len(transport._prepared_send[2].expected_counts) == 3
    with pytest.raises(RuntimeError, match="Stale"):
        transport.send(params, old_plan, 4)


def test_previous_rank_plan_is_rejected_without_closing_old_transport(prepared):
    transport, params, old_plan, destroyed, members, group = prepared
    with pytest.raises(RuntimeError, match="previous epoch"):
        transport.reconfigure(group, members, ("training", "", 0), params, old_plan)
    assert destroyed == []
    assert transport._handle == 42
    assert transport.membership_epoch == 0


def test_rebuild_failure_cannot_publish_with_partial_state(prepared, monkeypatch):
    transport, params, old_plan, destroyed, members, group = prepared

    def fail(*args, **kwargs):
        raise ValueError("unsupported new layout")

    monkeypatch.setattr(transport, "prepare_send", fail)
    plan = make_plan(2, members)
    with pytest.raises(ValueError, match="layout"):
        transport.reconfigure(group, members, ("training", "", 0), params, plan)
    assert destroyed == [42]
    assert transport._prepared_send is None
    with pytest.raises(RuntimeError, match="rebuild failed"):
        transport.send(params, old_plan, 4)


def test_skipped_epoch_and_changed_tp_are_rejected(prepared):
    transport, params, _, destroyed, members, group = prepared
    for bad in (
        RolloutMembership(2, 1, 1, members.engine_ids),
        RolloutMembership(1, 1, 2, members.engine_ids),
    ):
        with pytest.raises(RuntimeError):
            transport.reconfigure(
                group,
                bad,
                ("training", "", 0),
                params,
                make_plan(bad.inference_world_size, bad),
            )
    assert destroyed == []


def test_reconfiguration_waits_for_an_inflight_publication(prepared, monkeypatch):
    transport, params, old_plan, destroyed, members, group = prepared
    entered, release, rebuilding = Event(), Event(), Event()

    def blocked_run(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        assert destroyed == []
        return {}

    def rebuild():
        rebuilding.set()
        transport.reconfigure(
            group, members, ("training", "", 0), params, make_plan(2, members)
        )

    monkeypatch.setattr(transport, "_run", blocked_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        send = pool.submit(transport.send, params, old_plan, 0)
        assert entered.wait(timeout=5)
        join = pool.submit(rebuild)
        try:
            assert rebuilding.wait(timeout=5)
            assert not join.done()
            assert destroyed == []
        finally:
            release.set()
        assert send.result(timeout=5)["membership_epoch"] == 0
        join.result(timeout=5)
    assert destroyed == [42]
    assert transport.membership_epoch == 1


def test_independent_group_rank_is_used_instead_of_training_default(
    prepared, monkeypatch
):
    transport, params, _, _, members, group = prepared
    monkeypatch.setattr(module.dist, "get_rank", lambda _: 0)
    transport.reconfigure(
        group, members, ("training", "", 0), params, make_plan(2, members)
    )
    assert transport.rank == 2


def test_invalid_peer_partition_is_rejected_before_destroy(prepared):
    transport, params, _, destroyed, members, group = prepared
    plan = make_plan(2, members)
    plan.operations[0][0].recv_rank = 1
    with pytest.raises(RuntimeError, match="partition"):
        transport.reconfigure(group, members, ("training", "", 0), params, plan)
    assert destroyed == []


def test_failure_to_destroy_old_fifo_also_poison_publication(prepared):
    transport, params, old_plan, _, members, group = prepared

    def fail(handle):
        raise RuntimeError("destroy failed")

    transport._extension.destroy = fail
    with pytest.raises(RuntimeError, match="destroy failed"):
        transport.reconfigure(
            group, members, ("training", "", 0), params, make_plan(2, members)
        )
    with pytest.raises(RuntimeError, match="rebuild failed"):
        transport.send(params, old_plan, 1)


@pytest.mark.parametrize("field", [1, 2, 3])
def test_new_communicator_rejects_mismatched_epoch_or_geometry(
    prepared, monkeypatch, field
):
    transport, params, _, _, members, group = prepared
    transport.reconfigure(
        group, members, ("training", "", 0), params, make_plan(2, members)
    )
    monkeypatch.setattr(module, "_load_extension", lambda: SimpleNamespace())
    monkeypatch.setattr(module.device_util, "get_torch_device", lambda: "cpu")
    monkeypatch.setattr(
        module.dist, "all_reduce", lambda tensor, **kwargs: tensor.mul_(3)
    )

    def gather(outputs, identity, **kwargs):
        for output in outputs:
            output.copy_(identity)
        outputs[1][field] += 1

    monkeypatch.setattr(module.dist, "all_gather", gather)
    with pytest.raises(RuntimeError, match="Membership epoch and inference geometry"):
        transport._ensure_initialized()


def test_device_prepare_builds_cache_without_launching_or_changing_weights(prepared):
    transport, params, plan, _, _, _ = prepared
    source = params["weight"].clone()
    calls = []

    def prepare(*args):
        calls.append(args)
        return {
            "device_plan_ready": True,
            "kernel_launched": False,
            "plan_cache_hit": False,
        }

    transport._extension.prepare = prepare
    transport._extension.launch = lambda *_: pytest.fail(
        "Preparation launched a weight kernel"
    )
    assert not transport.is_device_plan_ready(params, plan, True)
    metrics = transport.initialize_prepared_plan(params, plan, True)
    assert metrics["device_plan_ready"] and not metrics["kernel_launched"]
    assert metrics["transport_init_time_ms"] == 0.0
    assert calls[0][0] == 42 and calls[0][-1] is True
    assert calls[0][2] == transport._prepared_send[2].lengths
    assert transport.is_device_plan_ready(params, plan, True)
    assert not transport.is_device_plan_ready(
        params, make_plan(1, RolloutMembership(0, 1, 1, ("old",))), True
    )
    assert torch.equal(params["weight"], source)
    transport.close()
    assert not transport.is_device_plan_ready(params, plan, True)


@pytest.mark.parametrize(
    "result",
    [
        {"device_plan_ready": False, "kernel_launched": False},
        {"device_plan_ready": True, "kernel_launched": True},
        {"device_plan_ready": True},
    ],
)
def test_prepare_failure_is_not_acknowledged_as_ready(prepared, result):
    transport, params, plan, destroyed, _, _ = prepared
    transport._extension.prepare = lambda *_: result
    with pytest.raises(RuntimeError, match="without launching"):
        transport.initialize_prepared_plan(params, plan, True)
    assert destroyed == [42]
    assert not transport.is_device_plan_ready(params, plan, True)
    with pytest.raises(RuntimeError, match="rebuild failed"):
        transport.send(params, plan, 1)


def test_prepare_does_not_initialize_wrong_host_plan(prepared):
    transport, params, _, destroyed, _, _ = prepared
    wrong_plan = make_plan(1, RolloutMembership(0, 1, 1, ("old",)))
    with pytest.raises(RuntimeError, match="exact host plan"):
        transport.initialize_prepared_plan(params, wrong_plan, True)
    assert destroyed == []

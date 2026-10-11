import pytest
from shardstream.membership import RolloutJoinCoordinator, RolloutMembership


def publish(coordinator, version):
    membership = coordinator.begin_publication(version)
    for participant in membership.participants:
        coordinator.acknowledge_updated(participant, membership.epoch, version)
    coordinator.finish_publication()


def test_queued_join_during_update_is_applied_only_at_boundary():
    membership = RolloutMembership(0, 2, 2, ("old",))
    coordinator = RolloutJoinCoordinator(membership)
    coordinator.begin_publication(0)
    assert coordinator.request_join("new", 2)
    with pytest.raises(RuntimeError, match="boundary"):
        coordinator.prepare_join()
    for participant in membership.participants:
        coordinator.acknowledge_updated(participant, 0, 0)
    coordinator.finish_publication()
    assert coordinator.serving_engine_ids == ("old",)
    next_membership = coordinator.prepare_join()
    assert next_membership.rank(("training", "", 0)) == 4
    assert membership.rank(("training", "", 0)) == 2
    assert next_membership.rank(("rollout", "old", 1)) == 1
    assert next_membership.rank(("rollout", "new", 0)) == 2
    assert coordinator.serving_engine_ids == ("old",)
    with pytest.raises(RuntimeError, match="Every old and new"):
        coordinator.commit_join()
    for participant in next_membership.participants:
        coordinator.acknowledge_prepared(participant, 1)
    coordinator.commit_join()
    assert coordinator.serving_engine_ids == ("old",)
    coordinator.begin_publication(1)
    for participant in next_membership.participants[:-1]:
        coordinator.acknowledge_updated(participant, 1, 1)
    with pytest.raises(RuntimeError, match="full snapshot"):
        coordinator.finish_publication()
    assert coordinator.serving_engine_ids == ()
    coordinator.acknowledge_updated(next_membership.participants[-1], 1, 1)
    coordinator.finish_publication()
    assert coordinator.serving_engine_ids == ("old", "new")
    publish(coordinator, 2)
    assert coordinator.last_version == 2


def test_repeated_joins_preserve_ids_and_require_new_snapshot():
    coordinator = RolloutJoinCoordinator(RolloutMembership(0, 1, 1, ("a",)))
    publish(coordinator, 3)
    for epoch, engine in enumerate(("b", "c"), 1):
        assert coordinator.request_join(engine, 1)
        assert not coordinator.request_join(engine, 1)
        pending = coordinator.prepare_join()
        assert pending.epoch == epoch
        with pytest.raises(ValueError, match="epoch"):
            coordinator.acknowledge_prepared(pending.participants[0], epoch - 1)
        with pytest.raises(ValueError, match="Unknown"):
            coordinator.acknowledge_prepared(("rollout", "missing", 0), epoch)
        for participant in pending.participants:
            coordinator.acknowledge_prepared(participant, epoch)
        coordinator.commit_join()
        with pytest.raises(RuntimeError, match="boundary"):
            coordinator.prepare_join()
        publish(coordinator, 3 + epoch)
    assert coordinator.serving_engine_ids == ("a", "b", "c")


def test_failures_never_make_a_new_engine_available():
    coordinator = RolloutJoinCoordinator(RolloutMembership(0, 1, 1, ("a",)))
    publish(coordinator, 0)
    coordinator.request_join("b", 1)
    coordinator.prepare_join()
    coordinator.fail("new worker timed out")
    assert coordinator.serving_engine_ids == ()
    for fn in (coordinator.commit_join, lambda: coordinator.begin_publication(1)):
        with pytest.raises(RuntimeError, match="timed out"):
            fn()


def test_duplicate_unknown_and_stale_update_acks_do_not_complete_publication():
    coordinator = RolloutJoinCoordinator(RolloutMembership(0, 1, 1, ("a",)))
    members = coordinator.begin_publication(0)
    participant = members.participants[0]
    coordinator.acknowledge_updated(participant, 0, 0)
    coordinator.acknowledge_updated(participant, 0, 0)
    for epoch, version in ((1, 0), (0, 1)):
        with pytest.raises(ValueError, match="Stale"):
            coordinator.acknowledge_updated(participant, epoch, version)
    with pytest.raises(ValueError, match="Unknown"):
        coordinator.acknowledge_updated(("training", "", 99), 0, 0)
    with pytest.raises(RuntimeError, match="All ranks"):
        coordinator.finish_publication()
    coordinator.acknowledge_updated(members.participants[-1], 0, 0)
    coordinator.finish_publication()
    with pytest.raises(ValueError, match="monotonically"):
        coordinator.begin_publication(0)
    with pytest.raises(ValueError, match="same inference TP"):
        coordinator.request_join("b", 2)


@pytest.mark.parametrize(
    "args",
    [
        (-1, 1, 1, ("a",)),
        (0, 0, 1, ("a",)),
        (0, 1, 0, ("a",)),
        (0, 1, 1, ()),
        (0, 1, 1, ("a", "a")),
        (0, 256, 1, ("a",)),
    ],
)
def test_invalid_memberships_are_rejected(args):
    with pytest.raises(ValueError):
        RolloutMembership(*args)


def test_membership_copies_mutable_engine_ids_and_rejects_over_capacity_join():
    engines = ["a"]
    membership = RolloutMembership(0, 254, 1, engines)
    engines.append("mutated")
    assert membership.engine_ids == ("a",)
    coordinator = RolloutJoinCoordinator(membership)
    assert coordinator.request_join("b", 1)
    with pytest.raises(ValueError, match="256"):
        coordinator.request_join("c", 1)


def test_leave_waits_for_publication_and_all_retirement_acks_before_commit():
    coordinator = RolloutJoinCoordinator(RolloutMembership(0, 2, 2, ("a", "b", "c")))
    coordinator.begin_publication(0)
    assert coordinator.request_leave("b")
    assert not coordinator.request_leave("b")
    with pytest.raises(RuntimeError, match="boundary"):
        coordinator.prepare_change()
    for participant in coordinator.membership.participants:
        coordinator.acknowledge_updated(participant, 0, 0)
    coordinator.finish_publication()
    pending = coordinator.prepare_change()
    assert pending.engine_ids == ("a", "c")
    assert pending.rank(("rollout", "c", 0)) == 2
    assert pending.rank(("training", "", 0)) == 4
    assert coordinator.serving_engine_ids == ("a", "c")
    for participant in pending.participants:
        coordinator.acknowledge_prepared(participant, 1)
    with pytest.raises(RuntimeError, match="leaving rank"):
        coordinator.commit_change()
    with pytest.raises(ValueError, match="epoch"):
        coordinator.acknowledge_retired(("rollout", "b", 0), 0)
    with pytest.raises(ValueError, match="Unknown"):
        coordinator.acknowledge_retired(("rollout", "a", 0), 1)
    coordinator.acknowledge_retired(("rollout", "b", 0), 1)
    coordinator.acknowledge_retired(("rollout", "b", 0), 1)
    with pytest.raises(RuntimeError, match="leaving rank"):
        coordinator.commit_change()
    coordinator.acknowledge_retired(("rollout", "b", 1), 1)
    coordinator.commit_change()
    with pytest.raises(ValueError, match="epoch"):
        coordinator.acknowledge_retired(("rollout", "b", 1), 1)
    publish(coordinator, 1)
    assert coordinator.request_join("b", 2)
    rejoined = coordinator.prepare_change()
    for participant in rejoined.participants:
        coordinator.acknowledge_prepared(participant, 2)
    coordinator.commit_change()
    assert coordinator.serving_engine_ids == ("a", "c")
    publish(coordinator, 2)
    assert coordinator.serving_engine_ids == ("a", "c", "b")


def test_removing_last_engine_and_unknown_engine_preserves_current_membership():
    coordinator = RolloutJoinCoordinator(RolloutMembership(0, 1, 1, ("a", "b")))
    publish(coordinator, 0)
    with pytest.raises(ValueError, match="Unknown"):
        coordinator.request_leave("missing")
    coordinator.request_leave("b")
    with pytest.raises(ValueError, match="remain"):
        coordinator.request_leave("a")
    assert coordinator.membership.engine_ids == ("a", "b")
    assert coordinator.prepare_change().engine_ids == ("a",)

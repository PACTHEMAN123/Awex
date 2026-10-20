from types import SimpleNamespace

import pytest

from shardstream.integrations.publication import weightbridge_native as wb


def test_factory_state_is_lazy_and_recallable():
    calls = []
    def factory():
        calls.append(1)
        return object()
    source = wb.ProbeSource(SimpleNamespace(config="original"), {"weight": factory})
    assert list(source.state) == ["weight"] and len(source.state) == 1
    assert source.state.source.get_all_keys() == ("weight",)
    assert "weight" in source.state and "missing" not in source.state
    assert not calls
    assert source.state["weight"] is not source.state["weight"]
    assert len(calls) == 2
    assert source.config == "original"
    with pytest.raises(KeyError):
        source.state["missing"]


def test_publisher_waits_for_consumption_not_just_pack_event(monkeypatch):
    events = []
    harness = SimpleNamespace(is_driver=False, _training_barrier=lambda: events.append("barrier"))
    publisher = wb.WeightBridgePublication(harness)
    publisher.adapter = SimpleNamespace(
        connect=lambda: events.append("connect"),
        send_weights=lambda: SimpleNamespace(synchronize=lambda: events.append("pack_complete")),
        wait_send_complete=lambda: events.append("consumed"),
        flush_profile_outputs=lambda: events.append("flush"),
    )
    publisher.publish()
    publisher.publish()
    assert events == ["barrier", "connect", "pack_complete", "consumed", "flush",
                      "barrier", "pack_complete", "consumed", "flush"]


def test_receiver_polls_until_actual_consumption(monkeypatch):
    events = []
    poll = iter((False, False, True))
    receiver = wb.WeightBridgeReceiver({"timeout_seconds": 10}, 2)
    receiver.adapter = SimpleNamespace(poll_requests=lambda: next(poll),
                                      flush_profile_outputs=lambda: events.append("flush"))
    monkeypatch.setattr(wb.torch.cuda, "synchronize", lambda: events.append("sync"))
    assert receiver.update(None, 5) == {"worker_rank": 2, "step_id": 5, "consumed": True}
    assert events == ["sync", "flush"]


def test_receiver_timeout_is_not_reported_as_consumed():
    receiver = wb.WeightBridgeReceiver({"timeout_seconds": 0}, 0)
    with pytest.raises(TimeoutError):
        receiver.update(None, 0)

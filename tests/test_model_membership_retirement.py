from types import SimpleNamespace

import pytest
import torch
from shardstream.integrations import membership


def worker():
    model = torch.nn.Linear(2, 2)
    events = []
    transport = SimpleNamespace(
        membership_epoch=2,
        num_infer_engines=4,
        close=lambda: events.append("close"),
    )
    return SimpleNamespace(
        device_transport=transport,
        weights_update_group=object(),
        engine_rank=3,
        model=model,
        transfer_plan=object(),
        already_initialized=True,
    ), events


def test_retirement_drains_gpu_and_releases_group_without_changing_parameters(
    monkeypatch,
):
    w, events = worker()
    pointers = [p.data_ptr() for p in w.model.parameters()]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("drain"))
    monkeypatch.setattr(
        membership.dist, "destroy_process_group", lambda _: events.append("group")
    )
    result = membership.retire_model_membership(w, {"epoch": 3, "num_engines": 2})
    assert events == ["drain", "close", "group"]
    assert result["retired"] and result["epoch"] == 3
    assert w.device_transport is None and w.weights_update_group is None
    assert w.transfer_plan is None and not w.already_initialized
    assert [p.data_ptr() for p in w.model.parameters()] == pointers


@pytest.mark.parametrize(
    "epoch,count,rank", [(2, 2, 3), (4, 2, 3), (3, 4, 3), (3, 0, 3), (3, 2, 1)]
)
def test_invalid_or_surviving_retirement_does_not_close_transport(epoch, count, rank):
    w, events = worker()
    w.engine_rank = rank
    with pytest.raises(ValueError, match="Retirement"):
        membership.retire_model_membership(w, {"epoch": epoch, "num_engines": count})
    assert not events and w.device_transport is not None

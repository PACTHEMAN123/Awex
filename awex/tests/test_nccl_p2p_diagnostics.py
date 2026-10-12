"""Wire-contract diagnostics catch mismatches without changing P2P scheduling."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from awex.transfer import nccl_comm


def op(function, peer, count, dtype=torch.bfloat16):
    return SimpleNamespace(op=function, peer=peer, tensor=torch.empty(count, dtype=dtype))


def test_sender_and_receiver_contracts_match_across_rank_and_view_shape():
    sends = [op(dist.isend, 2, 6), op(dist.isend, 2, 3, torch.float32)]
    sends[0].tensor = sends[0].tensor.reshape(2, 3)
    recvs = [op(dist.irecv, 7, 6), op(dist.irecv, 7, 3, torch.float32)]
    source = nccl_comm._p2p_wire_contracts(sends)["2"]["send"]
    destination = nccl_comm._p2p_wire_contracts(recvs)["7"]["recv"]
    assert source == destination
    assert source["count"] == 2
    assert source["bytes"] == 24


def test_contract_detects_wire_order_even_when_count_and_bytes_match():
    sends = [op(dist.isend, 2, 6), op(dist.isend, 2, 3, torch.float32)]
    recvs = [op(dist.irecv, 7, 3, torch.float32), op(dist.irecv, 7, 6)]
    source = nccl_comm._p2p_wire_contracts(sends)["2"]["send"]
    destination = nccl_comm._p2p_wire_contracts(recvs)["7"]["recv"]
    assert (source["count"], source["bytes"]) == (destination["count"], destination["bytes"])
    assert source["sha256"] != destination["sha256"]


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_diagnostics_preserve_single_group_and_cancel_timer_on_failure(monkeypatch, diagnostics, fail):
    events = []
    operations = [op(dist.isend, 2, 6), op(dist.isend, 1, 3)]
    if diagnostics:
        monkeypatch.setenv("AWEX_P2P_DIAGNOSTICS", "1")
    else:
        monkeypatch.delenv("AWEX_P2P_DIAGNOSTICS", raising=False)
        monkeypatch.setattr(nccl_comm, "_p2p_wire_contracts", lambda ops: pytest.fail("default path computed diagnostic metadata"))
    monkeypatch.setattr(nccl_comm.faulthandler, "dump_traceback_later", lambda *args, **kwargs: events.append("timer"))
    monkeypatch.setattr(nccl_comm.faulthandler, "cancel_dump_traceback_later", lambda: events.append("cancel"))
    monkeypatch.setattr(nccl_comm.device_util, "synchronize", lambda: events.append("sync"))

    def submit(ops):
        events.append("submit")
        assert ops == [operations[1], operations[0]]
        if fail:
            raise RuntimeError("submit failed")
        return [SimpleNamespace(wait=lambda: events.append("wait"))]

    monkeypatch.setattr(dist, "batch_isend_irecv", submit)
    if fail:
        with pytest.raises(RuntimeError, match="submit failed"):
            nccl_comm.batch_send_recv(operations, [], blocking=True)
    else:
        assert nccl_comm.batch_send_recv(operations, [], blocking=True) == []
    assert events == (["timer"] if diagnostics else []) + ["submit"] + ([] if fail else ["wait", "sync"]) + (["cancel"] if diagnostics else [])

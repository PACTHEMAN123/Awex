from types import SimpleNamespace

import pytest

from awex.transfer import nccl_comm


def _ops(peer, count):
    return [SimpleNamespace(peer=peer, ordinal=index) for index in range(count)]


def test_chunk_p2p_ops_preserves_peer_order_and_ordinal_rounds():
    batches = nccl_comm._chunk_p2p_ops_by_peer(
        _ops(3, 5) + _ops(1, 3), max_ops_per_peer=2
    )

    assert [[op.peer for op in batch] for batch in batches] == [
        [1, 3, 1, 3],
        [1, 3, 3],
        [3],
    ]
    for peer, expected in ((1, [0, 1, 2]), (3, [0, 1, 2, 3, 4])):
        actual = [op.ordinal for batch in batches for op in batch if op.peer == peer]
        assert actual == expected


def test_batch_send_recv_bounds_group_size_by_peer(monkeypatch):
    submitted = []
    synchronized = []

    class Work:
        def wait(self):
            return None

    def fake_batch_isend_irecv(ops):
        submitted.append(list(ops))
        return [Work() for _ in ops]

    monkeypatch.setenv("AWEX_NCCL_MAX_OPS_PER_PEER_BATCH", "2")
    monkeypatch.setattr(
        nccl_comm.dist, "batch_isend_irecv", fake_batch_isend_irecv
    )
    monkeypatch.setattr(
        nccl_comm.device_util, "synchronize", lambda: synchronized.append(True)
    )

    result = nccl_comm.batch_send_recv(
        send_ops=_ops(4, 5) + _ops(2, 3),
        recv_ops=[],
        blocking=True,
        use_group=True,
    )

    assert result == []
    assert [len(batch) for batch in submitted] == [4, 3, 1]
    assert len(synchronized) == 3


def test_chunk_p2p_ops_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="must be positive"):
        nccl_comm._chunk_p2p_ops_by_peer(_ops(0, 1), max_ops_per_peer=0)

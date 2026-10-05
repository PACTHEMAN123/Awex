from types import SimpleNamespace

import pytest

from awex.transfer import nccl_comm


def _ops(peer, count):
    return [
        SimpleNamespace(peer=peer, ordinal=index, group="process-group")
        for index in range(count)
    ]


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


def test_bipartite_peer_stages_are_inverse_matchings():
    world_size = 8
    for stage in range(world_size // 2):
        for writer_rank in range(world_size // 2, world_size):
            reader_rank = nccl_comm._bipartite_peer_for_stage(
                writer_rank, world_size, stage
            )
            assert (
                nccl_comm._bipartite_peer_for_stage(
                    reader_rank, world_size, stage
                )
                == writer_rank
            )


def test_batch_send_recv_serializes_bipartite_peers_by_stage(monkeypatch):
    submitted = []
    barriers = []

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
    monkeypatch.setattr(nccl_comm.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(nccl_comm.dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(
        nccl_comm.dist,
        "barrier",
        lambda group, device_ids: barriers.append((group, device_ids)),
    )
    monkeypatch.setattr(nccl_comm.device_util, "current_device", lambda: 0)
    monkeypatch.setattr(nccl_comm.device_util, "synchronize", lambda: None)

    result = nccl_comm.batch_send_recv(
        send_ops=[],
        recv_ops=_ops(4, 3) + _ops(5, 2),
        blocking=True,
        use_group=True,
        use_peer_stages=True,
    )

    assert result == []
    assert [[op.peer for op in batch] for batch in submitted] == [
        [4, 4],
        [4],
        [5, 5],
    ]
    assert len(barriers) == 4

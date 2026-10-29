from unittest.mock import patch

import pytest
import torch

from shardstream.integrations.publication.weightbridge_quantized import (
    GatherBeforeQuantize, MatrixPartition,
)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("rank", [0, 1])
def test_partition_and_scale_correspondence(axis, rank):
    p = MatrixPartition((256, 512), axis, rank, 2)
    assert p.logical_gather_egress_bytes == 131072
    assert tuple(torch.empty(p.shape)[p.slices()].shape) == p.local_shape
    assert tuple(torch.empty(2, 4)[p.slices(scales=True)].shape) == tuple(n // 128 for n in p.local_shape)


@pytest.mark.parametrize("args", [((256, 512), 2, 0, 2), ((128, 512), 0, 0, 2),
                                  ((257, 512), 0, 0, 2), ((256, 512), 0, 2, 2)])
def test_unsupported_partitions_fail(args):
    with pytest.raises(ValueError):
        MatrixPartition(*args)


@pytest.mark.parametrize("axis", [0, 1])
def test_every_update_gathers_full_live_tensor_before_quantization(axis):
    full = torch.arange(256 * 256).reshape(256, 256).to(torch.bfloat16)
    parts = [MatrixPartition(tuple(full.shape), axis, r, 2) for r in range(2)]
    local = [full[p.slices()].contiguous() for p in parts]
    observed = []

    def gather(out, source, group):
        assert torch.equal(source, local[0].view(-1))
        out.copy_(torch.cat([x.view(-1) for x in local]))
        observed.append("gather")

    def quantize(x, blocks):
        assert blocks == (128, 128)
        assert torch.equal(x, torch.cat(local, dim=axis))
        observed.append("quantize_full")
        return (x.float().div(256).to(torch.float8_e4m3fn), torch.ones(2, 2))

    with patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.all_gather_object", side_effect=lambda out, v, group: out.__setitem__(slice(None), [v, v])), \
         patch("torch.distributed.all_gather_into_tensor", side_effect=gather):
        frontend = GatherBeforeQuantize({"w": local[0]}, {"w": parts[0]}, quantize)
        pointer = frontend.payload["w"].data_ptr()
        first = frontend.prepare()
        q1 = frontend.payload["w"].clone()
        for x in local:
            x.mul_(-1)
        second = frontend.prepare()
        assert observed == ["gather", "quantize_full"] * 2
        assert frontend.payload["w"].data_ptr() == pointer
        assert not torch.equal(q1, frontend.payload["w"])
        assert first["logical_gather_egress_bytes"] == second["logical_gather_egress_bytes"] == 65536


def test_publish_waits_for_receiver_consumption():
    from types import SimpleNamespace
    events = []
    frontend = object.__new__(GatherBeforeQuantize)
    frontend._sync = lambda: None
    frontend.prepare = lambda: (events.append("gather_then_quantize") or {})
    adapter = SimpleNamespace(
        send_weights=lambda: (events.append("send") or SimpleNamespace(synchronize=lambda: events.append("pack_done"))),
        wait_send_complete=lambda: events.append("consumed"),
    )
    stats = frontend.publish(adapter)
    assert events == ["gather_then_quantize", "send", "pack_done", "consumed"]
    assert stats["complete_ms"] >= stats["native_transfer_ms"]

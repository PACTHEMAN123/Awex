from types import SimpleNamespace

import pytest
import torch

from awex.transfer import nccl_device_v2
from awex.transfer.nccl_device_v2 import _build_recv_batch, _build_send_batch
from awex.transfer.tensor_layout import StaticTensorLayout
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan


def operation(root, reader, shape, spans=()):
    return CommunicationOperation(
        root,
        SimpleNamespace(name="weight", shape=shape, dtype="bfloat16"),
        (0, 0),
        reader,
        SimpleNamespace(name="weight", shape=shape, dtype="float8_e4m3fn"),
        (0, 0),
        shape,
        (slice(None), slice(None)),
        (slice(None), slice(None)),
        send_tensor_span_numels=spans,
    )


def test_fp8_ring_preserves_block_records_and_scale_views(monkeypatch):
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    shape = (512, 256)
    weight = torch.empty(shape, dtype=torch.bfloat16)
    spans = (256 * 256, 256 * 256)
    source = StaticTensorLayout(shape, (weight[:256], weight[256:]))
    send = _build_send_batch(
        {"weight": source},
        TransferPlan(operations={i: [operation(3, i, shape, spans)] for i in range(3)}),
        3,
        4,
        0,
        allow_staging=False,
        infer_instance_world_size=1,
        num_infer_engines=3,
        ring_broadcast=True,
        ring_swizzle=True,
        fp8_block_shape=(128, 128),
    )
    target = torch.empty((512, 384), dtype=torch.float8_e4m3fn)[:, :256]
    scales = torch.empty((4, 4), dtype=torch.float32)[:, :2]
    recv = _build_recv_batch(
        {"weight": target, "weight_scale_inv": scales},
        TransferPlan(operations={3: [operation(3, 0, shape, spans)]}),
        0,
        4,
        0,
        allow_staging=False,
        infer_instance_world_size=1,
        num_infer_engines=3,
        ring_broadcast=True,
        ring_swizzle=True,
        fp8_block_shape=(128, 128),
    )
    assert send.lengths == recv.lengths == [4 * (128 * 128 + 16)] * 2
    assert send.tensor_row_bytes == [512, 512]
    assert send.tensor_row_strides == [512, 512]
    assert [tuple(t.shape) for t in send.tensors] == [(256, 256)] * 2
    assert send.expected_counts == [2, 0, 0, 0]
    assert recv.quantization[1][2] == scales[2:].data_ptr()
    assert recv.quantization[0][3] == 16
    assert not recv.copybacks
    assert recv.forward_peers == [1, 1]


def test_fp8_rejects_non_block_aligned_shard(monkeypatch):
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    shape = (128, 192)
    with pytest.raises(
        nccl_device_v2.NCCLDeviceV2UnavailableError, match="block-aligned"
    ):
        _build_send_batch(
            {"weight": torch.empty(shape, dtype=torch.bfloat16)},
            TransferPlan(operations={0: [operation(1, 0, shape)]}),
            1,
            2,
            0,
            fp8_block_shape=(128, 128),
        )

# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from types import SimpleNamespace

import pytest
import torch

from awex.transfer import nccl_device_v2
from awex.transfer.nccl_device_v2 import (
    NCCLDeviceV2UnavailableError,
    _build_recv_batch,
    _build_send_batch,
)
from awex.transfer.tensor_layout import (
    make_blockwise_fp8_layouts,
    make_blockwise_fp8_row_layouts,
)
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan


def _operation(source_dtype, destination_dtype, numel=32):
    send_shard = SimpleNamespace(name="weight", shape=(numel,), dtype=source_dtype)
    recv_shard = SimpleNamespace(name="weight", shape=(numel,), dtype=destination_dtype)
    return CommunicationOperation(
        send_rank=1,
        send_shard_meta=send_shard,
        send_offset=(0,),
        recv_rank=0,
        recv_shard_meta=recv_shard,
        recv_offset=(0,),
        overlap_shape=(numel,),
        train_slices=(slice(None),),
        inf_slices=(slice(None),),
    )


def _matrix_operation(
    name,
    source_dtype,
    destination_dtype,
    shape,
    train_slices=None,
):
    train_slices = train_slices or tuple(slice(None) for _ in shape)
    send_shard = SimpleNamespace(name=name, shape=shape, dtype=source_dtype)
    recv_shard = SimpleNamespace(name=name, shape=shape, dtype=destination_dtype)
    return CommunicationOperation(
        send_rank=1,
        send_shard_meta=send_shard,
        send_offset=tuple(0 for _ in shape),
        recv_rank=0,
        recv_shard_meta=recv_shard,
        recv_offset=tuple(0 for _ in shape),
        overlap_shape=shape,
        train_slices=train_slices,
        inf_slices=tuple(slice(None) for _ in shape),
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="PyTorch has no FP8 dtype"
)
def test_streaming_cast_uses_destination_dtype_as_wire_format(monkeypatch):
    operation = _operation(torch.bfloat16, torch.float8_e4m3fn)
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    send_batch = _build_send_batch(
        {"weight": torch.arange(32, dtype=torch.bfloat16)},
        TransferPlan(operations={0: [operation]}),
        rank=1,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )
    recv_batch = _build_recv_batch(
        {"weight": torch.empty(32, dtype=torch.float8_e4m3fn)},
        TransferPlan(operations={1: [operation]}),
        rank=0,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )

    assert send_batch.lengths == recv_batch.lengths == [32]
    assert send_batch.tensor_lengths == [64]
    assert recv_batch.tensor_lengths == [32]
    assert send_batch.wire_dtypes == recv_batch.wire_dtypes == [4]
    assert send_batch.wire_element_bytes == [1]
    assert send_batch.expected_counts == [1, 0]
    assert recv_batch.expected_counts == [0, 1]


def test_streaming_cast_rejects_unsupported_numeric_conversion(monkeypatch):
    operation = _operation(torch.int32, torch.float32)
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    with pytest.raises(
        NCCLDeviceV2UnavailableError, match="streaming cast does not support"
    ):
        _build_send_batch(
            {"weight": torch.arange(32, dtype=torch.int32)},
            TransferPlan(operations={0: [operation]}),
            rank=1,
            world_size=2,
            chunk_bytes=16,
            allow_staging=False,
        )


def test_receive_tensor_must_match_wire_dtype(monkeypatch):
    operation = _operation(torch.bfloat16, torch.float16)
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    with pytest.raises(
        NCCLDeviceV2UnavailableError, match="does not match the wire format"
    ):
        _build_recv_batch(
            {"weight": torch.empty(32, dtype=torch.bfloat16)},
            TransferPlan(operations={1: [operation]}),
            rank=0,
            world_size=2,
            chunk_bytes=16,
            allow_staging=False,
        )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="PyTorch has no FP8 dtype"
)
def test_blockwise_fp8_layout_matches_eager_quantization():
    from awex.converter.weights_converter import per_block_cast_to_fp8

    source = (torch.arange(129 * 257, dtype=torch.float32).reshape(129, 257) % 2048).to(
        torch.bfloat16
    )
    weight, scale = make_blockwise_fp8_layouts(source)

    actual_weight = weight.materialize()
    actual_scale = scale.materialize()
    expected_weight, expected_scale = per_block_cast_to_fp8(source, False)

    assert torch.equal(actual_weight.float(), expected_weight.float())
    assert torch.equal(actual_scale, expected_scale)
    assert tuple(actual_scale.shape) == (2, 3)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="PyTorch has no FP8 dtype"
)
def test_blockwise_fp8_row_layouts_share_state_and_translate_offsets(monkeypatch):
    from awex.converter.weights_converter import per_block_cast_to_fp8

    source = (
        torch.arange(256 * 256, dtype=torch.float32)
        .reshape(256, 256)
        .to(torch.bfloat16)
    )
    (first_weight, first_scale), (second_weight, second_scale) = (
        make_blockwise_fp8_row_layouts(source, ((0, 128), (128, 256)))
    )
    expected_weight, expected_scale = per_block_cast_to_fp8(source, False)

    assert first_weight.state is second_weight.state
    assert first_scale.state is second_scale.state
    assert torch.equal(first_weight.materialize().float(), expected_weight[:128].float())
    assert torch.equal(second_weight.materialize().float(), expected_weight[128:].float())
    assert torch.equal(first_scale.materialize(), expected_scale[:1])
    assert torch.equal(second_scale.materialize(), expected_scale[1:])

    operations = [
        _matrix_operation(
            "weight",
            torch.bfloat16,
            torch.float8_e4m3fn,
            tuple(second_weight.shape),
        ),
        _matrix_operation(
            "weight_scale_inv",
            torch.float32,
            torch.float32,
            tuple(second_scale.shape),
        ),
    ]
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)
    batch = _build_send_batch(
        {"weight": second_weight, "weight_scale_inv": second_scale},
        TransferPlan(operations={0: operations}),
        rank=1,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )

    assert batch.tensors[0].data_ptr() == source[128:].data_ptr()
    assert batch.quant_scale_tensors[0] is second_weight.state.scale
    assert batch.quant_row_offsets == [128, 0]
    assert batch.quant_cols == [256, 0]
    assert batch.quant_group_ids == [0, 0]


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="PyTorch has no FP8 dtype"
)
def test_blockwise_fp8_send_batch_uses_source_and_shared_scale(monkeypatch):
    source = (
        torch.arange(256 * 384, dtype=torch.float32)
        .reshape(256, 384)
        .to(torch.bfloat16)
    )
    weight, scale = make_blockwise_fp8_layouts(source)
    operations = [
        _matrix_operation(
            "weight",
            torch.bfloat16,
            torch.float8_e4m3fn,
            tuple(source.shape),
        ),
        _matrix_operation(
            "weight_scale_inv",
            torch.float32,
            torch.float32,
            tuple(scale.shape),
        ),
    ]
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    batch = _build_send_batch(
        {"weight": weight, "weight_scale_inv": scale},
        TransferPlan(operations={0: operations}),
        rank=1,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )

    assert weight.state._materialized is None
    assert batch.tensors[0].data_ptr() == source.data_ptr()
    assert batch.quant_scale_tensors[0] is weight.state.scale
    assert batch.lengths == [source.numel(), scale.numel() * 4]
    assert batch.tensor_lengths == [source.numel() * 2, scale.numel() * 4]
    assert batch.wire_dtypes == [4, 3]
    assert batch.quant_modes == [1, 0]
    assert batch.quant_rows == [256, 0]
    assert batch.quant_cols == [384, 0]
    assert batch.quant_row_offsets == [0, 0]
    assert batch.quant_col_offsets == [0, 0]
    assert batch.quant_scale_row_strides == [3, 0]
    assert batch.quant_block_rows == [128, 0]
    assert batch.quant_block_cols == [128, 0]
    assert batch.quant_group_ids == [0, 0]
    assert batch.tensor_row_bytes == [384 * source.element_size(), scale.numel() * 4]
    assert batch.tensor_row_strides == [
        384 * source.element_size(),
        scale.numel() * 4,
    ]


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="PyTorch has no FP8 dtype"
)
def test_blockwise_fp8_group_survives_interleaved_plan_tasks(monkeypatch):
    source = torch.empty((128, 256), dtype=torch.bfloat16)
    weight, scale = make_blockwise_fp8_layouts(source)
    operations = [
        _matrix_operation(
            "weight", torch.bfloat16, torch.float8_e4m3fn, tuple(source.shape)
        ),
        _matrix_operation("bias", torch.float32, torch.float32, (16,)),
        _matrix_operation(
            "weight_scale_inv", torch.float32, torch.float32, tuple(scale.shape)
        ),
    ]
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    send_batch = _build_send_batch(
        {"weight": weight, "bias": torch.empty(16), "weight_scale_inv": scale},
        TransferPlan(operations={0: operations}),
        rank=1,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )
    recv_batch = _build_recv_batch(
        {
            "weight": torch.empty(source.shape, dtype=torch.float8_e4m3fn),
            "bias": torch.empty(16),
            "weight_scale_inv": torch.empty(scale.shape),
        },
        TransferPlan(operations={1: operations}),
        rank=0,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )

    assert send_batch.quant_group_ids == [0, -1, 0]
    assert recv_batch.quant_group_ids == [0, -1, 0]


def test_blockwise_fp8_rejects_unaligned_transfer_slice():
    source = torch.empty((256, 256), dtype=torch.bfloat16)
    weight, _ = make_blockwise_fp8_layouts(source)

    with pytest.raises(ValueError, match="must align to 128-element boundaries"):
        weight.state.source_fragments((slice(1, 129), slice(None)))

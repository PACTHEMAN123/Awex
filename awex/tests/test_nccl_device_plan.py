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

from awex.transfer import nccl_device
from awex.transfer.nccl_device import (
    NCCLDeviceUnavailableError,
    _build_recv_batch,
    _build_send_batch,
    _resolve_chunk_bytes,
    _split_contiguous_tensor,
)
from awex.transfer.tensor_layout import StaticTensorLayout, slice_layout_fragments
from awex.transfer.transfer_plan import (
    CommunicationOperation,
    TransferChunk,
    TransferPlan,
    build_transfer_chunks,
)


def test_build_transfer_chunks_uses_fixed_ranges_and_short_tail():
    assert build_transfer_chunks(10, 4) == [
        TransferChunk(0, 4),
        TransferChunk(4, 4),
        TransferChunk(8, 2),
    ]


def test_build_transfer_chunks_can_preserve_one_task_per_tensor():
    assert build_transfer_chunks(10, 0) == [TransferChunk(0, 10)]
    assert build_transfer_chunks(0, 4) == []


def test_split_contiguous_tensor_preserves_storage_and_byte_ranges():
    tensor = torch.arange(10, dtype=torch.int32)

    chunks = _split_contiguous_tensor(tensor, 16)

    assert [(offset, length) for _, offset, length in chunks] == [
        (0, 16),
        (16, 16),
        (32, 8),
    ]
    assert torch.equal(torch.cat([chunk for chunk, _, _ in chunks]), tensor)
    assert chunks[1][0].data_ptr() == tensor.data_ptr() + 16


def test_chunk_size_must_be_vector_aligned():
    with pytest.raises(NCCLDeviceUnavailableError, match="multiple of 16"):
        _resolve_chunk_bytes(15)


def test_chunk_size_rejects_negative_values():
    with pytest.raises(NCCLDeviceUnavailableError, match="non-negative"):
        _resolve_chunk_bytes(-16)


def test_send_plan_lowers_tensor_to_chunk_tasks(monkeypatch):
    tensor = torch.arange(10, dtype=torch.int32)
    shard = SimpleNamespace(name="weight")
    operation = CommunicationOperation(
        send_rank=1,
        send_shard_meta=shard,
        send_offset=(0,),
        recv_rank=0,
        recv_shard_meta=shard,
        recv_offset=(0,),
        overlap_shape=(10,),
        train_slices=(slice(None),),
        inf_slices=(slice(None),),
    )
    plan = TransferPlan(operations={0: [operation]})
    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)

    batch = _build_send_batch(
        {"weight": tensor}, plan, rank=1, world_size=2, chunk_bytes=16
    )

    assert batch.offsets == [0, 16, 32]
    assert batch.lengths == [16, 16, 8]
    assert batch.ordinals == [0, 1, 2]
    assert batch.expected_counts == [3, 0]
    assert batch.region_bytes == [0, 40]
    assert torch.equal(torch.cat(batch.tensors), tensor)


def test_send_plan_marks_identical_engine_replicas_for_multicast(monkeypatch):
    tensor = torch.arange(16, dtype=torch.int32)
    shard = SimpleNamespace(name="weight")

    def operation(recv_rank):
        return CommunicationOperation(
            send_rank=4,
            send_shard_meta=shard,
            send_offset=(0,),
            recv_rank=recv_rank,
            recv_shard_meta=shard,
            recv_offset=(0,),
            overlap_shape=(16,),
            train_slices=(slice(None),),
            inf_slices=(slice(None),),
        )

    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)
    batch = _build_send_batch(
        {"weight": tensor},
        TransferPlan(operations={0: [operation(0)], 2: [operation(2)]}),
        rank=4,
        world_size=5,
        chunk_bytes=16,
        allow_staging=False,
        infer_instance_world_size=2,
        num_infer_engines=2,
    )

    assert batch.expected_counts == [4, 0, 4, 0, 0]
    assert batch.multicast_groups == [[0, 2], [], [], [], []]


def test_send_plan_does_not_multicast_multiple_logical_streams(monkeypatch):
    tensor = torch.arange(16, dtype=torch.int32)
    shard = SimpleNamespace(name="weight")

    def operation(recv_rank):
        return CommunicationOperation(
            send_rank=4,
            send_shard_meta=shard,
            send_offset=(0,),
            recv_rank=recv_rank,
            recv_shard_meta=shard,
            recv_offset=(0,),
            overlap_shape=(16,),
            train_slices=(slice(None),),
            inf_slices=(slice(None),),
        )

    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)
    batch = _build_send_batch(
        {"weight": tensor},
        TransferPlan(
            operations={peer: [operation(peer)] for peer in range(4)}
        ),
        rank=4,
        world_size=5,
        chunk_bytes=16,
        allow_staging=False,
        infer_instance_world_size=2,
        num_infer_engines=2,
    )

    assert batch.multicast_groups == [[], [], [], [], []]


def test_send_plan_does_not_multicast_different_source_slices(monkeypatch):
    tensor = torch.arange(16, dtype=torch.int32)
    shard = SimpleNamespace(name="weight")
    first = CommunicationOperation(
        send_rank=4,
        send_shard_meta=shard,
        send_offset=(0,),
        recv_rank=0,
        recv_shard_meta=shard,
        recv_offset=(0,),
        overlap_shape=(8,),
        train_slices=(slice(0, 8),),
        inf_slices=(slice(0, 8),),
    )
    second = CommunicationOperation(
        send_rank=4,
        send_shard_meta=shard,
        send_offset=(8,),
        recv_rank=2,
        recv_shard_meta=shard,
        recv_offset=(0,),
        overlap_shape=(8,),
        train_slices=(slice(8, 16),),
        inf_slices=(slice(0, 8),),
    )

    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)
    batch = _build_send_batch(
        {"weight": tensor},
        TransferPlan(operations={0: [first], 2: [second]}),
        rank=4,
        world_size=5,
        chunk_bytes=16,
        allow_staging=False,
        infer_instance_world_size=2,
        num_infer_engines=2,
    )

    assert batch.multicast_groups == [[], [], [], [], []]


def test_static_layout_slices_logical_order_without_materializing():
    source = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    layout = StaticTensorLayout(
        shape=(4, 4),
        spans=(source.narrow(0, 0, 2), source.narrow(0, 4, 2)),
    )

    fragments = layout.slice((slice(1, 3), slice(None)))

    assert [fragment.data_ptr() for fragment in fragments] == [
        source[1].data_ptr(),
        source[4].data_ptr(),
    ]
    assert torch.equal(
        torch.cat(fragments).reshape(2, 4),
        torch.stack((source[1], source[4])),
    )


def test_static_layout_fragment_schedule_handles_partial_columns():
    fragments = slice_layout_fragments(
        shape=(4, 4),
        slices=(slice(1, 4), slice(1, 3)),
        span_numels=(8, 8),
    )

    assert fragments == [
        (0, 5, 2),
        (1, 1, 2),
        (1, 5, 2),
    ]


def test_send_plan_lowers_static_layout_to_matching_contiguous_tasks(monkeypatch):
    source = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    layout = StaticTensorLayout(
        shape=(4, 4),
        spans=(source.narrow(0, 0, 2), source.narrow(0, 4, 2)),
    )
    shard = SimpleNamespace(name="weight", shape=(4, 4))
    operation = CommunicationOperation(
        send_rank=1,
        send_shard_meta=shard,
        send_offset=(0, 0),
        recv_rank=0,
        recv_shard_meta=shard,
        recv_offset=(0, 0),
        overlap_shape=(4, 4),
        train_slices=(slice(None), slice(None)),
        inf_slices=(slice(None), slice(None)),
        send_tensor_span_numels=(8, 8),
    )
    plan = TransferPlan(operations={0: [operation]})
    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)

    batch = _build_send_batch(
        {"weight": layout}, plan, rank=1, world_size=2, chunk_bytes=16
    )

    assert batch.offsets == [0, 16, 32, 48]
    assert batch.lengths == [16, 16, 16, 16]
    assert batch.ordinals == [0, 1, 2, 3]
    assert batch.expected_counts == [4, 0]
    assert torch.equal(
        torch.cat(batch.tensors).reshape(4, 4),
        torch.cat((source[:2], source[4:])),
    )


def test_static_layout_sender_and_receiver_batches_are_symmetric(monkeypatch):
    source = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    layout = StaticTensorLayout(
        shape=(4, 4),
        spans=(source.narrow(0, 0, 2), source.narrow(0, 4, 2)),
    )
    destination = torch.empty((4, 4), dtype=torch.int32)
    shard = SimpleNamespace(name="weight", shape=(4, 4))
    operation = CommunicationOperation(
        send_rank=1,
        send_shard_meta=shard,
        send_offset=(0, 0),
        recv_rank=0,
        recv_shard_meta=shard,
        recv_offset=(0, 0),
        overlap_shape=(4, 4),
        train_slices=(slice(None), slice(None)),
        inf_slices=(slice(None), slice(None)),
        send_tensor_span_numels=(8, 8),
    )
    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)

    send_batch = _build_send_batch(
        {"weight": layout},
        TransferPlan(operations={0: [operation]}),
        rank=1,
        world_size=2,
        chunk_bytes=16,
    )
    recv_batch = _build_recv_batch(
        {"weight": destination},
        TransferPlan(operations={1: [operation]}),
        rank=0,
        world_size=2,
        chunk_bytes=16,
    )

    assert send_batch.offsets == recv_batch.offsets
    assert send_batch.lengths == recv_batch.lengths
    assert send_batch.ordinals == recv_batch.ordinals
    for source_chunk, destination_chunk in zip(send_batch.tensors, recv_batch.tensors):
        destination_chunk.copy_(source_chunk)
    assert torch.equal(destination, torch.cat((source[:2], source[4:])))


def test_strict_plan_lowers_static_partial_columns_without_staging(monkeypatch):
    source = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    layout = StaticTensorLayout(
        shape=(4, 4),
        spans=(source.narrow(0, 0, 2), source.narrow(0, 4, 2)),
    )
    destination = torch.empty((4, 2), dtype=torch.int32)
    send_shard = SimpleNamespace(name="weight", shape=(4, 4))
    recv_shard = SimpleNamespace(name="weight", shape=(4, 2))
    operation = CommunicationOperation(
        send_rank=1,
        send_shard_meta=send_shard,
        send_offset=(0, 1),
        recv_rank=0,
        recv_shard_meta=recv_shard,
        recv_offset=(0, 0),
        overlap_shape=(4, 2),
        train_slices=(slice(None), slice(1, 3)),
        inf_slices=(slice(None), slice(None)),
        send_tensor_span_numels=(8, 8),
    )
    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)

    send_batch = _build_send_batch(
        {"weight": layout},
        TransferPlan(operations={0: [operation]}),
        rank=1,
        world_size=2,
        allow_staging=False,
    )
    recv_batch = _build_recv_batch(
        {"weight": destination},
        TransferPlan(operations={1: [operation]}),
        rank=0,
        world_size=2,
        allow_staging=False,
    )

    assert send_batch.copybacks == []
    assert recv_batch.copybacks == []
    assert send_batch.lengths == recv_batch.lengths
    for source_chunk, destination_chunk in zip(send_batch.tensors, recv_batch.tensors):
        destination_chunk.copy_(source_chunk)
    expected = torch.cat((source[:2], source[4:]))[:, 1:3]
    assert torch.equal(destination, expected)


def test_strict_send_plan_lowers_strided_view_without_staging(monkeypatch):
    tensor = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    shard = SimpleNamespace(name="weight", shape=(4, 4))
    operation = CommunicationOperation(
        send_rank=1,
        send_shard_meta=shard,
        send_offset=(0, 1),
        recv_rank=0,
        recv_shard_meta=shard,
        recv_offset=(0, 0),
        overlap_shape=(4, 2),
        train_slices=(slice(None), slice(1, 3)),
        inf_slices=(slice(None), slice(0, 2)),
    )

    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)
    batch = _build_send_batch(
        {"weight": tensor},
        TransferPlan(operations={0: [operation]}),
        rank=1,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )

    assert batch.copybacks == []
    assert batch.lengths == [16, 16]
    assert batch.tensor_offsets == [0, 16]
    assert batch.tensor_row_bytes == [8, 8]
    assert batch.tensor_row_strides == [16, 16]
    assert all(not task.is_contiguous() for task in batch.tensors)


def test_strict_recv_plan_lowers_strided_view_without_staging(monkeypatch):
    tensor = torch.empty((4, 4), dtype=torch.int32)
    shard = SimpleNamespace(name="weight", shape=(4, 4))
    operation = CommunicationOperation(
        send_rank=1,
        send_shard_meta=shard,
        send_offset=(0, 0),
        recv_rank=0,
        recv_shard_meta=shard,
        recv_offset=(0, 1),
        overlap_shape=(4, 2),
        train_slices=(slice(None), slice(0, 2)),
        inf_slices=(slice(None), slice(1, 3)),
    )

    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)
    batch = _build_recv_batch(
        {"weight": tensor},
        TransferPlan(operations={1: [operation]}),
        rank=0,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )

    target = tensor[:, 1:3]
    assert batch.copybacks == []
    assert batch.lengths == [16, 16]
    assert batch.tensor_offsets == [0, 16]
    assert batch.tensor_row_bytes == [8, 8]
    assert batch.tensor_row_strides == [16, 16]
    assert all(task.data_ptr() == target.data_ptr() for task in batch.tensors)
    assert all(not task.is_contiguous() for task in batch.tensors)

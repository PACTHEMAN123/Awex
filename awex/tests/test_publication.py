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

import copy

import torch

from awex.publication.verl_nccl import (
    BucketMeta,
    McoreFullTensorExporter,
    TensorChunkAssembler,
    TensorChunkMeta,
    VerlNcclBroadcastSender,
    VerlNcclBroadcastReceiver,
    _flatten_result_dicts,
)
from awex.tests.megatron_parallel import resolve_megatron_parallelism
from awex.tests.weights_exchange_multi_vllm_it import MultiVLLMWeightsExchangeIT
from awex.tests.weights_exchange_vllm_it import (
    VLLMWeightsExchangeIT,
    vllm_inference_config,
)
from awex.util import device as device_util


def test_bucket_metadata_round_trip():
    metadata = BucketMeta(
        length=16,
        is_last=True,
        chunks=[
            TensorChunkMeta(
                name="model.weight",
                shape=(2, 2),
                dtype="float32",
                chunk_offset=0,
                chunk_size=16,
                bucket_offset=0,
            )
        ],
    )

    assert BucketMeta.from_bytes(metadata.to_bytes()) == metadata


def test_tensor_chunk_assembler_handles_split_and_unaligned_tensors():
    large = torch.arange(10, dtype=torch.float32)
    small = torch.arange(3, dtype=torch.bfloat16)
    large_bytes = large.view(torch.uint8)
    small_bytes = small.view(torch.uint8)

    first = torch.empty(17, dtype=torch.uint8)
    first.copy_(large_bytes[:17])
    first_meta = BucketMeta(
        length=17,
        is_last=False,
        chunks=[
            TensorChunkMeta(
                name="large",
                shape=tuple(large.shape),
                dtype="float32",
                chunk_offset=0,
                chunk_size=17,
                bucket_offset=0,
            )
        ],
    )

    remaining = large_bytes.numel() - 17
    second = torch.empty(remaining + small_bytes.numel(), dtype=torch.uint8)
    second[:remaining].copy_(large_bytes[17:])
    second[remaining:].copy_(small_bytes)
    second_meta = BucketMeta(
        length=second.numel(),
        is_last=True,
        chunks=[
            TensorChunkMeta(
                name="large",
                shape=tuple(large.shape),
                dtype="float32",
                chunk_offset=17,
                chunk_size=remaining,
                bucket_offset=0,
            ),
            TensorChunkMeta(
                name="small",
                shape=tuple(small.shape),
                dtype="bfloat16",
                chunk_offset=0,
                chunk_size=small_bytes.numel(),
                bucket_offset=remaining,
            ),
        ],
    )

    assembler = TensorChunkAssembler()
    assert assembler.consume(first_meta, first) == []
    completed = dict(assembler.consume(second_meta, second))

    torch.testing.assert_close(completed["large"], large)
    torch.testing.assert_close(completed["small"], small)


def test_sender_fills_bucket_across_tensor_boundaries(monkeypatch):
    metadata = []
    payloads = []

    class _Store:
        def set(self, key, value):
            metadata.append((key, BucketMeta.from_bytes(value)))

    class _Work:
        def wait(self):
            return None

    def _capture_broadcast(process_group, tensor):
        payloads.append(tensor.clone())
        return _Work()

    monkeypatch.setattr("awex.publication.verl_nccl._broadcast", _capture_broadcast)
    monkeypatch.setattr(device_util, "synchronize", lambda: None)
    sender = VerlNcclBroadcastSender(
        host="127.0.0.1",
        port=1234,
        world_size=2,
        group_id="test",
        bucket_size=8,
        timeout_seconds=1,
    )
    sender.store = _Store()
    sender.process_group = object()
    sender.buffers = [torch.empty(8, dtype=torch.uint8) for _ in range(2)]

    metrics = sender.broadcast_weights(
        7,
        [
            ("first", torch.arange(6, dtype=torch.uint8)),
            ("second", torch.arange(10, 14, dtype=torch.uint8)),
        ],
    )

    assert metrics == {"payload_bytes": 10, "tensor_count": 2, "bucket_count": 2}
    assert [item.length for _, item in metadata] == [8, 2]
    assert [item.is_last for _, item in metadata] == [False, True]
    assert torch.equal(
        payloads[0], torch.tensor([0, 1, 2, 3, 4, 5, 10, 11], dtype=torch.uint8)
    )
    assert torch.equal(payloads[1], torch.tensor([12, 13], dtype=torch.uint8))


def test_result_metrics_are_found_inside_vllm_wrappers():
    response = [
        {
            "results": [
                {
                    "publication_rank": 1,
                    "payload_bytes": 32,
                    "bucket_count": 2,
                }
            ]
        }
    ]

    flattened = list(_flatten_result_dicts(response))

    assert any(item.get("publication_rank") == 1 for item in flattened)


def test_full_tensor_exporter_prefers_mbridge_generator(monkeypatch):
    expected = torch.arange(4, dtype=torch.float32)

    class _Bridge:
        def export_weights(self, models):
            assert len(models) == 1
            yield "model.weight", expected

    engine = type(
        "Engine",
        (),
        {"model": object(), "publication_bridge": _Bridge()},
    )()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    exporter = McoreFullTensorExporter(engine, inference_tp_size=1)

    exporter.initialize()

    exported = list(exporter.iter_full_tensors())
    assert len(exported) == 1
    assert exported[0][0] == "model.weight"
    assert exported[0][1] is expected


def test_verl_mechanism_uses_common_file_writer_without_awex_meta(monkeypatch):
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cuda")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "0,1,2")
    config = copy.deepcopy(vllm_inference_config)
    config["tp_size"] = 1

    integration = VLLMWeightsExchangeIT(
        inference_config=config,
        comm_backend="nccl",
        train_tp_size=2,
        publication_mechanism="verl_nccl_broadcast",
    )

    assert integration.train_config["comm_backend"] == "file"
    assert not integration.publication.uses_awex_meta_server


def test_context_parallelism_contributes_to_dense_world_size():
    parallelism = resolve_megatron_parallelism(
        tp_size=2,
        pp_size=4,
        cp_size=2,
        ep_size=4,
        expert_tp_size=1,
    )

    assert parallelism.dense_parallel_size == 16
    assert parallelism.expert_parallel_size == 16
    parallelism.validate_world_size(16)


def test_remote_multi_engine_publication_endpoints(monkeypatch):
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cuda")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "0")
    config = copy.deepcopy(vllm_inference_config)
    config["num_engines"] = 2

    integration = MultiVLLMWeightsExchangeIT(
        inference_config=config,
        comm_backend="nccl",
        host="10.0.0.8",
        port=18000,
        remote_inference=True,
    )

    assert integration.publication_endpoints() == [
        (0, "10.0.0.8", 18000),
        (1, "10.0.0.8", 18001),
    ]
    assert integration.vllm_visible_devices == []


def test_bucket_receiver_offsets_rank_for_second_engine(monkeypatch):
    class _Work:
        def wait(self):
            return None

    monkeypatch.setattr(
        "awex.publication.verl_nccl._create_tcp_store", lambda *args, **kwargs: object()
    )
    captured = {}

    def _create_process_group(store, prefix, rank, world_size, timeout_seconds):
        captured.update(rank=rank, world_size=world_size)
        return object()

    monkeypatch.setattr(
        "awex.publication.verl_nccl._create_nccl_process_group",
        _create_process_group,
    )
    monkeypatch.setattr(
        "awex.publication.verl_nccl._broadcast", lambda *args, **kwargs: _Work()
    )
    monkeypatch.setattr(device_util, "get_torch_device", lambda: torch.device("cpu"))
    receiver = VerlNcclBroadcastReceiver(
        {
            "store_host": "127.0.0.1",
            "store_port": 1234,
            "world_size": 9,
            "group_id": "test",
            "bucket_size": 8,
            "timeout_seconds": 1,
            "rank_offset": 4,
        },
        worker_rank=2,
    )

    assert receiver.initialize() == {"publication_rank": 7}
    assert captured == {"rank": 7, "world_size": 9}

"""Run veRL's installed checkpoint engine in the standalone Awex harness.

Ray actor context, single-worker GPU identity and collective teardown are
bridged: harness workers are independent processes rather than Ray actors.
Packing, metadata PUB/SUB, double buffering, and NCCL broadcasts remain veRL's.
"""

import asyncio
import hashlib
import inspect
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import requests
import torch

from awex import logging
from awex.publication.registry import (
    register_publication_mechanism,
    register_vllm_publication_receiver,
)
from awex.publication.verl_nccl import (
    McoreFullTensorExporter,
    VerlNcclBroadcastPublicationMechanism,
    publication_endpoints,
)
from awex.util.profile import emit_profile, profile_phase

logger = logging.getLogger(__name__)


def _engine_class():
    import ray
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

    source = Path(inspect.getfile(NCCLCheckpointEngine))
    logger.info(
        "Native veRL NCCLCheckpointEngine source=%s sha256=%s",
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    if not ray.is_initialized():
        ray.init(
            address=os.environ.get("AWEX_NATIVE_VERL_RAY_ADDRESS", "11.18.56.89:6379"),
            namespace="awex-native-verl",
            logging_level="ERROR",
        )
    return NCCLCheckpointEngine


@contextmanager
def _standalone_context():
    # Keep the original Ray collective implementation; allow this explicitly
    # selected standalone integration harness to call it outside an actor.
    from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup

    original_barrier = NCCLGroup.barrier
    original_group_key = NCCLGroup._generate_group_key

    def current_device_barrier(group, *args, **kwargs):
        # veRL Ray actors normally see one GPU. Harness workers see the node's
        # GPU list, but each owns only its current device. Ray's first barrier
        # otherwise creates a communicator across every visible GPU per rank.
        if not group._used_gpu_indices:
            group._used_gpu_indices.add(torch.cuda.current_device())
        return original_barrier(group, *args, **kwargs)

    def worker_group_key(group, comm_key):
        # Ray keys the shared rendezvous by local CUDA ordinals. A one-GPU
        # actor always sees ordinal 0; standalone workers can own different
        # physical ordinals. Normalize only this single-device shared key,
        # keeping the original local cache keys, devices and communicators.
        if comm_key == str(torch.cuda.current_device()):
            comm_key = "0"
        return original_group_key(group, comm_key)

    with ExitStack() as stack:
        stack.enter_context(
            patch("ray.util.collective.collective._check_inside_actor", lambda: None)
        )
        stack.enter_context(patch.object(NCCLGroup, "barrier", current_device_barrier))
        stack.enter_context(
            patch.object(NCCLGroup, "_generate_group_key", worker_group_key)
        )
        yield


class _MetadataCounter:
    """Observe native metadata without changing serialization or payloads."""

    def __init__(self, socket):
        self.socket = socket
        self.reset()

    def reset(self):
        self.payload_bytes = 0
        self.bucket_count = 0

    def _record(self, metadata):
        self.payload_bytes += int(metadata["length"])
        self.bucket_count += 1

    def send_pyobj(self, metadata, *args, **kwargs):
        self._record(metadata)
        return self.socket.send_pyobj(metadata, *args, **kwargs)

    def recv_pyobj(self, *args, **kwargs):
        metadata = self.socket.recv_pyobj(*args, **kwargs)
        self._record(metadata)
        return metadata

    def __getattr__(self, name):
        return getattr(self.socket, name)


def _close_engine(engine):
    import ray.util.collective as collective

    with _standalone_context():
        if collective.is_group_initialized(engine.group_name):
            collective.destroy_collective_group(engine.group_name)
    engine.finalize()
    engine.socket.close(linger=0)


class NativeVerlSender:
    def __init__(self, *, world_size, group_id, bucket_size):
        self.world_size = world_size
        self.engine = _engine_class()(
            bucket_size=bucket_size, group_name=group_id, is_master=True
        )
        self.metadata = self.engine.prepare()
        self.host, self.port = self.metadata.zmq_ip, self.metadata.zmq_port
        self.engine.socket = _MetadataCounter(self.engine.socket)

    def initialize_process_group(self):
        with _standalone_context():
            self.engine.init_process_group(0, self.world_size, self.metadata)

    def broadcast_weights(self, step_id, weights):
        started = time.perf_counter()
        self.engine.socket.reset()
        tensor_count = 0

        def counted_weights():
            nonlocal tensor_count
            for name, tensor in weights:
                tensor_count += 1
                yield name, tensor

        with _standalone_context():
            asyncio.run(
                self.engine.send_weights(counted_weights(), global_steps=step_id)
            )
        result = {
            "payload_bytes": self.engine.socket.payload_bytes,
            "tensor_count": tensor_count,
            "bucket_count": self.engine.socket.bucket_count,
        }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        emit_profile(
            logger,
            event="weight_transfer",
            role="writer",
            rank=0,
            backend="verl_native_nccl",
            comm_backend="verl-nccl-bucket",
            step_id=step_id,
            phase=profile_phase(step_id),
            backend_execute_time_ms=elapsed_ms,
            native_pack_and_transfer_time_ms=elapsed_ms,
            **result,
        )
        return result

    def close(self):
        _close_engine(self.engine)


@register_vllm_publication_receiver("verl_native_nccl")
class NativeVerlReceiver:
    def __init__(self, config, worker_rank):
        self.config = config
        self.rank = int(config.get("rank_offset", 0)) + worker_rank + 1
        self.engine = None

    def initialize(self):
        from verl.checkpoint_engine.nccl_checkpoint_engine import MasterMetadata

        world_size = int(self.config["world_size"])
        if not 1 <= self.rank < world_size:
            raise ValueError(f"Invalid native veRL receiver rank {self.rank}")
        self.engine = _engine_class()(
            bucket_size=int(self.config["bucket_size"]),
            group_name=self.config["group_id"],
            is_master=False,
        )
        self.engine.prepare()
        metadata = MasterMetadata(
            zmq_ip=self.config["store_host"], zmq_port=int(self.config["store_port"])
        )
        with _standalone_context():
            self.engine.init_process_group(self.rank, world_size, metadata)
        self.engine.socket = _MetadataCounter(self.engine.socket)
        return {"publication_rank": self.rank}

    @torch.no_grad()
    def update(self, model, step_id):
        if self.engine is None:
            raise RuntimeError("Native veRL receiver is not initialized")
        started = time.perf_counter()
        self.engine.socket.reset()
        tensor_count = 0

        async def load():
            nonlocal tensor_count
            async for name, tensor in self.engine.receive_weights(global_steps=step_id):
                model.load_weights(iter([(name, tensor)]))
                tensor_count += 1
            torch.cuda.synchronize()

        with _standalone_context():
            asyncio.run(load())
        result = {
            "publication_rank": self.rank,
            "payload_bytes": self.engine.socket.payload_bytes,
            "received_tensors": tensor_count,
            "bucket_count": self.engine.socket.bucket_count,
        }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        emit_profile(
            logger,
            event="weight_transfer",
            role="reader",
            rank=self.rank,
            backend="verl_native_nccl",
            comm_backend="verl-nccl-bucket",
            step_id=step_id,
            phase=profile_phase(step_id),
            backend_execute_time_ms=elapsed_ms,
            native_receive_and_load_time_ms=elapsed_ms,
            **result,
        )
        return result

    def close(self):
        if self.engine is not None:
            _close_engine(self.engine)
            self.engine = None
        return {"closed": True}


@register_publication_mechanism("verl_native_nccl")
class NativeVerlPublicationMechanism(VerlNcclBroadcastPublicationMechanism):
    """Reuse harness orchestration, with the actual veRL checkpoint data path."""

    def initialize_training(self):
        harness = self.harness
        self.exporter = McoreFullTensorExporter(
            harness.megatron_engine,
            inference_tp_size=harness.inference_config["tp_size"],
        )
        self.exporter.initialize()
        if harness.is_driver:
            self.sender = NativeVerlSender(
                world_size=int(harness.inference_config["tp_size"])
                * int(harness.inference_config.get("num_engines", 1))
                + 1,
                group_id=self.group_id,
                bucket_size=self.bucket_size,
            )

    def close(self):
        # ncclCommDestroy can wait for peers. All receivers and the sender
        # must join teardown together, rather than sequential HTTP requests.
        # Keep sender CUDA calls on this thread's already selected device.
        def close_receiver(endpoint):
            engine_rank, host, port = endpoint
            try:
                response = requests.post(
                    f"http://{host}:{port}/publication_close",
                    timeout=min(self.timeout_seconds, 60),
                )
                if response.status_code != 200:
                    logger.warning(
                        "Native publication close failed for engine %s: %s",
                        engine_rank,
                        response.text,
                    )
            except requests.RequestException as exc:
                logger.warning(
                    "Native publication close request failed for engine %s: %s",
                    engine_rank,
                    exc,
                )

        endpoints = list(publication_endpoints(self.harness))
        with ThreadPoolExecutor(max_workers=max(1, len(endpoints))) as executor:
            futures = [executor.submit(close_receiver, item) for item in endpoints]
            if self.sender is not None:
                self.sender.close()
                self.sender = None
            for future in futures:
                future.result()

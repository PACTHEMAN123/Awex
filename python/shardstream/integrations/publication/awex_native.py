"""Run the upstream Awex writer/reader under the common model harness."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os

import torch

import requests
import torch.distributed as dist

from shardstream.integrations.publication.registry import (
    PublicationMechanism, publication_endpoints, register_publication_mechanism,
)


def close_awex_nccl_group(holder, role):
    """Release only the benchmark-owned exchange group on every participant."""
    group = getattr(holder, "weights_update_group", None)
    if group is None:
        return {"closed": True, "already_closed": True, "pid": os.getpid()}
    # Both trainer ranks and every rollout TP/DP worker enter this collective
    # before any default/training or inference process group is destroyed.
    torch.cuda.synchronize()
    dist.barrier(group=group)
    dist.destroy_process_group(group)
    holder.weights_update_group = None
    holder.already_initialized = False
    result = dict(closed=True, already_closed=False, role=role, pid=os.getpid(),
                  rank=holder.transfer_rank)
    print("AWEX_NCCL_CLOSE " + json.dumps(result), flush=True)
    return result


def close_awex_nccl_worker(**kwargs):
    reader = kwargs["model_context"]["scheduler"].awes_weights_reader
    return close_awex_nccl_group(reader, "receive")


@register_publication_mechanism('awex_native_nccl')
class AwexNCCLPublication(PublicationMechanism):
    # The harness loads the same Megatron model before creating the upstream
    # Awex engine. No ShardStream transport or metadata service is initialized.
    training_engine_backend = None
    comm_backend = "nccl"

    def __init__(self, harness, timeout_seconds=1800, **kwargs):
        super().__init__(harness, **kwargs)
        self.timeout_seconds = timeout_seconds

    def initialize_training(self):
        from awex.engine.mcore import MegatronEngine
        from awex.meta.meta_server import start_meta_server

        harness = self.harness
        address = None
        if harness.is_driver:
            host, port = start_meta_server(host=harness.meta_server_host,
                                          port=harness.meta_server_port)
            address = f'{host}:{port}'
        addresses = [address]
        if harness.world_size > 1:
            dist.broadcast_object_list(addresses, src=0)
        harness.meta_server_addr = addresses[0]
        harness.train_config.update(comm_backend=self.comm_backend, meta_server_addr=addresses[0])
        harness.inference_config.update(comm_backend=self.comm_backend, meta_server_addr=addresses[0])

        publication = self

        class BenchmarkEngine(MegatronEngine):
            def close(self):
                if publication.comm_backend == "nccl":
                    publication.close_training()

        harness.megatron_engine = BenchmarkEngine(
            harness.train_config, harness.mcore_hf_config, harness.mcore_model)
        harness.megatron_engine.initialize()

    def initialize_driver(self):
        endpoints = publication_endpoints(self.harness)
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [executor.submit(self._initialize_endpoint, endpoint) for endpoint in endpoints]
            for future in futures:
                future.result()

    def _initialize_endpoint(self, endpoint):
        rank, host, port = endpoint
        payload = {'meta_server_addr': self.harness.meta_server_addr,
                   'engine_rank': rank,
                   'num_engines': self.harness.inference_config['num_engines'],
                   'comm_backend': self.comm_backend, 'enable_debug_mode': False,
                   'nnodes': 1, 'node_rank': 0}
        response = requests.post(f'http://{host}:{port}/areal_awex_init', json=payload,
                                 timeout=self.timeout_seconds)
        if response.status_code != 200:
            raise RuntimeError(f'Awex init {rank}: {response.text}')

    def publish(self):
        harness = self.harness
        context = ThreadPoolExecutor(max_workers=1) if harness.is_driver else nullcontext()
        with context as executor:
            future = executor.submit(self._request_update) if executor is not None else None
            harness._training_barrier()
            harness.megatron_engine.write_weights()
            if future is not None:
                future.result()

    def _request_update(self):
        payload = {'step_id': int(self.harness.megatron_engine.global_step), 'kwargs': {}}
        endpoints = publication_endpoints(self.harness)
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [executor.submit(requests.post, f'http://{host}:{port}/areal_awex_update',
                                       json=payload, timeout=self.timeout_seconds)
                       for _, host, port in endpoints]
            for endpoint, future in zip(endpoints, futures):
                response = future.result()
                if response.status_code != 200:
                    raise RuntimeError(f'Awex update {endpoint[0]}: {response.text}')

    def _request_close(self):
        endpoints = publication_endpoints(self.harness)
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [executor.submit(requests.post,
                       f'http://{host}:{port}/awex_benchmark_close', timeout=120)
                       for _, host, port in endpoints]
            expected = self.harness.inference_config['tp_size'] * self.harness.inference_config.get('dp_size', 1)
            for endpoint, future in zip(endpoints, futures):
                response = future.result()
                response.raise_for_status()
                workers = response.json()['ranks']
                if len(workers) != expected or not all(w['closed'] for w in workers):
                    raise RuntimeError(f'Incomplete Awex NCCL close on engine {endpoint[0]}')

    def close_training(self):
        # Run receiver close RPCs concurrently with trainer-side close. NCCL
        # communicator finalization can wait for remote participants.
        context = ThreadPoolExecutor(max_workers=1) if self.harness.is_driver else nullcontext()
        with context as executor:
            future = executor.submit(self._request_close) if executor is not None else None
            result = close_awex_nccl_group(self.harness.megatron_engine.weights_exchange_writer, 'send')
            if future is not None:
                future.result()
            return result

    def close(self):
        if self.harness.is_driver and self.harness.meta_server_addr is not None:
            from awex.meta.meta_server import stop_meta_server
            stop_meta_server()
            self.harness.meta_server_addr = None


@register_publication_mechanism("awex_native_rdma")
class AwexRDMAPublication(AwexNCCLPublication):
    """Official Awex Mooncake planner, writer and reader using GPU RDMA."""

    comm_backend = "mooncake"

    def initialize_training(self):
        from shardstream.integrations.publication.awex_rdma import enable_rdma_audit

        enable_rdma_audit()
        super().initialize_training()


@register_publication_mechanism("awex_native_astate")
class AwexAStatePublication(AwexNCCLPublication):
    """Official Awex AState writer/reader with persistent host staging."""

    comm_backend = "astate"

    def initialize_training(self):
        from shardstream.integrations.publication.awex_astate import enable_astate_audit

        enable_astate_audit()
        super().initialize_training()

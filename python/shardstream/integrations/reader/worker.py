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


import gc
import os
import time

import torch
import torch.distributed as dist

from shardstream import logging
from shardstream._utils import device as device_util
from shardstream._utils.common import compute_statistics, get_free_port, get_ip_address
from shardstream._utils.profile import emit_profile, profile_phase
from shardstream.integrations.models.device_layout import (
    STATIC_DEVICE_LAYOUT_ARCHITECTURES,
    annotate_device_transfer_plan,
)
from shardstream.integrations.reader.base import WorkerWeightsReader
from shardstream.plan import (
    TransferPlanBuilder,
    compute_transfer_plan_hash,
    compute_transfer_plan_stats,
)
from shardstream.transport import Transport

logger = logging.getLogger(__name__)


class TransportWorkerReader(WorkerWeightsReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_plan = None
        self.weights_update_group = None
        self.send_ranks = None
        self.send_ranks_sample = None
        self.num_to_recvs = None
        self.rank_coordinate = None
        self.device_transport = None

    def initialize(self):
        super().initialize()
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta, self.training_params_meta, self.transfer_rank
        )
        if self.model_arch_name in STATIC_DEVICE_LAYOUT_ARCHITECTURES:
            annotated = annotate_device_transfer_plan(
                self.transfer_plan, self.model_arch_name, self.hf_config
            )
            logger.info(
                "Reader rank %s annotated %s source layout operations",
                self.transfer_rank,
                annotated,
            )
        inter_hash = compute_transfer_plan_hash(self.transfer_plan)
        logger.info(
            "Reader rank %s inter plan hash: %s", self.transfer_rank, inter_hash
        )
        logger.info(
            "Reader rank %s transfer plan stats: %s",
            self.transfer_rank,
            compute_transfer_plan_stats(self.transfer_plan),
        )
        if self.transfer_rank == 0:
            master_address = get_ip_address()
            master_port = get_free_port()
            master_info = (master_address, master_port)
            self.meta_server_client.put_object("master_info", master_info)
            logger.info(
                f"Put master info to meta server for rank {self.transfer_rank}: {master_info}"
            )
        else:
            master_info = self.meta_server_client.get_object(
                "master_info", timeout=self.timeout
            )
            (master_address, master_port) = master_info
            logger.info(
                f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
            )
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )
        self.master_address = master_address
        self.master_port = master_port
        self.world_size = self.transfer_world_size
        self._set_device()
        self._init_weights_exchange_process_group()
        self._shake_hands_with_writer()
        transport_type = Transport
        self.device_transport = transport_type(
            self.weights_update_group,
            self.transfer_rank,
            self.world_size,
            infer_instance_world_size=self.infer_instance_world_size,
            num_infer_engines=self.num_engines,
        )
        self.device_transport.resolve_rollout_topology()
        if self.model_arch_name in STATIC_DEVICE_LAYOUT_ARCHITECTURES:
            self.device_transport.prepare_recv(
                self.parameters, self.transfer_plan, allow_staging=False
            )
        self.send_ranks = list(self.transfer_plan.operations.keys())
        self.send_ranks_sample = (
            self.send_ranks[:8] + ["..."] + self.send_ranks[-8:]
            if len(self.send_ranks) > 16
            else self.send_ranks
        )
        self.num_to_recvs = sum(
            (len(operations) for operations in self.transfer_plan.operations.values())
        )
        self.rank_coordinate = (
            f"{self.engine_rank}-{self.rank_info.global_rank}-{self.transfer_rank}"
        )
        self.deserialized_weights = {}
        logger.info(
            f"Created NCCL weights reader for rank {self.rank_info.global_rank}, engine rank {self.engine_rank}"
        )

    def _shake_hands_with_writer(self):
        from shardstream._utils.process_group import setup_batch_isend_irecv

        if self.transfer_rank == 0:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.recv(
                self.ready_tensor,
                src=self.world_size - 1,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: recv tensor from rank 0 for rank {self.transfer_rank}"
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, self.world_size
        )

    def _set_device(self):
        gpu_id = getattr(self.scheduler, "gpu_id", None) or getattr(
            self.scheduler, "local_rank", None
        )
        if gpu_id is None:
            gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        device_type = device_util.get_device_type()
        device_count = device_util.device_count() or 1
        if device_type == "cuda":
            prev_device = torch.cuda.current_device()
            logger.info(
                f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, previous device is {prev_device}, device_count is {device_count}, CUDA_VISIBLE_DEVICES env is {os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            torch.cuda.set_device(gpu_id)
            self.barrier_device = torch.cuda.current_device()
            self.backend = "nccl"
            self.ready_tensor = torch.tensor(1).cuda()
        else:
            logger.info(
                f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, previous device is {device_util.current_device()}, device_count is {device_count}, {'/'.join(device_util.visible_devices_env_names())} env is {device_util.visible_devices_env_value() or '(unset)'}"
            )
            device_util.set_device(gpu_id)
            self.barrier_device = device_util.current_device()
            self.backend = self.comm_backend
            self.ready_tensor = torch.tensor(1, device=device_util.get_torch_device())

    def _init_weights_exchange_process_group(self):
        if self.already_initialized:
            return
        from shardstream._utils.process_group import init_weights_update_group

        self.weights_update_group = init_weights_update_group(
            master_address=self.master_address,
            master_port=self.master_port,
            rank=self.transfer_rank,
            world_size=self.world_size,
            group_name="weights_exchange",
            backend="nccl",
            role="inference",
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank}, engine rank {self.engine_rank}"
        )
        dist.barrier(group=self.weights_update_group, device_ids=[self.barrier_device])
        logger.info(f"Barrier passed for weights reader with rank {self.transfer_rank}")
        self.already_initialized = True

    def _destroy_weights_exchange_process_group(self):
        pass

    def _update_weights(self, step_id, **kwargs):
        """
        Asynchronously receive weights from training ranks using torch.distributed.irecv.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to receive
        2. We start irecv operations in parallel from all sender ranks
        3. When a receive completes, we immediately start the next receive from that rank
        4. We continue until all operations from all sender ranks are completed

        Args:
            step_id: The training step ID
            **kwargs: Additional keyword arguments (unused)
        """
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from {len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) for rank {self.rank_coordinate}."
        )
        self._init_weights_exchange_process_group()
        start_time = time.perf_counter()
        p2p_op_list = None
        profile_metrics = {
            "build_batch_time_ms": 0.0,
            "metadata_upload_time_ms": 0.0,
            "kernel_transfer_time_ms": 0.0,
            "reader_wait_time_ms": 0.0,
            "reader_copyback_time_ms": 0.0,
            "payload_bytes": 0.0,
        }
        logger.info("Reader: submitting device task batch")
        sync_start_barrier_time_ms = 0.0
        if os.environ.get("SHARDSTREAM_PROFILE_SYNC_START", "0") == "1":
            sync_start = time.perf_counter()
            dist.barrier(
                group=self.weights_update_group,
                device_ids=[device_util.current_device()],
            )
            sync_start_barrier_time_ms = (time.perf_counter() - sync_start) * 1000.0
        backend_execute_start = time.perf_counter()
        profile_metrics.update(
            self.device_transport.recv(self.parameters, self.transfer_plan, step_id)
        )
        device_util.synchronize(device_id=device_util.current_device())
        profile_metrics["backend_execute_time_ms"] = (
            time.perf_counter() - backend_execute_start
        ) * 1000.0
        duration = time.perf_counter() - start_time
        logger.info(
            f"Finished receiving weights for step {step_id} using NCCL from {len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) to rank {self.rank_coordinate} with {self.num_to_recvs} receives, took {duration:.4f} seconds"
        )
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        completion_barrier_start = time.perf_counter()
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        completion_barrier_time_ms = (
            time.perf_counter() - completion_barrier_start
        ) * 1000.0
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )
        kernel_transfer_time_ms = profile_metrics.get("kernel_transfer_time_ms", 0.0)
        effective_gbps = 0.0
        if kernel_transfer_time_ms > 0:
            effective_gbps = profile_metrics.get("payload_bytes", 0.0) / (
                kernel_transfer_time_ms * 1000000.0
            )
        backend_execute_time_ms = profile_metrics.get("backend_execute_time_ms", 0.0)
        backend_effective_gbps = 0.0
        if backend_execute_time_ms > 0:
            backend_effective_gbps = profile_metrics.get("payload_bytes", 0.0) / (
                backend_execute_time_ms * 1000000.0
            )
        self.last_transfer_metrics = dict(profile_metrics)
        emit_profile(
            logger,
            event="weight_transfer",
            role="reader",
            backend=self.comm_backend,
            phase=profile_phase(step_id),
            step_id=int(step_id),
            rank=int(self.transfer_rank),
            sync_start_barrier_time_ms=sync_start_barrier_time_ms,
            completion_barrier_time_ms=completion_barrier_time_ms,
            total_transfer_time_ms=duration * 1000.0,
            effective_gbps=effective_gbps,
            backend_effective_gbps=backend_effective_gbps,
            **profile_metrics,
        )
        should_collect_garbage = p2p_op_list is not None
        resource_cleanup_start = time.perf_counter()
        if p2p_op_list is not None:
            p2p_op_list.clear()
            del p2p_op_list
        self._destroy_weights_exchange_process_group()
        resource_cleanup_time_ms = (
            time.perf_counter() - resource_cleanup_start
        ) * 1000.0
        gc_collect_time_ms = 0.0
        if should_collect_garbage:
            gc_collect_start = time.perf_counter()
            gc.collect()
            gc_collect_time_ms = (time.perf_counter() - gc_collect_start) * 1000.0
        emit_profile(
            logger,
            event="weight_transfer_cleanup",
            role="reader",
            backend=self.comm_backend,
            phase=profile_phase(step_id),
            step_id=int(step_id),
            rank=int(self.transfer_rank),
            resource_cleanup_time_ms=resource_cleanup_time_ms,
            gc_collect_time_ms=gc_collect_time_ms,
            gc_collect_skipped=not should_collect_garbage,
        )

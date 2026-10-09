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
from shardstream._utils.common import compute_statistics
from shardstream._utils.gpu import print_current_gpu_status
from shardstream._utils.process_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from shardstream._utils.profile import emit_profile, profile_phase
from shardstream.integrations.writer.base import WeightsExchangeShardingWriter
from shardstream.plan import (
    TransferPlanBuilder,
    compute_transfer_plan_hash,
    compute_transfer_plan_stats,
)
from shardstream.transport import Transport

logger = logging.getLogger(__name__)
_QWEN3_STATIC_DEVICE_LAYOUT_ARCHITECTURES = {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"}


class TransportWriter(WeightsExchangeShardingWriter):
    def prepare_membership(self, specification: dict) -> dict:
        from shardstream.integrations.membership import prepare_model_membership

        with self.lock:
            if not self.initialized:
                raise RuntimeError("Publish the initial model snapshot before joining")
            return prepare_model_membership(self, specification, sender=True)

    def _initialize(self):
        super()._initialize()
        self.device_transport = None
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )
        logger.info(f"Start to build transfer plan for rank {self.transfer_rank}")
        self.transfer_plan = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_infer_engines,
            self.enable_debug_mode,
        ).build_local_transfer_plan(
            self.infer_params_meta, self.parameters_meta, self.transfer_rank
        )
        if self.model_arch_name in _QWEN3_STATIC_DEVICE_LAYOUT_ARCHITECTURES:
            from shardstream.integrations.models.qwen3 import (
                annotate_qwen3_dense_transfer_plan,
            )

            annotated = annotate_qwen3_dense_transfer_plan(
                self.transfer_plan, self.hf_config
            )
            logger.info(
                "Writer rank %s annotated %s Qwen3 device operations",
                self.transfer_rank,
                annotated,
            )
        inter_hash = compute_transfer_plan_hash(self.transfer_plan)
        logger.info(
            "Writer rank %s inter plan hash: %s", self.transfer_rank, inter_hash
        )
        logger.info(
            "Writer rank %s transfer plan stats: %s",
            self.transfer_rank,
            compute_transfer_plan_stats(self.transfer_plan),
        )
        self.recv_ranks = list(self.transfer_plan.operations.keys())
        self.required_param_names = {
            op.send_shard_meta.name
            for ops in self.transfer_plan.operations.values()
            for op in ops
        }
        self.device_parameters = None
        if self.model_arch_name in _QWEN3_STATIC_DEVICE_LAYOUT_ARCHITECTURES:
            self.device_parameters = self.compile_device_parameters(
                self.required_param_names
            )
        logger.info(
            f"Writer rank {self.transfer_rank}: Built transfer plan to send to ranks: {self.recv_ranks}"
        )
        logger.info(
            "Writer rank %s: required converted params for this plan: %s",
            self.transfer_rank,
            len(self.required_param_names),
        )
        logger.info(
            f"Writer rank {self.transfer_rank}: Operations per rank: {[(rank, len(ops)) for (rank, ops) in self.transfer_plan.operations.items()]}"
        )
        self.recv_ranks_sample = (
            self.recv_ranks[:8] + ["..."] + self.recv_ranks[-8:]
            if len(self.recv_ranks) > 16
            else self.recv_ranks
        )
        self.num_to_sends = sum(
            (len(operations) for operations in self.transfer_plan.operations.values())
        )
        logger.info(f"Finished building transfer plan for rank {self.transfer_rank}")
        logger.info(
            f"Start to get master info from meta server for rank {self.transfer_rank}"
        )
        master_info = self.meta_server_client.get_object(
            "master_info", timeout=self.timeout
        )
        (self.master_address, self.master_port) = master_info
        logger.info(
            f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
        )
        self._set_device()
        self._init_weights_exchange_process_group()
        self._shake_hands_with_reader()
        transport_type = Transport
        self.device_transport = transport_type(
            self.weights_update_group,
            self.transfer_rank,
            self.transfer_world_size,
            infer_instance_world_size=self.infer_instance_world_size,
            num_infer_engines=self.num_infer_engines,
        )
        if self.device_parameters is not None:
            self.device_transport.prepare_send(
                self.device_parameters, self.transfer_plan, allow_staging=False
            )
        logger.info(
            f"Finished initializing NCCL weights writer for rank {self.transfer_rank}"
        )

    def close(self) -> None:
        self.device_transport.close()
        self.device_transport = None

    def _shake_hands_with_reader(self):
        if self.transfer_rank == self.transfer_world_size - 1:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.send(
                torch.tensor(1, device=device_util.get_torch_device()),
                dst=0,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: send tensor to rank {self.transfer_world_size - 1} from rank {self.transfer_rank}"
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, self.transfer_world_size
        )

    def _set_device(self):
        device = device_util.current_device()
        count = device_util.device_count() or 1
        gpu_id = int(os.environ.get("DEVICE", device)) % count
        logger.info(
            f"[TransportWriter] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, previous device is {device}, device_count is {count}, {'/'.join(device_util.visible_devices_env_names())} env is {device_util.visible_devices_env_value() or '(unset)'}"
        )
        device_util.set_device(gpu_id)

    def _init_weights_exchange_process_group(self):
        if self.already_initialized:
            return
        self.weights_update_group = init_weights_update_group(
            master_address=self.master_address,
            master_port=self.master_port,
            rank=self.transfer_rank,
            world_size=self.transfer_world_size,
            group_name="weights_exchange",
            backend="nccl",
            role="train",
        )
        logger.info(f"Initialized NCCL weights writer for rank {self.transfer_rank}")
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        logger.info(f"Barrier passed for weights writer with rank {self.transfer_rank}")
        self.already_initialized = True

    def _destroy_weights_exchange_process_group(self):
        pass

    @torch.no_grad()
    def _write_weights(self, step_id, **kwargs):
        """
        Asynchronously send weights to inference ranks using torch.distributed.isend.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to send
        2. We start isend operations in parallel to all sender ranks
        3. When a send completes, we immediately start the next send to that rank
        4. We continue until all operations to all sender ranks are completed

        Args:
            step_id: The training step ID used as communication tag
            **kwargs: Additional keyword arguments (unused)
        """
        rank_coordinate = self.transfer_rank
        logger.info(
            f"Start to send weights using NCCL to {len(self.transfer_plan.operations)} ranks({self.recv_ranks_sample}) from rank {rank_coordinate} with {self.num_to_sends} sends"
        )
        self._init_weights_exchange_process_group()
        start_time = time.perf_counter()
        parameters = None
        parameters_are_static = False
        p2p_op_list = None
        using_device_transport = True
        profile_metrics = {
            "build_batch_time_ms": 0.0,
            "metadata_upload_time_ms": 0.0,
            "kernel_transfer_time_ms": 0.0,
            "payload_bytes": 0.0,
        }
        convert_time_ms = 0.0
        try:
            if using_device_transport and self.device_parameters is not None:
                parameters = self.device_parameters
                parameters_are_static = True
                logger.info(
                    "Writer: using compiled Qwen3 device parameters; skipping format conversion"
                )
            else:
                if self.enable_mem_debug:
                    print_current_gpu_status(
                        f"writer-{self.transfer_rank} before convert"
                    )
                convert_start = time.perf_counter()
                parameters = self.convert_parameters(
                    required_names=self.required_param_names
                )
                convert_time_ms = (time.perf_counter() - convert_start) * 1000.0
                logger.info("Writer: Converting parameters completed")
                if self.enable_mem_debug:
                    print_current_gpu_status(
                        f"writer-{self.transfer_rank} after convert"
                    )
            logger.info("Writer: submitting device task batch")
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
                self.device_transport.send(parameters, self.transfer_plan, step_id)
            )
            device_util.synchronize(device_id=device_util.current_device())
            profile_metrics["backend_execute_time_ms"] = (
                time.perf_counter() - backend_execute_start
            ) * 1000.0
            if self.enable_mem_debug:
                print_current_gpu_status(f"writer-{self.transfer_rank} after send")
            duration = time.perf_counter() - start_time
            logger.info(
                f"Finished sending weights for step {step_id} using NCCL to {len(self.transfer_plan.operations)} ranks({self.recv_ranks_sample}) from rank {rank_coordinate} with {self.num_to_sends} sends, took {duration:.4f} seconds"
            )
            compute_statistics(
                self._history_write_weights_time,
                step_id,
                duration,
                "Send weights using NCCL",
            )
            completion_barrier_start = time.perf_counter()
            dist.barrier(
                group=self.weights_update_group,
                device_ids=[device_util.current_device()],
            )
            completion_barrier_time_ms = (
                time.perf_counter() - completion_barrier_start
            ) * 1000.0
            logger.info(
                f"Barrier passed for writer step {step_id} with rank {self.transfer_rank}"
            )
            kernel_transfer_time_ms = profile_metrics.get(
                "kernel_transfer_time_ms", 0.0
            )
            effective_gbps = 0.0
            if kernel_transfer_time_ms > 0:
                effective_gbps = profile_metrics.get("payload_bytes", 0.0) / (
                    kernel_transfer_time_ms * 1000000.0
                )
            backend_execute_time_ms = profile_metrics.get(
                "backend_execute_time_ms", 0.0
            )
            backend_effective_gbps = 0.0
            if backend_execute_time_ms > 0:
                backend_effective_gbps = profile_metrics.get("payload_bytes", 0.0) / (
                    backend_execute_time_ms * 1000000.0
                )
            emit_profile(
                logger,
                event="weight_transfer",
                role="writer",
                backend=self.comm_backend,
                phase=profile_phase(step_id),
                step_id=int(step_id),
                rank=int(self.transfer_rank),
                convert_time_ms=convert_time_ms,
                sync_start_barrier_time_ms=sync_start_barrier_time_ms,
                completion_barrier_time_ms=completion_barrier_time_ms,
                total_transfer_time_ms=duration * 1000.0,
                effective_gbps=effective_gbps,
                backend_effective_gbps=backend_effective_gbps,
                **profile_metrics,
            )
        finally:
            resource_cleanup_start = time.perf_counter()
            if p2p_op_list is not None:
                p2p_op_list.clear()
            if parameters is not None and (not parameters_are_static):
                parameters.clear()
            self._destroy_weights_exchange_process_group()
            resource_cleanup_time_ms = (
                time.perf_counter() - resource_cleanup_start
            ) * 1000.0
            should_collect_garbage = p2p_op_list is not None or (
                parameters is not None and (not parameters_are_static)
            )
            gc_collect_time_ms = 0.0
            if should_collect_garbage:
                gc_collect_start = time.perf_counter()
                gc.collect()
                gc_collect_time_ms = (time.perf_counter() - gc_collect_start) * 1000.0
            emit_profile(
                logger,
                event="weight_transfer_cleanup",
                role="writer",
                backend=self.comm_backend,
                phase=profile_phase(step_id),
                step_id=int(step_id),
                rank=int(self.transfer_rank),
                resource_cleanup_time_ms=resource_cleanup_time_ms,
                gc_collect_time_ms=gc_collect_time_ms,
                gc_collect_skipped=not should_collect_garbage,
            )
            if self.enable_mem_debug:
                if device_util.get_device_type() == "cuda":
                    torch.cuda.empty_cache()
                print_current_gpu_status(f"writer-{self.transfer_rank} after cleanup")

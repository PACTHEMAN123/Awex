"""Prepare device-v2 membership against live Megatron/vLLM model storage.

Applications must finish the previous publication on all ranks first. These
operations replace only the borrowed transfer group and its transport plans;
the training group and loaded models stay alive.
"""

from __future__ import annotations

import os
import time

import torch.distributed as dist

from awex.models.qwen3 import annotate_qwen3_dense_transfer_plan
from awex.transfer.nccl_device_v2 import NCCLDeviceV2Transport, _local_node_id
from awex.transfer.rollout_membership import RolloutMembership
from awex.transfer.transfer_plan import TransferPlanBuilder
from awex.util.common import get_free_port, get_ip_address
from awex.util.process_group import init_weights_update_group


def prepare_model_membership(worker, specification: dict, sender: bool) -> dict:
    """Collectively prepare a larger homogeneous cohort on every live rank."""
    if worker.comm_backend != "nccl_device_v2" or worker.enable_colocate_mode:
        raise ValueError("Model membership requires non-colocated device v2")
    if worker.model_arch_name not in ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM"):
        raise ValueError("Model membership requires a stable Qwen3 device layout")
    epoch, engines = specification["epoch"], specification["num_engines"]
    transport = worker.device_transport
    if transport is not None:
        if (
            epoch != transport.membership_epoch + 1
            or engines <= transport.num_infer_engines
        ):
            raise ValueError("Model membership must advance one epoch and add engines")
    elif epoch < 1:
        raise ValueError("New model workers require an explicit join epoch")
    membership = RolloutMembership(
        epoch,
        worker.training_world_size,
        worker.infer_instance_world_size,
        tuple(f"engine-{i}" for i in range(engines)),
    )
    participant = (
        ("training", "", worker.rank_info.global_rank)
        if sender
        else ("rollout", f"engine-{worker.engine_rank}", worker.rank_info.global_rank)
    )
    rank = membership.rank(participant)
    phases = []
    started_ns = time.time_ns()
    started = time.perf_counter()
    models = worker.model if sender else [worker.model]
    pointers = {
        f"{stage}/{name}": parameter.data_ptr()
        for stage, model in enumerate(models)
        for name, parameter in model.named_parameters()
    }

    def phase(name, call):
        begin = time.time_ns()
        result = call()
        phases.append({"name": name, "start_ns": begin, "end_ns": time.time_ns()})
        return result

    plan = phase(
        "model_plan_build",
        lambda: TransferPlanBuilder(
            membership.inference_world_size,
            membership.training_world_size,
            engines,
            worker.enable_debug_mode,
        ).build_local_transfer_plan(
            worker.infer_params_meta if sender else worker.parameters_meta,
            worker.parameters_meta if sender else worker.training_params_meta,
            rank,
        ),
    )
    annotate_qwen3_dense_transfer_plan(plan, worker.hf_config)
    if sender:
        required = {
            op.send_shard_meta.name
            for operations in plan.operations.values()
            for op in operations
        }
        parameters = worker.device_parameters
        if parameters is None or set(parameters) != required:
            parameters = phase(
                "bind_source_views", lambda: worker.compile_device_parameters(required)
            )
        if parameters is None:
            raise ValueError("Model membership requires compiled stable source views")
    else:
        parameters = worker.parameters
    master_key = f"elastic/master/{epoch}"
    if rank == 0:
        worker.meta_server_client.put_object(
            master_key, (get_ip_address(), get_free_port())
        )
    master_address, master_port = worker.meta_server_client.get_object(
        master_key, timeout=300
    )
    group = phase(
        "process_group",
        lambda: init_weights_update_group(
            master_address,
            master_port,
            rank,
            membership.world_size,
            group_name=f"weights_exchange_epoch_{epoch}",
            backend="nccl",
            role="train" if sender else "inference",
        ),
    )
    old_group = worker.weights_update_group
    try:

        def rollout_topology():
            nodes = [None] * membership.world_size
            dist.all_gather_object(nodes, _local_node_id(), group=group)
            engine_nodes = []
            for engine in range(engines):
                tp_nodes = nodes[
                    engine * membership.inference_tp_size : (engine + 1)
                    * membership.inference_tp_size
                ]
                if len(set(tp_nodes)) != 1:
                    raise ValueError(
                        "Model joins require each rollout TP group on one node"
                    )
                engine_nodes.append(tp_nodes[0])
            return tuple(engine_nodes)

        rollout_node_ids = phase("rollout_topology", rollout_topology)
        if transport is None:
            transport = NCCLDeviceV2Transport(
                group,
                rank,
                membership.world_size,
                infer_instance_world_size=membership.inference_tp_size,
                num_infer_engines=engines,
                membership_epoch=epoch,
                rollout_node_ids=rollout_node_ids,
            )
            phase(
                "host_bind",
                lambda: (transport.prepare_send if sender else transport.prepare_recv)(
                    parameters, plan, allow_staging=False
                ),
            )
        else:
            phase(
                "old_release_and_host_bind",
                lambda: transport.reconfigure(
                    group,
                    membership,
                    participant,
                    parameters,
                    plan,
                    rollout_node_ids=rollout_node_ids,
                ),
            )
        metrics = phase(
            "device_communicator_fifo_cache",
            lambda: transport.initialize_prepared_plan(parameters, plan, sender),
        )
        if not transport.is_device_plan_ready(parameters, plan, sender):
            raise RuntimeError("Model device cache is not ready")
        phase("ready_barrier", lambda: dist.barrier(group=group))
    except Exception:
        transport.close()
        dist.destroy_process_group(group)
        raise
    if old_group is not None:
        dist.destroy_process_group(old_group)
    if pointers != {
        f"{stage}/{name}": parameter.data_ptr()
        for stage, model in enumerate(models)
        for name, parameter in model.named_parameters()
    }:
        raise AssertionError("Preparing membership replaced loaded model storage")
    worker.weights_update_group = group
    worker.device_transport = transport
    worker.transfer_plan = plan
    worker.transfer_rank = rank
    worker.transfer_world_size = membership.world_size
    worker.infer_world_size = membership.inference_world_size
    worker.already_initialized = True
    worker.master_address, worker.master_port = master_address, master_port
    if sender:
        worker.num_infer_engines = engines
        worker.device_parameters = parameters
        worker.required_param_names = required
        worker.recv_ranks = list(plan.operations)
        worker.recv_ranks_sample = worker.recv_ranks[:8]
        worker.num_to_sends = sum(map(len, plan.operations.values()))
    else:
        worker.num_engines = engines
        worker.world_size = membership.world_size
        worker.send_ranks = list(plan.operations)
        worker.send_ranks_sample = worker.send_ranks[:8]
        worker.num_to_recvs = sum(map(len, plan.operations.values()))
        worker.rank_coordinate = (
            f"{worker.engine_rank}-{worker.rank_info.global_rank}-{rank}"
        )
    return {
        "pid": os.getpid(),
        "pointers": pointers,
        "epoch": epoch,
        "rank": rank,
        "start_ns": started_ns,
        "end_ns": time.time_ns(),
        "prepare_ms": (time.perf_counter() - started) * 1000,
        "phases": phases,
        "preparation_metrics": metrics,
        "rollout_node_ids": rollout_node_ids,
    }

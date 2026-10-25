"""Exercise official AState across hosts, retaining its default buffer sizes."""
import argparse
import ctypes
import json
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["send", "receive"])
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--counter-library", required=True)
    args = parser.parse_args()
    send = args.role == "send"
    local_port, peer_port = (30171, 30181) if send else (30181, 30171)
    os.environ.update(
        POD_IP=args.local_ip,
        ASTATE_OPTIONS_LOAD_MODE="ENV",
        TENSOR_TRANSFER_SERVICE_TYPE="PULL",
        TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY="true",
        TRANSFER_ENGINE_SERVICE_FIXED_PORT="true",
        TRANSFER_ENGINE_LOCAL_ADDRESS=args.local_ip,
        TRANSFER_ENGINE_LOCAL_PORT=str(local_port),
        TRANSFER_ENGINE_SERVICE_ADDRESS=args.local_ip,
        TRANSFER_ENGINE_SERVICE_PORT=str(local_port + 1),
        TRANSFER_ENGINE_PEERS_HOST=f"{args.peer_ip}:{peer_port}:{peer_port + 1}",
        TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION="false",
        ASTATE_LOG_TO_CONSOLE="true",
        TRANSFER_ENGINE_LOG_TENSOR_META="false",
    )
    assert "UTRANS_DISABLE_RDMA" not in os.environ
    import torch
    import astate
    from astate.utils import create_sharded_key
    from shardstream.integrations.publication.awex_astate import configure_astate_devices

    torch.cuda.set_device(0)
    devices = configure_astate_devices()
    counters = ctypes.CDLL(args.counter_library)
    for name in ("astate_probe_reg_calls", "astate_probe_mr_calls"):
        getattr(counters, name).restype = ctypes.c_ulonglong
    config_factory = (astate.ParallelConfig.create_training_config if send
                      else astate.ParallelConfig.create_inference_config)
    table = astate.create_remote_table("rdma_qualification", config_factory(1, 0))
    baseline = counters.astate_probe_reg_calls()
    initial_mrs = counters.astate_probe_mr_calls()
    previous = []
    first_regs = first_mrs = None
    # Match the full-model harness, including its initial version -1.
    for index, version in enumerate(range(-1, 3)):
        # Keep previous allocations alive to force different source addresses.
        dense = ((torch.arange(1024 * 1024, device="cuda").reshape(1024, 1024)
                  % 97) + version * 7).to(torch.bfloat16).T
        small = ((torch.arange(17 * 31, device="cuda").reshape(17, 31)
                  % 53) - version * 3).to(torch.float32)
        names = ["large_strided", "small_compact"]
        sources = [dense, small]
        keys = [create_sharded_key(name, list(t.shape), [0, 0])
                for name, t in zip(names, sources)]
        before = time.monotonic()
        if send:
            assert table.multi_put(version, list(zip(keys, sources)))
            # Published snapshots must remain valid after caller storage changes.
            for source in sources:
                source.fill_(-777)
            torch.cuda.synchronize()
            previous.extend(sources)
        else:
            backings = [torch.full((t.shape[1] + 2, t.shape[0]), -999,
                                   dtype=t.dtype, device="cuda") for t in sources]
            targets = [backing[1:-1].T for backing in backings]
            target_addresses = [t.data_ptr() for t in targets]
            assert table.multi_get(version, list(zip(keys, targets)))
            torch.cuda.synchronize()
            for source, target, backing, address in zip(
                    sources, targets, backings, target_addresses):
                assert target.data_ptr() == address
                assert torch.equal(source, target), (version, source.shape)
                assert torch.all(backing[[0, -1]] == -999)
        table.complete(version)
        reg_calls = counters.astate_probe_reg_calls()
        mr_calls = counters.astate_probe_mr_calls()
        if index == 0:
            first_regs, first_mrs = reg_calls, mr_calls
            if send:
                assert reg_calls > baseline, "registration observer did not see first publication"
        else:
            if send:
                assert reg_calls == first_regs, "sender registered parameter slices again"
            if first_mrs > 0:
                assert mr_calls == first_mrs, "new verbs MR registration after initial exchange"
        print(json.dumps(dict(event="ASTATE_PROBE", role=args.role,
                              version=version, verified=True, pid=os.getpid(),
                              seconds=time.monotonic() - before,
                              register_calls=reg_calls, verbs_mr_calls=mr_calls,
                              initial_register_calls=baseline,
                              initial_verbs_mr_calls=initial_mrs,
                              source_addresses=[t.data_ptr() for t in sources],
                              rdma_devices=devices,
                              astate_module=astate.__file__)), flush=True)


if __name__ == "__main__":
    main()

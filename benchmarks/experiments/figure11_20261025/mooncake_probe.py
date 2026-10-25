"""Three-host RDMA check of persistent snapshots with two independent readers."""
import argparse
import json
import os
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['send', 'receive'])
    parser.add_argument('--local-ip', required=True)
    parser.add_argument('--sender-ip', required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    os.environ.update(AWEX_DEVICE_TYPE='cuda', AWEX_MOONCAKE_PROTOCOL='rdma',
                      AWEX_MOONCAKE_HOST=args.local_ip, MC_GID_INDEX='3',
                      MC_HANDSHAKE_MAX_LENGTH='67108864')
    import requests
    import torch
    from shardstream.topology import _active_rdma_endpoints
    from shardstream.integrations.publication.awex_rdma import enable_rdma_audit
    from awex.meta.meta_server import MetaServer, MetaServerClient
    from awex.transfer.mooncake_comm import MooncakeTransport
    from awex.transfer.transfer_plan import TransferPlan, CommunicationOperation

    torch.cuda.set_device(0)
    devices = [endpoint.name for endpoint in _active_rdma_endpoints()]
    assert len(devices) == 4
    os.environ['AWEX_MOONCAKE_DEVICE'] = ','.join(devices)
    enable_rdma_audit()
    if args.role == 'send':
        server = MetaServer(args.local_ip, 30190)
        server.start()
    for _ in range(120):
        try:
            if requests.get(f'http://{args.sender_ip}:30190/v1/health', timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.25)
    else:
        raise TimeoutError('qualification meta server')
    client = MetaServerClient(args.sender_ip, 30190)
    transport = MooncakeTransport(client, args.rank, timeout=120)
    assert transport.registration_policy == 'persistent_snapshots'

    def operations(peer):
        return [CommunicationOperation(send_rank=2, recv_rank=peer,
                send_shard_meta=SimpleNamespace(name=name), recv_shard_meta=SimpleNamespace(name=name),
                send_offset=(0, 0), recv_offset=(0, 0), overlap_shape=shape,
                train_slices=(slice(None), slice(None)), inf_slices=(slice(None), slice(None)))
                for name, shape in [('large', (1024, 1024)), ('small', (17, 31)), ('empty', (0, 3))]]

    plan = TransferPlan(operations=({peer: operations(peer) for peer in [0, 1]}
                                   if args.role == 'send' else {2: operations(args.rank)}))
    previous = []
    first_count = first_addresses = None
    for version in range(-1, 3):
        expected = {
            'large': ((torch.arange(1024 * 1024, device='cuda').reshape(1024, 1024) % 97)
                      + version * 7).to(torch.bfloat16).T,
            'small': ((torch.arange(17 * 31, device='cuda').reshape(17, 31) % 53)
                      - version * 3).to(torch.float32),
            'empty': torch.empty((0, 3), dtype=torch.bfloat16, device='cuda'),
        }
        start = time.monotonic()
        source_addresses = [t.data_ptr() for t in expected.values()]
        if args.role == 'send':
            parameters = dict(expected)
            def poison():
                assert not parameters
                for value in expected.values():
                    value.fill_(-777)
                torch.cuda.synchronize()
            transport.send(parameters, plan, 42, on_ready=poison)
            previous.append(expected)
            for peer in [0, 1]:
                assert client.get_object(f'verified_{version}_{peer}', timeout=120)
        else:
            backing = {name: torch.full((t.shape[1] + 2, t.shape[0]), -999,
                       dtype=t.dtype, device='cuda') for name, t in expected.items()}
            targets = {name: value[1:-1].T for name, value in backing.items()}
            pointers = {name: value.data_ptr() for name, value in targets.items()}
            transport.receive(targets, plan, 42)
            for name, target in targets.items():
                assert target.data_ptr() == pointers[name]
                assert torch.equal(target, expected[name]), (name, version)
                assert torch.all(backing[name][[0, -1]] == -999)
            client.put_object(f'verified_{version}_{args.rank}', True)
        count = transport._shardstream_registration_count
        addresses = sorted(transport.buffers)
        if first_count is None:
            first_count, first_addresses = count, addresses
        assert count == first_count and addresses == first_addresses
        assert count == 2
        print(json.dumps(dict(event='MOONCAKE_PROBE', role=args.role, rank=args.rank,
              version=version, verified=True, register_calls=count,
              registered_addresses=addresses, source_addresses=source_addresses,
              seconds=time.monotonic()-start, rdma_devices=devices)), flush=True)
    transport.close()
    assert not transport.buffers
    print(json.dumps(dict(event='MOONCAKE_PROBE_CLOSED', rank=args.rank, released=True)), flush=True)


if __name__ == '__main__':
    main()

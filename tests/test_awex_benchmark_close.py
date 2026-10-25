"""The harness must close a shared exchange group before trainer teardown."""
from threading import Event
from types import SimpleNamespace

import pytest

from shardstream.integrations.publication import awex_native as module


def test_close_orders_gpu_completion_barrier_and_group_destruction(monkeypatch):
    group = object()
    holder = SimpleNamespace(weights_update_group=group, transfer_rank=7, already_initialized=True)
    events = []
    monkeypatch.setattr(module.torch.cuda, 'synchronize', lambda: events.append('gpu'))
    monkeypatch.setattr(module.dist, 'barrier', lambda **kw: events.append(('barrier', kw['group'])))
    monkeypatch.setattr(module.dist, 'destroy_process_group', lambda g: events.append(('destroy', g)))
    result = module.close_awex_nccl_group(holder, 'send')
    assert events == ['gpu', ('barrier', group), ('destroy', group)]
    assert result['closed'] and result['rank'] == 7
    assert holder.weights_update_group is None and not holder.already_initialized
    assert module.close_awex_nccl_group(holder, 'send')['already_closed']
    assert len(events) == 3


def test_failed_destroy_is_not_reported_or_cleared(monkeypatch):
    group = object()
    holder = SimpleNamespace(weights_update_group=group, transfer_rank=0, already_initialized=True)
    monkeypatch.setattr(module.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(module.dist, 'barrier', lambda **kw: None)
    def fail(g):
        raise RuntimeError('cannot finalize')
    monkeypatch.setattr(module.dist, 'destroy_process_group', fail)
    with pytest.raises(RuntimeError, match='cannot finalize'):
        module.close_awex_nccl_group(holder, 'send')
    assert holder.weights_update_group is group and holder.already_initialized


def test_driver_launches_receiver_close_concurrently_with_trainer_close(monkeypatch):
    remote_started = Event()
    local_started = Event()
    publication = module.AwexNCCLPublication.__new__(module.AwexNCCLPublication)
    publication.harness = SimpleNamespace(is_driver=True,
        megatron_engine=SimpleNamespace(weights_exchange_writer=object()))
    def remote():
        remote_started.set()
        assert local_started.wait(2), 'trainer close never reached the shared collective'
    def local(holder, role):
        local_started.set()
        assert remote_started.wait(2), 'receiver close must run concurrently'
        return {'closed': True}
    publication._request_close = remote
    monkeypatch.setattr(module, 'close_awex_nccl_group', local)
    assert publication.close_training()['closed']


def test_close_requires_results_from_every_dp_core(monkeypatch):
    publication = module.AwexNCCLPublication.__new__(module.AwexNCCLPublication)
    publication.harness = SimpleNamespace(inference_config={'tp_size': 2, 'dp_size': 4})
    publication.timeout_seconds = 120
    monkeypatch.setattr(module, 'publication_endpoints', lambda h: [(0, 'example', 8000)])
    response = SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {'ranks': [{'closed': True}] * 7})
    monkeypatch.setattr(module.requests, 'post', lambda *a, **kw: response)
    with pytest.raises(RuntimeError, match='Incomplete Awex NCCL close'):
        publication._request_close()

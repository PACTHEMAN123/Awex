from types import SimpleNamespace

import pytest

from shardstream.integrations.publication.awex_native import AwexAStatePublication, AwexRDMAPublication
from shardstream.integrations.publication.awex_astate import configure_astate_discovery
from shardstream.integrations.publication.awex_rdma import enable_rdma_audit


@pytest.mark.parametrize("protocol", [None, "tcp"])
def test_rdma_refuses_implicit_or_tcp_protocol(monkeypatch, protocol):
    monkeypatch.delenv("AWEX_MOONCAKE_PROTOCOL", raising=False)
    if protocol is not None:
        monkeypatch.setenv("AWEX_MOONCAKE_PROTOCOL", protocol)
    with pytest.raises(RuntimeError, match="explicit"):
        enable_rdma_audit()


def test_rdma_requests_upstream_mooncake_reader(monkeypatch):
    from shardstream.integrations.publication import awex_native

    requests = []
    monkeypatch.setattr(awex_native.requests, "post", lambda url, **kwargs:
                        requests.append((url, kwargs)) or SimpleNamespace(status_code=200))
    harness = SimpleNamespace(meta_server_addr="host:1234", inference_config={"num_engines": 2})
    publication = AwexRDMAPublication(harness)
    publication._initialize_endpoint((1, "receiver", 4321))
    assert requests[0][0] == "http://receiver:4321/areal_awex_init"
    assert requests[0][1]["json"]["comm_backend"] == "mooncake"
    assert requests[0][1]["json"]["num_engines"] == 2


def test_astate_requests_native_reader_and_reuses_awex_discovery(monkeypatch):
    from shardstream.integrations.publication import awex_native

    requests = []
    monkeypatch.setattr(awex_native.requests, "post", lambda url, **kwargs:
                        requests.append((url, kwargs)) or SimpleNamespace(status_code=200))
    monkeypatch.delenv("UTRANS_DISABLE_RDMA", raising=False)
    monkeypatch.delenv("TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION", raising=False)
    for key in ("DISCOVERY_CONFIG_CENTER_TYPE", "ASYSTEM_META_SERVER_IP", "ASYSTEM_META_SERVER_PORT"):
        monkeypatch.setenv(key, "previous")
    monkeypatch.setenv("TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION", "false")
    harness = SimpleNamespace(meta_server_addr="trainer:1234", inference_config={"num_engines": 2})
    AwexAStatePublication(harness)._initialize_endpoint((1, "receiver", 4321))
    assert requests[0][1]["json"]["comm_backend"] == "astate"
    configure_astate_discovery(harness.meta_server_addr)
    import os
    assert os.environ["DISCOVERY_CONFIG_CENTER_TYPE"] == "HTTP"
    assert os.environ["ASYSTEM_META_SERVER_IP"] == "trainer"
    assert os.environ["ASYSTEM_META_SERVER_PORT"] == "1234"


@pytest.mark.parametrize("variable,value", [("UTRANS_DISABLE_RDMA", "0"),
                                             ("TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION", "true")])
def test_astate_rejects_disabled_rdma_or_ignored_errors(monkeypatch, variable, value):
    monkeypatch.delenv("UTRANS_DISABLE_RDMA", raising=False)
    monkeypatch.delenv("TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION", raising=False)
    monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError):
        configure_astate_discovery("trainer:1234")


def test_astate_selects_active_devices_for_the_current_gpu(monkeypatch):
    import os
    import torch
    from shardstream import topology
    from shardstream.integrations.publication.awex_astate import configure_astate_devices

    monkeypatch.delenv("TRANSFER_ENGINE_MAX_RDMA_DEVICES", raising=False)
    monkeypatch.setenv("TRANSFER_ENGINE_RDMA_DEVICES", "previous")
    endpoints = [SimpleNamespace(name=name, pci_path=str(i))
                 for i, name in enumerate(["mlx5_5", "mlx5_15", "mlx5_20", "mlx5_28"])]
    monkeypatch.setattr(topology, "_active_rdma_endpoints", lambda: endpoints)
    monkeypatch.setattr(topology, "_gpu_hca_topology", lambda endpoints, count:
                        [[0, 3, 4, 4], [4, 4, 2, 0]])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert configure_astate_devices() == ["mlx5_28", "mlx5_20"]
    assert os.environ["TRANSFER_ENGINE_RDMA_DEVICES"] == "mlx5_28,mlx5_20"


def test_astate_refuses_unresolved_gpu_topology(monkeypatch):
    import torch
    from shardstream import topology
    from shardstream.integrations.publication.awex_astate import configure_astate_devices

    monkeypatch.setattr(topology, "_active_rdma_endpoints", lambda: [])
    monkeypatch.setattr(topology, "_gpu_hca_topology", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="Cannot resolve"):
        configure_astate_devices()

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

import pytest

from awex.transfer.nccl_device_v2 import (
    NCCLDeviceV2UnavailableError,
    _resolve_gin_connections,
    _resolve_gin_context_count,
    _resolve_gin_doorbell_batch,
    _resolve_gin_skip_credit_check,
    _resolve_network_channels_per_peer,
    _resolve_network_step_bytes,
)


def test_network_step_defaults_to_nccl_cross_node_chunk(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES", raising=False)
    monkeypatch.delenv("NCCL_P2P_NET_CHUNKSIZE", raising=False)

    assert _resolve_network_step_bytes(None) == 128 * 1024


def test_network_step_honors_nccl_configuration(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES", raising=False)
    monkeypatch.setenv("NCCL_P2P_NET_CHUNKSIZE", str(256 * 1024))

    assert _resolve_network_step_bytes(None) == 256 * 1024


def test_awex_network_step_overrides_nccl_configuration(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES", str(64 * 1024))
    monkeypatch.setenv("NCCL_P2P_NET_CHUNKSIZE", str(256 * 1024))

    assert _resolve_network_step_bytes(None) == 64 * 1024


@pytest.mark.parametrize("value", [0, -16])
def test_network_step_rejects_non_positive_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match="must be positive"):
        _resolve_network_step_bytes(value)


def test_network_step_must_be_vector_aligned():
    with pytest.raises(NCCLDeviceV2UnavailableError, match="multiple of 16"):
        _resolve_network_step_bytes(127)


def test_network_channels_default_to_auto(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_CHANNELS_PER_PEER", raising=False)
    monkeypatch.delenv("NCCL_NCHANNELS_PER_NET_PEER", raising=False)

    assert _resolve_network_channels_per_peer(None) == 0


def test_network_channels_honor_nccl_configuration(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_NET_CHANNELS_PER_PEER", raising=False)
    monkeypatch.setenv("NCCL_NCHANNELS_PER_NET_PEER", "8")

    assert _resolve_network_channels_per_peer(None) == 8


@pytest.mark.parametrize("value", [-1, 65])
def test_network_channels_reject_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[0, 64\]"):
        _resolve_network_channels_per_peer(value)


def test_gin_connections_default_to_all_available_slots(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS", raising=False)
    monkeypatch.delenv("NCCL_GIN_NCONNECTIONS", raising=False)

    assert _resolve_gin_connections(None) == 4


def test_gin_connections_honor_nccl_configuration(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS", raising=False)
    monkeypatch.setenv("NCCL_GIN_NCONNECTIONS", "2")

    assert _resolve_gin_connections(None) == 2


def test_gin_connections_prefer_awex_override(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS", "3")
    monkeypatch.setenv("NCCL_GIN_NCONNECTIONS", "2")

    assert _resolve_gin_connections(None) == 3


@pytest.mark.parametrize("value", [0, 5])
def test_gin_connections_reject_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[1, 4\]"):
        _resolve_gin_connections(value)


def test_gin_contexts_default_to_one_per_connection(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_CONTEXTS", raising=False)

    assert _resolve_gin_context_count(None) == 4


@pytest.mark.parametrize("value", [0, 65])
def test_gin_contexts_reject_out_of_range_values(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[1, 64\]"):
        _resolve_gin_context_count(value)


def test_gin_doorbell_batch_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_DOORBELL_BATCH", raising=False)

    assert _resolve_gin_doorbell_batch(None) == 1


def test_gin_doorbell_batch_honors_environment(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_GIN_DOORBELL_BATCH", "8")

    assert _resolve_gin_doorbell_batch(None) == 8


@pytest.mark.parametrize("value", [0, 9])
def test_gin_doorbell_batch_rejects_values_beyond_fifo(value):
    with pytest.raises(NCCLDeviceV2UnavailableError, match=r"must be in \[1, 8\]"):
        _resolve_gin_doorbell_batch(value)


def test_gin_skip_credit_check_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("AWEX_NCCL_DEVICE_V2_GIN_SKIP_CREDIT_CHECK", raising=False)

    assert not _resolve_gin_skip_credit_check(None)


def test_gin_skip_credit_check_honors_environment(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_GIN_SKIP_CREDIT_CHECK", "1")

    assert _resolve_gin_skip_credit_check(None)


def test_gin_skip_credit_check_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_GIN_SKIP_CREDIT_CHECK", "true")

    with pytest.raises(NCCLDeviceV2UnavailableError, match="must be 0 or 1"):
        _resolve_gin_skip_credit_check(None)

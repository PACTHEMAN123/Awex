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

"""Reject invalid tuning and preserve queue concurrency during connection sweeps."""

import pytest

from shardstream.topology import (
    TransportUnavailableError,
    _resolve_gin_context_count,
)


@pytest.mark.parametrize(
    "connections,channels,expected",
    [(4, 64, 8), (2, 64, 4), (1, 64, 2), (0, 64, 1), (4, 4, 4)],
)
def test_default_queue_count_preserves_existing_transport(
    monkeypatch, connections, channels, expected
):
    monkeypatch.delenv("SHARDSTREAM_GIN_CONTEXTS", raising=False)
    assert _resolve_gin_context_count(connections, channels) == expected


@pytest.mark.parametrize("connections", [1, 2, 4])
def test_explicit_queues_can_hold_concurrency_when_connections_change(
    monkeypatch, connections
):
    monkeypatch.setenv("SHARDSTREAM_GIN_CONTEXTS", "8")
    assert _resolve_gin_context_count(connections, 32) == 8


@pytest.mark.parametrize(
    "value,channels",
    [("0", 64), ("-1", 64), ("65", 64), ("8", 4), ("eight", 64)],
)
def test_invalid_queue_count_fails_before_native_initialization(
    monkeypatch, value, channels
):
    monkeypatch.setenv("SHARDSTREAM_GIN_CONTEXTS", value)
    with pytest.raises(TransportUnavailableError):
        _resolve_gin_context_count(2, channels)

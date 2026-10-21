import pytest

from shardstream_benchmarks.recipe_runner import nccl_buffer_environment


def test_defaults_do_not_override_nccl():
    assert nccl_buffer_environment() == {}


def test_explicit_large_pool_chunk():
    assert nccl_buffer_environment(32 * 2**20, 4 * 2**20) == {
        "NCCL_BUFFSIZE": "33554432",
        "NCCL_P2P_NET_CHUNKSIZE": "4194304",
    }


@pytest.mark.parametrize("buffer", [0, -1, 2**31, 2**32])
def test_reject_invalid_signed_buffer(buffer):
    with pytest.raises(ValueError, match="below 2 GiB"):
        nccl_buffer_environment(buffer)


@pytest.mark.parametrize("chunk", [0, -1, 524289])
def test_reject_invalid_or_silently_clamped_default_chunk(chunk):
    with pytest.raises(ValueError, match="BUFFSIZE / 8"):
        nccl_buffer_environment(net_chunk_bytes=chunk)


def test_preserve_buffer_only_tuning():
    assert nccl_buffer_environment(2**30) == {"NCCL_BUFFSIZE": str(2**30)}
    assert nccl_buffer_environment(net_chunk_bytes=524288) == {
        "NCCL_P2P_NET_CHUNKSIZE": "524288"}

import pytest

from shardstream_benchmarks.recipe_runner import weightbridge_buffer_environment


def test_default_does_not_change_weightbridge():
    assert weightbridge_buffer_environment("weightbridge_native") == {}


@pytest.mark.parametrize("cap", [8, 16, 32, 64, 128])
def test_explicit_native_logical_cap(cap):
    assert weightbridge_buffer_environment("weightbridge_native", cap) == {
        "WBRIDGE_LOGICAL_TENSOR_CAP_BYTES": str(cap * 2**20)}


@pytest.mark.parametrize("cap", [0, -1])
def test_reject_nonpositive_cap(cap):
    with pytest.raises(ValueError, match="positive"):
        weightbridge_buffer_environment("weightbridge_native", cap)


def test_reject_other_backend():
    with pytest.raises(ValueError, match="weightbridge_native"):
        weightbridge_buffer_environment("shardstream", 8)

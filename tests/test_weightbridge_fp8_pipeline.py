from types import SimpleNamespace

import pytest

from shardstream.integrations.publication.weightbridge_fp8_pipeline import (
    RoundPreparation, overlap_pairs, preparation_schedule,
)


def test_scale_first_and_multiple_fragments_quantize_physical_tensor_once():
    schedule = preparation_schedule([
        {'a_scale_inv', 'b::0'}, {'a', 'b::1'}, {'b_scale_inv'},
    ], ['a', 'b'], lambda name: name.split('::')[0])
    assert schedule == [['a', 'b'], [], []]


def test_missing_tensor_is_not_silently_left_stale():
    with pytest.raises(ValueError, match='absent'):
        preparation_schedule([{'a'}], ['a', 'b'], lambda name: name)


def test_overlap_only_counts_earlier_wire_rounds_and_positive_intersections():
    assert overlap_pairs([(1, 20, 50), (3, 80, 90)], {
        0: (10, 30), 1: (20, 45), 2: (60, 80),
    }) == [dict(preparing_round=1, transmitting_round=0, overlap_ms=10 / 1e6)]


def test_sparse_local_rounds_still_prepare_skipped_collectives_once(monkeypatch):
    from shardstream.integrations.publication import weightbridge_fp8_pipeline as module
    events = []
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
    monkeypatch.setattr(module.torch.cuda, 'Event', Event)
    pipeline = RoundPreparation.__new__(RoundPreparation)
    pipeline.schedule = [['a'], ['b'], [], ['c']]
    pipeline.next_round = 0
    pipeline.preparation, pipeline.gpu_intervals = [], []
    pipeline.quantizer = SimpleNamespace(prepare_names=lambda names: events.append(names))
    pipeline.prepare_through(1)
    pipeline.prepare_through(1)
    pipeline.prepare_through(3)
    assert events == [['a'], ['b'], ['c']]
    assert pipeline.next_round == 4

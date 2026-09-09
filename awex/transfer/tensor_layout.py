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

"""Static, copy-only tensor layouts used by device transfer plans."""

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import torch


def _shape_numel(shape: Iterable[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _selected_flat_runs(
    shape: Sequence[int], slices: Sequence[slice]
) -> List[Tuple[int, int]]:
    """Lower a unit-stride rectangular slice to row-major flat runs."""

    shape = tuple(int(dim) for dim in shape)
    slices = tuple(slices)
    if len(slices) != len(shape):
        raise ValueError(
            f"Expected {len(shape)} slices for shape {shape}, got {len(slices)}"
        )

    normalized = []
    for dim, item in zip(shape, slices):
        if not isinstance(item, slice):
            raise ValueError("Static tensor layouts only support slice indices")
        start, stop, step = item.indices(dim)
        if step != 1:
            raise ValueError("Static tensor layouts only support unit-stride slices")
        if stop <= start:
            return []
        normalized.append((start, stop, start == 0 and stop == dim))

    last_partial = -1
    for index, (_, _, is_full) in enumerate(normalized):
        if not is_full:
            last_partial = index
    if last_partial < 0:
        return [(0, _shape_numel(shape))]

    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    strides.reverse()

    prefix_ranges = [
        range(normalized[index][0], normalized[index][1])
        for index in range(last_partial)
    ]
    prefixes = product(*prefix_ranges) if prefix_ranges else [()]
    run_start = normalized[last_partial][0]
    run_length = (normalized[last_partial][1] - run_start) * strides[last_partial]

    runs = []
    for prefix in prefixes:
        flat_start = sum(index * strides[dim] for dim, index in enumerate(prefix))
        flat_start += run_start * strides[last_partial]
        runs.append((flat_start, run_length))
    return runs


def slice_layout_fragments(
    shape: Sequence[int],
    slices: Sequence[slice],
    span_numels: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Return ``(span index, span offset, length)`` for a logical slice."""

    span_numels = tuple(int(numel) for numel in span_numels)
    if any(numel <= 0 for numel in span_numels):
        raise ValueError("Static tensor layout spans must be non-empty")
    expected_numel = _shape_numel(shape)
    if sum(span_numels) != expected_numel:
        raise ValueError(
            "Static tensor layout does not cover its logical shape: "
            f"shape={tuple(shape)} expected={expected_numel} spans={span_numels}"
        )

    span_bounds = []
    start = 0
    for numel in span_numels:
        span_bounds.append((start, start + numel))
        start += numel

    fragments = []
    span_index = 0
    for run_start, run_numel in _selected_flat_runs(shape, slices):
        run_end = run_start + run_numel
        while span_bounds[span_index][1] <= run_start:
            span_index += 1
        current = run_start
        current_span = span_index
        while current < run_end:
            span_start, span_end = span_bounds[current_span]
            fragment_end = min(run_end, span_end)
            fragments.append(
                (current_span, current - span_start, fragment_end - current)
            )
            current = fragment_end
            if current == span_end and current < run_end:
                current_span += 1
        span_index = current_span
    return fragments


@dataclass(frozen=True)
class StaticTensorLayout:
    """A logical row-major tensor assembled from contiguous source views."""

    shape: Tuple[int, ...]
    spans: Tuple[torch.Tensor, ...]

    def __post_init__(self):
        if not self.spans:
            raise ValueError("StaticTensorLayout requires at least one span")
        dtype = self.spans[0].dtype
        device = self.spans[0].device
        for span in self.spans:
            if not span.is_contiguous():
                raise ValueError("StaticTensorLayout spans must be contiguous")
            if span.dtype != dtype or span.device != device:
                raise ValueError("StaticTensorLayout spans must share dtype and device")
        if sum(self.span_numels) != _shape_numel(self.shape):
            raise ValueError(
                "StaticTensorLayout spans do not cover logical shape "
                f"{self.shape}: spans={self.span_numels}"
            )

    @property
    def span_numels(self) -> Tuple[int, ...]:
        return tuple(int(span.numel()) for span in self.spans)

    def slice(self, slices: Sequence[slice]) -> List[torch.Tensor]:
        fragments = slice_layout_fragments(self.shape, slices, self.span_numels)
        return [
            self.spans[span_index].reshape(-1).narrow(0, offset, length)
            for span_index, offset, length in fragments
        ]

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

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class BlockwiseFp8SourceFragment:
    """One rectangular source view in a logical block-quantized matrix."""

    tensor: torch.Tensor
    row_offset: int
    col_offset: int


@dataclass
class BlockwiseFp8State:
    """Shared source and scale storage for one 2-D block-wise FP8 weight."""

    source: torch.Tensor | StaticTensorLayout
    block_shape: Tuple[int, int] = (128, 128)
    scale: torch.Tensor = field(init=False)
    _materialized: Tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if self.block_shape != (128, 128):
            raise ValueError(
                "Only 128 x 128 block-wise FP8 layouts are supported, got "
                f"{self.block_shape}"
            )
        if len(self.shape) != 2:
            raise ValueError(
                f"Block-wise FP8 source must be 2-D, got shape {self.shape}"
            )
        rows, cols = self.shape
        block_rows, block_cols = self.block_shape
        self.scale = torch.empty(
            (ceil_div(rows, block_rows), ceil_div(cols, block_cols)),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        if isinstance(self.source, StaticTensorLayout):
            return tuple(int(dim) for dim in self.source.shape)
        return tuple(int(dim) for dim in self.source.shape)

    @property
    def device(self) -> torch.device:
        if isinstance(self.source, StaticTensorLayout):
            return self.source.spans[0].device
        return self.source.device

    @property
    def source_spans(self) -> Tuple[torch.Tensor, ...]:
        if isinstance(self.source, StaticTensorLayout):
            return self.source.spans
        return (self.source,)

    def source_fragments(
        self, slices: Sequence[slice]
    ) -> List[BlockwiseFp8SourceFragment]:
        """Return block-aligned rectangular views for a logical matrix slice."""

        if len(slices) != 2:
            raise ValueError("Block-wise FP8 layouts require two slice dimensions")
        normalized = []
        for dim, item, block in zip(self.shape, slices, self.block_shape):
            if not isinstance(item, slice):
                raise ValueError("Block-wise FP8 layouts only support slice indices")
            start, stop, step = item.indices(dim)
            if step != 1 or stop <= start:
                raise ValueError(
                    "Block-wise FP8 layouts require non-empty unit-stride slices"
                )
            if start % block or (stop != dim and stop % block):
                raise ValueError(
                    "Block-wise FP8 transfer slices must align to 128-element "
                    f"boundaries: shape={self.shape}, slices={tuple(slices)}"
                )
            normalized.append((start, stop))
        (row_start, row_stop), (col_start, col_stop) = normalized

        if not isinstance(self.source, StaticTensorLayout):
            view = self.source[row_start:row_stop, col_start:col_stop]
            return [BlockwiseFp8SourceFragment(view, row_start, col_start)]

        logical_cols = self.shape[1]
        fragments = []
        span_row_start = 0
        for span in self.source.spans:
            if span.dim() != 2 or int(span.shape[1]) != logical_cols:
                raise ValueError(
                    "Block-wise FP8 static spans must be row-aligned 2-D views"
                )
            span_row_stop = span_row_start + int(span.shape[0])
            begin = max(row_start, span_row_start)
            end = min(row_stop, span_row_stop)
            if begin < end:
                view = span[
                    begin - span_row_start : end - span_row_start,
                    col_start:col_stop,
                ]
                fragments.append(
                    BlockwiseFp8SourceFragment(view, begin, col_start)
                )
            span_row_start = span_row_stop
        if not fragments:
            raise ValueError("Block-wise FP8 slice did not select any source spans")
        return fragments

    def materialize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialize the eager NCCL baseline once for this converted weight."""

        if self._materialized is None:
            if isinstance(self.source, StaticTensorLayout):
                source = torch.cat(self.source.spans, dim=0)
            else:
                source = self.source
            from awex.converter.weights_converter import per_block_cast_to_fp8

            self._materialized = per_block_cast_to_fp8(source, False)
        return self._materialized


@dataclass(frozen=True)
class BlockwiseFp8Layout:
    """Tensor-like logical output backed by a BF16 source and shared scales."""

    state: BlockwiseFp8State
    kind: str
    row_start: int = 0
    row_stop: int | None = None

    def __post_init__(self):
        if self.kind not in {"weight", "scale"}:
            raise ValueError(f"Invalid block-wise FP8 layout kind: {self.kind}")
        stop = self.state.shape[0] if self.row_stop is None else int(self.row_stop)
        if (
            self.row_start < 0
            or stop <= self.row_start
            or stop > self.state.shape[0]
            or self.row_start % self.state.block_shape[0]
            or (stop != self.state.shape[0] and stop % self.state.block_shape[0])
        ):
            raise ValueError(
                "Block-wise FP8 layout rows must be non-empty, in bounds, and "
                f"block aligned: rows=({self.row_start}, {stop}), "
                f"state_shape={self.state.shape}"
            )
        object.__setattr__(self, "row_stop", stop)

    @property
    def shape(self) -> Tuple[int, ...]:
        rows = int(self.row_stop) - self.row_start
        if self.kind == "weight":
            return (rows, self.state.shape[1])
        block_rows = self.state.block_shape[0]
        scale_begin = self.row_start // block_rows
        scale_end = ceil_div(int(self.row_stop), block_rows)
        return (scale_end - scale_begin, self.state.scale.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn if self.kind == "weight" else torch.float32

    @property
    def device(self) -> torch.device:
        return self.state.device

    def numel(self) -> int:
        return _shape_numel(self.shape)

    def is_contiguous(self) -> bool:
        return True

    def source_fragments(
        self, slices: Sequence[slice]
    ) -> List[BlockwiseFp8SourceFragment]:
        if self.kind != "weight":
            raise ValueError("Only FP8 weight layouts expose source fragments")
        if len(slices) != 2:
            raise ValueError("Block-wise FP8 layouts require two slice dimensions")
        row_start, row_stop, row_step = slices[0].indices(self.shape[0])
        if row_step != 1:
            raise ValueError("Block-wise FP8 layouts require unit-stride slices")
        translated = (
            slice(self.row_start + row_start, self.row_start + row_stop),
            slices[1],
        )
        return self.state.source_fragments(translated)

    def scale_view(self, slices: Sequence[slice]) -> torch.Tensor:
        if self.kind != "scale":
            raise ValueError("Only FP8 scale layouts expose scale views")
        if len(slices) != 2:
            raise ValueError("Block-wise FP8 scale layouts require two slices")
        row_start, row_stop, row_step = slices[0].indices(self.shape[0])
        if row_step != 1:
            raise ValueError("Block-wise FP8 scale layouts require unit-stride slices")
        base = self.row_start // self.state.block_shape[0]
        return self.state.scale[
            slice(base + row_start, base + row_stop), slices[1]
        ]

    def materialize(self) -> torch.Tensor:
        weight, scale = self.state.materialize()
        if self.kind == "weight":
            return weight[self.row_start : self.row_stop]
        block_rows = self.state.block_shape[0]
        return scale[
            self.row_start // block_rows : ceil_div(int(self.row_stop), block_rows)
        ]


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def make_blockwise_fp8_layouts(
    source: torch.Tensor | StaticTensorLayout,
) -> Tuple[BlockwiseFp8Layout, BlockwiseFp8Layout]:
    state = BlockwiseFp8State(source=source)
    return (
        BlockwiseFp8Layout(state=state, kind="weight"),
        BlockwiseFp8Layout(state=state, kind="scale"),
    )


def make_blockwise_fp8_row_layouts(
    source: torch.Tensor | StaticTensorLayout,
    row_ranges: Sequence[Tuple[int, int]],
) -> List[Tuple[BlockwiseFp8Layout, BlockwiseFp8Layout]]:
    """Create block-aligned row views that share one quantization state."""

    state = BlockwiseFp8State(source=source)
    return [
        (
            BlockwiseFp8Layout(state, "weight", row_start, row_stop),
            BlockwiseFp8Layout(state, "scale", row_start, row_stop),
        )
        for row_start, row_stop in row_ranges
    ]

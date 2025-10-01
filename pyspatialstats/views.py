"""
This module describes the views into the data to be processed by the focal statistics functions
"""

from dataclasses import dataclass
from itertools import product
from typing import Generator

from pyspatialstats.types.windows import WindowT
from pyspatialstats.windows import Window, define_window


@dataclass
class ArrayView:
    offset: tuple[int, ...]
    shape: tuple[int, ...]

    def __post_init__(self):
        if len(self.offset) != len(self.shape):
            raise ValueError(f'offset and shape must have the same length: {self}')

        if any(x <= 0 for x in self.shape):
            raise ValueError(f'shape must be positive: {self}')

        if any(x < 0 for x in self.offset):
            raise ValueError(f'offset must be non-negative: {self}')

    @property
    def ndim(self):
        return len(self.offset)

    @property
    def slices(self) -> list[slice]:
        return [slice(self.offset[i], self.offset[i] + self.shape[i]) for i in range(self.ndim)]

    def get_external_slices(self, window: Window, reduce: bool) -> list[slice]:
        fringes = window.get_fringes(ndim=2, reduce=reduce)
        return [
            slice(self.offset[i] - fringes[i], self.offset[i] + self.shape[i] + fringes[i]) for i in range(self.ndim)
        ]

    def get_external_shape(self, window: Window, reduce: bool) -> list[int]:
        fringes = window.get_fringes(ndim=2, reduce=reduce)
        return [self.shape[i] + 2 * fringes[i] for i in range(self.ndim)]


@dataclass
class ArrayViewPair:
    input: ArrayView
    output: ArrayView


def define_window_views(
    start: tuple[int, ...],
    stop: tuple[int, ...],
    step: tuple[int, ...],
    window_shape: tuple[int, ...],
) -> Generator[ArrayView, None, None]:
    """No bounds checking"""
    dim_ranges = [range(start[d], stop[d], step[d]) for d in range(len(start))]
    return (ArrayView(offset=offsets, shape=window_shape) for offsets in product(*dim_ranges))


def define_tile_views(
    start: tuple[int, ...],
    stop: tuple[int, ...],
    step: tuple[int, ...],
    tile_shape: tuple[int, ...],
) -> Generator[ArrayView, None, None]:
    """No bounds checking"""
    ndim = len(start)

    def compute_indices(start_d, stop_d, step_d, tile_d):
        indices = []
        i = start_d
        while i < stop_d:
            indices.append(i)
            if i + tile_d >= stop_d:
                break
            i += step_d
        return indices

    dim_indices = [compute_indices(start[d], stop[d], step[d], tile_shape[d]) for d in range(ndim)]

    for offsets in product(*dim_indices):
        shapes = tuple(min(tile_shape[d], stop[d] - offsets[d]) for d in range(len(start)))
        yield ArrayView(offset=offsets, shape=shapes)


def construct_window_views(
    data_shape: tuple[int, ...],
    window: WindowT,
    reduce: bool = False,
) -> Generator[ArrayViewPair, None, None]:
    """Define slices for input and output data for windowed calculations (N dimensions)."""
    ndim = len(data_shape)

    window = define_window(window)
    window.validate(reduce, allow_even=reduce, shape=data_shape)

    window_shape = window.get_shape(ndim=ndim)
    fringes = window.get_fringes(reduce, ndim=ndim)

    step = window_shape if reduce else (1,) * ndim

    stop = (
        tuple(data_shape[i] // window_shape[i] for i in range(ndim))
        if reduce
        else tuple(data_shape[i] - fringes[i] for i in range(ndim))
    )

    input_stop = tuple(data_shape[i] - window_shape[i] + 1 for i in range(ndim))

    input_views = define_window_views(
        start=(0,) * ndim,
        stop=input_stop,
        step=step,
        window_shape=window_shape,
    )

    output_views = define_window_views(
        start=tuple(fringes[i] for i in range(ndim)),
        stop=stop,
        step=(1,) * ndim,
        window_shape=(1,) * ndim,
    )

    return (ArrayViewPair(input=iw, output=ow) for iw, ow in zip(input_views, output_views, strict=True))


def construct_windowed_tile_views(
    data_shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    window: WindowT,
    reduce: bool = False,
) -> list[ArrayViewPair]:
    ndim = len(data_shape)

    if len(data_shape) != len(tile_shape):
        raise ValueError('Data shape is not compatible with tile shape')

    window = define_window(window)
    window.validate(reduce, allow_even=reduce, shape=data_shape)

    window_shape = window.get_shape(ndim=ndim)
    fringes = window.get_fringes(reduce, ndim=ndim)

    if any(window_shape[i] >= tile_shape[i] for i in range(ndim)):
        raise IndexError("Window can't be bigger than the tiles")

    input_step = tile_shape if reduce else [tile_shape[i] - window_shape[i] + 1 for i in range(ndim)]

    input_views = define_tile_views(
        start=(0,) * ndim,
        stop=data_shape,
        step=tuple(input_step),
        tile_shape=tile_shape,
    )

    if reduce:
        output_stop = tuple(data_shape[i] // window_shape[i] for i in range(ndim))
        output_tile_shape = tuple(tile_shape[i] // window_shape[i] for i in range(ndim))
        output_start = (0,) * ndim
    else:
        output_stop = tuple(data_shape[i] - fringes[i] for i in range(ndim))
        output_tile_shape = tuple(tile_shape[i] - 2 * fringes[i] for i in range(ndim))
        output_start = tuple(fringes[i] for i in range(ndim))

    output_views = define_tile_views(
        start=output_start,
        stop=output_stop,
        step=output_tile_shape,
        tile_shape=output_tile_shape,
    )

    pairs = list(ArrayViewPair(input=iw, output=ow) for iw, ow in zip(input_views, output_views, strict=True))

    # Validate window shapes against first and last tiles
    window.validate(reduce, shape=pairs[0].input.shape)
    window.validate(reduce, shape=pairs[-1].input.shape)

    return pairs


def construct_tile_views(data_shape: tuple[int, ...], tile_shape: tuple[int, ...]) -> Generator[ArrayView, None, None]:
    if len(data_shape) != len(tile_shape):
        raise ValueError('Data shape is not compatible with tile shape')

    ndim = len(data_shape)

    return define_tile_views(
        start=(0,) * ndim,
        stop=data_shape,
        step=tile_shape,
        tile_shape=tile_shape,
    )

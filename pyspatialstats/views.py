"""
This module describes the views into the data to be processed by the focal statistics functions
"""

from dataclasses import dataclass
from typing import Generator

from pyspatialstats.types.windows import WindowT
from pyspatialstats.windows import Window, define_window


@dataclass
class RasterView:
    col_off: int
    row_off: int
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f'width and height must be positive: {self}')

        if self.col_off < 0 or self.row_off < 0:
            raise ValueError(f'col_off and row_off must be non-negative: {self}')

    @property
    def slices(self) -> tuple[slice, slice]:
        return (
            slice(self.row_off, self.row_off + self.height),
            slice(self.col_off, self.col_off + self.width),
        )

    def get_external_slices(self, window: Window, reduce: bool) -> tuple[slice, slice]:
        fringes = window.get_fringes(ndim=2, reduce=reduce)
        return (
            slice(self.row_off - fringes[0], self.row_off + self.height + fringes[0]),
            slice(self.col_off - fringes[1], self.col_off + self.width + fringes[1]),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def get_external_shape(self, window: Window, reduce: bool) -> tuple[int, int]:
        fringes = window.get_fringes(ndim=2, reduce=reduce)
        return self.height + 2 * fringes[0], self.width + 2 * fringes[1]


@dataclass
class RasterViewPair:
    input: RasterView
    output: RasterView


def define_window_views(
    start: tuple[int, int],
    stop: tuple[int, int],
    step: tuple[int, int],
    window_shape: tuple[int, int],
) -> Generator[RasterView, None, None]:
    """No bounds checking"""
    return (
        RasterView(
            col_off=x_idx,
            row_off=y_idx,
            width=window_shape[1],
            height=window_shape[0],
        )
        for y_idx in range(start[0], stop[0], step[0])
        for x_idx in range(start[1], stop[1], step[1])
    )


def define_tile_views(
    start: tuple[int, int],
    stop: tuple[int, int],
    step: tuple[int, int],
    tile_shape: tuple[int, int],
) -> Generator[RasterView, None, None]:
    """No bounds checking"""
    height = 0
    for y_idx in range(start[0], stop[0], step[0]):
        for x_idx in range(start[1], stop[1], step[1]):
            width = min(stop[1] - x_idx, tile_shape[1])
            height = min(stop[0] - y_idx, tile_shape[0])
            yield RasterView(
                col_off=x_idx,
                row_off=y_idx,
                width=width,
                height=height,
            )
            if width + x_idx == stop[1]:
                break
        if height + y_idx == stop[0]:
            break


def construct_window_views(
    raster_shape: tuple[int, int],
    window: WindowT,
    reduce: bool = False,
) -> Generator[RasterViewPair, None, None]:
    """define slices for input and output data for windowed calculations"""
    window = define_window(window)
    window.validate(reduce, allow_even=reduce, shape=raster_shape)
    window_shape = window.get_raster_shape()
    fringes = window.get_fringes(reduce)

    step = window_shape if reduce else (1, 1)
    stop = (
        (
            raster_shape[0] // window_shape[0],
            raster_shape[1] // window_shape[1],
        )
        if reduce
        else (
            raster_shape[0] - fringes[0],
            raster_shape[1] - fringes[1],
        )
    )

    input_views = define_window_views(
        start=(0, 0),
        stop=(
            raster_shape[0] - window_shape[0] + 1,
            raster_shape[1] - window_shape[1] + 1,
        ),
        step=step,
        window_shape=window_shape,
    )
    output_views = define_window_views(
        start=(fringes[0], fringes[1]),
        stop=stop,
        step=(1, 1),
        window_shape=(1, 1),
    )

    return (RasterViewPair(input=iw, output=ow) for iw, ow in zip(input_views, output_views, strict=True))


def construct_tile_views(
    raster_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    window: WindowT,
    reduce: bool = False,
) -> tuple[RasterViewPair, ...]:
    """define slices for input and output data for tiled and windowed calculations"""
    window = define_window(window)
    window.validate(reduce, allow_even=reduce, shape=raster_shape)
    window_shape = window.get_raster_shape()
    fringes = window.get_fringes(reduce)

    if window_shape[0] >= tile_shape[0] or window_shape[1] >= tile_shape[1]:
        raise IndexError("Window can't be bigger than the tiles")

    input_step = tile_shape if reduce else (tile_shape[0] - window_shape[0] + 1, tile_shape[1] - window_shape[1] + 1)

    input_views = define_tile_views(
        start=(0, 0),
        stop=raster_shape,
        step=input_step,
        tile_shape=tile_shape,
    )

    output_stop = (
        (raster_shape[0] // window_shape[0], raster_shape[1] // window_shape[1])
        if reduce
        else (raster_shape[0] - fringes[0], raster_shape[1] - fringes[1])
    )

    output_tile_shape = (
        (tile_shape[0] // window_shape[0], tile_shape[1] // window_shape[1])
        if reduce
        else (tile_shape[0] - 2 * fringes[0], tile_shape[1] - 2 * fringes[1])
    )

    output_views = define_tile_views(
        start=(fringes[0], fringes[1]),
        stop=output_stop,
        step=output_tile_shape,
        tile_shape=output_tile_shape,
    )

    pairs = tuple(RasterViewPair(input=iw, output=ow) for iw, ow in zip(input_views, output_views, strict=True))

    window.validate(reduce, shape=pairs[0].input.shape)
    window.validate(reduce, shape=pairs[-1].input.shape)

    return pairs

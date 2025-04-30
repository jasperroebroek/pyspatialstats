"""
This module describes the views into the data to be processed by the focal statistics functions
"""

from typing import Callable, Generator, List, Tuple

import numpy as np
from pydantic import BaseModel, validate_call

from pyspatialstats.types import PositiveInt, Shape2D, UInt

# ViewFunction arguments correspond to keys in the `input` dictionary, and should return another dict with the keys
# corresponding to the `output` dictionary of the `focal_function` function.
ViewFunction = Callable[[List[np.ndarray]], List[float]]


class RasterView(BaseModel):
    col_off: UInt
    row_off: UInt
    width: PositiveInt
    height: PositiveInt

    @property
    def slices(self) -> tuple[slice, slice]:
        return (
            slice(self.row_off, self.row_off + self.height),
            slice(self.col_off, self.col_off + self.width),
        )


class RasterViewPair(BaseModel):
    input: RasterView
    output: RasterView


def define_views(
    raster_shape: Tuple[int, int],
    view_shape: Tuple[int, int],
    fringe: Tuple[int, int],
    step: Tuple[int, int],
) -> Generator[RasterView, None, None]:
    return (
        RasterView(
            col_off=x_idx, row_off=y_idx, width=view_shape[1], height=view_shape[0]
        )
        for y_idx in range(
            fringe[0], raster_shape[0] - view_shape[0] - fringe[0] + 1, step[0]
        )
        for x_idx in range(
            fringe[1], raster_shape[1] - view_shape[1] - fringe[1] + 1, step[1]
        )
    )


def define_tiles(
    raster_shape: Shape2D, tile_shape: Shape2D, fringe: Tuple[PositiveInt, PositiveInt]
) -> Generator[RasterView, None, None]:
    return (
        RasterView(
            col_off=x_idx - fringe[1],
            row_off=y_idx - fringe[0],
            width=tile_shape[1] + fringe[1] * 2,
            hight=tile_shape[0] + fringe[0] * 2,
        )
        for y_idx in range(0, raster_shape[0], tile_shape[0])
        for x_idx in range(0, raster_shape[1], tile_shape[1])
    )


@validate_call
def construct_views(
    raster_shape: Shape2D,
    view_shape: Shape2D,
    reduce: bool = False,
) -> Generator[RasterViewPair, None, None]:
    """define slices for input and output data for windowed calculations"""
    if reduce:
        output_shape = (
            raster_shape[0] // view_shape[0],
            raster_shape[1] // view_shape[1],
        )
        output_fringe = (0, 0)
        input_step = view_shape
    else:
        output_shape = raster_shape
        output_fringe = (view_shape[0] // 2, view_shape[1] // 2)
        input_step = (1, 1)

    input_views = define_views(
        raster_shape, view_shape=view_shape, fringe=(0, 0), step=input_step
    )
    output_views = define_views(
        output_shape, view_shape=(1, 1), fringe=output_fringe, step=(1, 1)
    )

    return (
        RasterViewPair(input=iw, output=ow) for iw, ow in zip(input_views, output_views)
    )


@validate_call
def construct_tiles(
    raster_shape: Shape2D,
    tile_shape: Shape2D,
    view_shape: Shape2D,
    reduce: bool = False,
) -> Generator[RasterViewPair, None, None]:
    """define slices for input and output data for tiled and windowed calculations"""
    if reduce:
        fringe = (0, 0)
        output_shape = (
            raster_shape[0] // view_shape[0],
            raster_shape[1] // view_shape[1],
        )
        output_tile_shape = (
            tile_shape[0] // view_shape[0],
            tile_shape[1] // view_shape[1],
        )
    else:
        fringe = (view_shape[0] // 2, view_shape[1] // 2)
        output_shape = raster_shape
        output_tile_shape = tile_shape

    input_views = define_tiles(
        raster_shape,
        tile_shape=tile_shape,
        fringe=fringe,
    )
    output_views = define_tiles(
        output_shape,
        tile_shape=output_tile_shape,
        fringe=(0, 0),
    )

    return (
        RasterViewPair(input=iw, output=ow) for iw, ow in zip(input_views, output_views)
    )

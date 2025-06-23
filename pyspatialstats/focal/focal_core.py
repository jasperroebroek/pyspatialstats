from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from pyspatialstats.focal.result_config import FocalArrayResultConfig, FocalResultConfig
from pyspatialstats.types.results import FocalStatResults, StatResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.types.arrays import Array, RasterFloat64
from pyspatialstats.types.functions import FocalStatsFunction
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import parse_raster
from pyspatialstats.views import RasterViewPair, construct_tile_views
from pyspatialstats.windows import Window, define_window


def focal_stats_base(
    *args: ArrayLike,
    cy_func: Callable,
    window: Window,
    fraction_accepted: float,
    reduce: bool,
    result_config: FocalResultConfig = FocalArrayResultConfig(),
    out: Optional[StatResult | Array] = None,
    **kwargs,
) -> StatResult | RasterFloat64:
    args = [parse_raster(arg) for arg in args]

    for arg in args:
        if arg.shape != args[0].shape:
            raise ValueError('All input rasters must have the same shape')

    mask = window.get_mask(2)
    fringe = window.get_fringes(reduce, ndim=2)
    threshold = window.get_threshold(fraction_accepted=fraction_accepted)

    out = result_config.parse_output(a=args[0], out=out, window=window, reduce=reduce)
    cy_results = result_config.get_cython_input(args[0], window, reduce, out)
    args_windowed = [rolling_window(arg, window=window.get_raster_shape(), reduce=reduce) for arg in args]

    cy_func(
        *args_windowed,
        mask=mask,
        fringe=np.asarray(fringe, dtype=np.int32),
        threshold=threshold,
        reduce=reduce,
        **kwargs,
        **cy_results,
    )

    return result_config.write_output(window=window, reduce=reduce, out=out, cy_result=cy_results)


def focal_stats_parallel_tile(
    *args: Array,
    func: FocalStatsFunction,
    window: Window,
    reduce: bool,
    fraction_accepted: float,
    out: StatResult | Array,
    result_config: FocalResultConfig,
    tile_view: RasterViewPair,
) -> None:
    tile_out = result_config.create_tile_output(out, tile_view, window, reduce)
    result = func(
        *(arg[*tile_view.input.slices] for arg in args),
        window=window,
        reduce=reduce,
        fraction_accepted=fraction_accepted,
        out=tile_out,
        result_config=result_config,
    )
    result_config.write_tile_output(window, reduce, out, result, tile_view)


def focal_stats_parallel(
    *args: Array,
    func: FocalStatsFunction,
    window: Window,
    reduce: bool,
    fraction_accepted: float,
    tile_shape: tuple[int, int],
    result_config: FocalResultConfig,
    out: Optional[StatResult | Array],
) -> StatResult | RasterFloat64:
    out = result_config.parse_output(a=args[0], out=out, window=window, reduce=reduce)

    shape = args[0].shape[0], args[0].shape[1]
    tile_views = construct_tile_views(shape, tile_shape, window, reduce)

    Parallel(prefer='threads', mmap_mode='r+')(
        delayed(focal_stats_parallel_tile)(
            *args,
            func=func,
            window=window,
            reduce=reduce,
            out=out,
            result_config=result_config,
            tile_view=tile_view,
            fraction_accepted=fraction_accepted,
        )
        for tile_view in tile_views
    )

    return out


def focal_stats(
    *args: Array,
    func: FocalStatsFunction,
    window: WindowT,
    fraction_accepted: float,
    reduce: bool,
    result_config: FocalResultConfig = FocalArrayResultConfig(),
    chunks: Optional[int | tuple[int, int]] = None,
    out: Optional[FocalStatResults] = None,
) -> StatResult | RasterFloat64:
    window = define_window(window)
    window.validate(reduce, allow_even=False, a=args[0])

    if chunks is None:
        return func(
            *args,
            window=window,
            reduce=reduce,
            fraction_accepted=fraction_accepted,
            result_config=result_config,
            out=out,
        )
    else:
        tile_shape = chunks if isinstance(chunks, tuple) else (chunks, chunks)

        return focal_stats_parallel(
            *args,
            func=func,
            window=window,
            reduce=reduce,
            fraction_accepted=fraction_accepted,
            tile_shape=tile_shape,
            result_config=result_config,
            out=out,
        )

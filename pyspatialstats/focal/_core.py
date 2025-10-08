from typing import Callable, Dict, Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from pyspatialstats.focal.result_config import FocalArrayResultConfig, FocalResultConfig
from pyspatialstats.results.stats import FocalStatResult, StatResult
from pyspatialstats.types.arrays import Array, RasterFloat64
from pyspatialstats.types.functions import FocalStatsFunction
from pyspatialstats.types.windows import WindowT
from pyspatialstats.views import ArrayViewPair, construct_windowed_tile_views
from pyspatialstats.windows import Window, define_window


def focal_stats_base(
    data: Dict[str, ArrayLike],
    stat_func: Callable,
    window: Window,
    fraction_accepted: float,
    reduce: bool,
    result_config: FocalResultConfig = FocalArrayResultConfig(),
    out: Optional[StatResult | Array] = None,
    **kwargs,
) -> StatResult | RasterFloat64:
    parsed_data = {name: result_config.parse_array(name, raster) for name, raster in data.items()}

    first_key, first_raster = next(iter(parsed_data.items()))
    raster_shape = first_raster.shape[:2]

    for k, a in parsed_data.items():
        if a.shape[:2] != raster_shape:
            raise ValueError(
                f'All input rasters must have the same shape: {k}->{a.shape[:2]} {first_key}->{raster_shape}'
            )

    mask = window.get_mask(2)
    fringe = window.get_fringes(reduce, ndim=2)
    threshold = window.get_threshold(fraction_accepted=fraction_accepted)

    out = result_config.parse_output(raster_shape=raster_shape, out=out, window=window, reduce=reduce)
    result_arrays = result_config.get_cython_input(raster_shape=raster_shape, window=window, reduce=reduce, out=out)

    data_windowed = result_config.window_data(parsed_data, window=window, reduce=reduce)

    # Debugging
    # print("data_parsed")
    # for k, v in data_windowed.items():
    #     print(k, v.shape, v.dtype)
    # print('result_arrays')
    # for k, v in result_arrays.items():
    #     print(k, v.shape, v.dtype)

    stat_func(
        **data_windowed,
        mask=mask,
        fringe=np.asarray(fringe, dtype=np.int32),
        threshold=threshold,
        reduce=reduce,
        **kwargs,
        **result_arrays,
    )

    return result_config.write_output(window=window, reduce=reduce, out=out, cy_result=result_arrays)


def focal_stats_parallel_tile(
    data: Dict[str, Array],
    func: FocalStatsFunction,
    window: Window,
    reduce: bool,
    fraction_accepted: float,
    out: StatResult | Array,
    result_config: FocalResultConfig,
    tile_view: ArrayViewPair,
) -> None:
    tile_out = result_config.create_tile_output(out, tile_view, window, reduce)
    tile_data = {name: arg[tuple(tile_view.input.slices)] for name, arg in data.items()}

    func(
        data=tile_data,
        window=window,
        reduce=reduce,
        fraction_accepted=fraction_accepted,
        out=tile_out,
        result_config=result_config,
    )


def focal_stats_parallel(
    data: Dict[str, Array],
    func: FocalStatsFunction,
    window: Window,
    reduce: bool,
    fraction_accepted: float,
    tile_shape: tuple[int, int],
    result_config: FocalResultConfig,
    out: Optional[StatResult | Array],
) -> StatResult | RasterFloat64:
    name, arg = next(iter(data.items()))

    shape = arg.shape[0], arg.shape[1]
    out = result_config.parse_output(raster_shape=shape, out=out, window=window, reduce=reduce)
    tile_views = construct_windowed_tile_views(shape, tile_shape, window, reduce)

    Parallel(prefer='threads', mmap_mode='r+')(
        delayed(focal_stats_parallel_tile)(
            data=data,
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
    data: Dict[str, Array],
    func: FocalStatsFunction,
    window: WindowT,
    fraction_accepted: float,
    reduce: bool,
    result_config: FocalResultConfig = FocalArrayResultConfig(),
    chunks: Optional[int | tuple[int, int]] = None,
    out: Optional[FocalStatResult] = None,
) -> StatResult | RasterFloat64:
    window = define_window(window)
    window.validate(reduce, allow_even=False, shape=next(iter(data.values())).shape[:2])

    if chunks is None:
        return func(
            data=data,
            window=window,
            reduce=reduce,
            fraction_accepted=fraction_accepted,
            result_config=result_config,
            out=out,
        )
    else:
        tile_shape = chunks if isinstance(chunks, tuple) else (chunks, chunks)

        return focal_stats_parallel(
            data=data,
            func=func,
            window=window,
            reduce=reduce,
            fraction_accepted=fraction_accepted,
            tile_shape=tile_shape,
            result_config=result_config,
            out=out,
        )

import time
from functools import wraps
from typing import Tuple

import numpy as np
from numpy.typing import DTypeLike
from numpydantic.ndarray import NDArray

from pyspatialstats.types import Fraction, RasterBool, RasterFloat64, RasterWindowShape
from pyspatialstats.windows import Window


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"{func.__name__}{args}{kwargs} Took {total_time:.4f} seconds")

        return result

    return timeit_wrapper


def parse_raster(a: NDArray) -> RasterFloat64:
    """Convert to 2D array with dtype float64"""
    a_parsed = np.asarray(a, dtype=np.float64)
    if a_parsed.ndim != 2:
        raise IndexError("Only 2D data is supported")
    return a_parsed


def parse_nans(
    a: RasterFloat64, dtype_original: DTypeLike
) -> Tuple[bool, bool, RasterBool]:
    if not np.issubdtype(dtype_original, np.floating):
        return False, False, np.zeros(a.shape, dtype=np.bool_)

    nan_mask = np.isnan(a)
    empty_flag = (~nan_mask).sum() == 0
    nan_flag = nan_mask.sum() > 0

    return empty_flag, nan_flag, nan_mask


def create_output_array(
    a: RasterFloat64, window_shape: RasterWindowShape, reduce: bool
) -> RasterFloat64:
    shape = list(np.asarray(a.shape) // window_shape) if reduce else a.shape
    return np.full(shape, dtype=np.float64, fill_value=np.nan)


def calc_count_values(
    window: Window, nan_mask: RasterBool, reduce: bool, ind_inner: Tuple[slice, slice]
) -> RasterFloat64:
    if window.masked:
        from pyspatialstats.focal_stats.focal_statistics import focal_sum

        count_values = np.asarray(
            focal_sum(~nan_mask, window=window, reduce=reduce, fraction_accepted=1)
        )[ind_inner]

    else:
        from pyspatialstats.rolling.rolling_stats import rolling_sum

        count_values = rolling_sum(~nan_mask, window=window, reduce=reduce)

    if not reduce:
        count_values[nan_mask[ind_inner]] = 0

    return count_values


def calc_below_fraction_accepted_mask(
    window: Window,
    nan_mask: RasterBool,
    ind_inner: Tuple[slice, slice],
    fraction_accepted: Fraction,
    reduce: bool,
) -> RasterBool:
    threshold = fraction_accepted * window.get_mask(2).sum()
    count_values = calc_count_values(window, nan_mask, reduce, ind_inner)

    return count_values < threshold

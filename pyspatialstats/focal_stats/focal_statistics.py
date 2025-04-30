from enum import IntEnum
from typing import Literal, Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import validate_call

from pyspatialstats.focal_stats.core.stats import (
    _focal_majority,
    _focal_max,
    _focal_min,
    _focal_std,
    _focal_sum,
)
from pyspatialstats.rolling import rolling_sum, rolling_window
from pyspatialstats.types import Fraction, Mask, PositiveInt, RasterFloat64
from pyspatialstats.utils import (
    calc_below_fraction_accepted_mask,
    calc_count_values,
    create_output_array,
    parse_nans,
    parse_raster,
    timeit,
)
from pyspatialstats.windows import (
    Window,
    define_ind_inner,
    define_window,
    validate_window,
)


class MajorityMode(IntEnum):
    ascending = 0
    descending = 1
    nan = 2


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_min(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
) -> RasterFloat64:
    """
    Focal minimum

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        
        * ``0``: all windows are calculated if at least 1 value is present
        * ``1``: only windows completely filled with values are calculated
        * ``0-1``: fraction of acceptability
        
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same raster_shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The raster_shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviors might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_shape = 5
                >>> fringe = window_shape // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The raster_shape
            will be the original raster_shape divided by the ``window_shape``. Dimensions
            remain equal. No border of NaN values is present.
    """
    dtype_original = a.dtype
    a = parse_raster(a)
    empty_flag, nan_flag, nan_mask = parse_nans(a, dtype_original)

    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce, allow_even=False)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    ind_inner = define_ind_inner(window, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return create_output_array(a, window_shape, reduce)

    if not nan_flag and not window.masked:
        r = create_output_array(a, window_shape, reduce)

        r[ind_inner] = rolling_window(a, window=window, reduce=reduce).min(axis=(2, 3))

        return r

    return np.asarray(_focal_min(a, window_shape, mask, fraction_accepted, reduce))


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_max(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
) -> RasterFloat64:
    """
    Focal maximum

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        
        * ``0``: all windows are calculated if at least 1 value is present
        * ``1``: only windows completely filled with values are calculated
        * ``0-1``: fraction of acceptability
        
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same raster_shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The raster_shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviors might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_shape = 5
                >>> fringe = window_shape // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The raster_shape
            will be the original raster_shape divided by the ``window_shape``. Dimensions
            remain equal. No border of NaN values is present.
    """
    dtype_original = a.dtype
    a = parse_raster(a)
    empty_flag, nan_flag, nan_mask = parse_nans(a, dtype_original)

    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce, allow_even=False)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    ind_inner = define_ind_inner(window, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return create_output_array(a, window_shape, reduce)

    if not nan_flag and not window.masked:
        r = create_output_array(a, window_shape, reduce)

        r[ind_inner] = rolling_window(a, window=window, reduce=reduce).max(axis=(2, 3))

        return r

    return np.asarray(_focal_max(a, window_shape, mask, fraction_accepted, reduce))


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_mean(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
) -> RasterFloat64:
    """
    Focal mean

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    mask : array_like, optional
        A boolean array (2D). If provided, its raster_shape will be used as ``window_shape``, and its entries are used to mask
        every window.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        
        * ``0``: all windows are calculated if at least 1 value is present
        * ``1``: only windows completely filled with values are calculated
        * ``0-1``: fraction of acceptability
        
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same raster_shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The raster_shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_shape = 5
                >>> fringe = window_shape // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The raster_shape
            will be the original raster_shape divided by the ``window_shape``. Dimensions
            remain equal. No border of NaN values is present.
    """
    dtype_original = a.dtype
    a = parse_raster(a)
    empty_flag, nan_flag, nan_mask = parse_nans(a, dtype_original)

    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce, allow_even=False)

    ind_inner = define_ind_inner(window, reduce)

    r = focal_sum(
        a,
        window=window,
        fraction_accepted=fraction_accepted,
        verbose=verbose,
        reduce=reduce,
    )

    count_values = calc_count_values(window, nan_mask, reduce, ind_inner)

    r[ind_inner] = r[ind_inner] / count_values
    return r


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_std(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
    std_df: Literal[0, 1] = 0,
) -> RasterFloat64:
    """
    Focal standard deviation

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        
        * ``0``: all windows are calculated if at least 1 value is present
        * ``1``: only windows completely filled with values are calculated
        * ``0-1``: fraction of acceptability
        
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same raster_shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.
    std_df : {1,0}, optional
        Degrees of freedom; meaning if the function is divided by the count of observations or the count of observations
        minus one. Should be 0 or 1. See :func:`numpy.std` for documentation.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The raster_shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_shape = 5
                >>> fringe = window_shape // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The raster_shape
            will be the original raster_shape divided by the ``window_shape``. Dimensions
            remain equal. No border of NaN values is present.
    """
    dtype_original = a.dtype
    a = parse_raster(a)
    empty_flag, nan_flag, nan_mask = parse_nans(a, dtype_original)

    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce, allow_even=False)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return create_output_array(a, window_shape, reduce)

    return np.asarray(
        _focal_std(a, window_shape, mask, fraction_accepted, reduce, std_df)
    )


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_sum(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
) -> RasterFloat64:
    """
    Focal summation

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        
        * ``0``: all windows are calculated if at least 1 value is present
        * ``1``: only windows completely filled with values are calculated
        * ``0-1``: fraction of acceptability
        
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same raster_shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The raster_shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_shape = 5
                >>> fringe = window_shape // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The raster_shape
            will be the original raster_shape divided by the ``window_shape``. Dimensions
            remain equal. No border of NaN values is present.
    """
    dtype_original = a.dtype
    a = parse_raster(a)
    empty_flag, nan_flag, nan_mask = parse_nans(a, dtype_original)

    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce, allow_even=False)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    ind_inner = define_ind_inner(window, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return create_output_array(a, window_shape, reduce)

    if not window.masked:
        r = create_output_array(a, window_shape, reduce)
        a_parsed = a.copy()
        a_parsed[nan_mask] = 0

        below_fraction_accepted_mask = calc_below_fraction_accepted_mask(
            window, nan_mask, ind_inner, fraction_accepted, reduce
        )
        r[ind_inner] = rolling_sum(a_parsed, window=window, reduce=reduce)
        r[ind_inner][below_fraction_accepted_mask] = np.nan
        return r

    return np.asarray(_focal_sum(a, window_shape, mask, fraction_accepted, reduce))


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_majority(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
    majority_mode: Literal["nan", "ascending", "descending"] = "nan",
) -> RasterFloat64:
    """
    Focal majority

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        
        * ``0``: all windows are calculated if at least 1 value is present
        * ``1``: only windows completely filled with values are calculated
        * ``0-1``: fraction of acceptability
        
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same raster_shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.
    majority_mode : {"nan", "ascending", "descending"}, optional
        Differt modes of dealing with more than one value occuring equally often:

            - ``nan``: when more than one class has the same score NaN will be assigned
            - ``ascending``: the first occurrence of the maximum count will be assigned
            - ``descending``: the last occurrence of the maximum count will be assigned

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The raster_shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_shape = 5
                >>> fringe = window_shape // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The raster_shape
            will be the original raster_shape divided by the ``window_shape``. Dimensions
            remain equal. No border of NaN values is present.
    """
    dtype_original = a.dtype
    a = parse_raster(a)
    empty_flag, nan_flag, nan_mask = parse_nans(a, dtype_original)

    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce, allow_even=False)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return create_output_array(a, window_shape, reduce)

    return np.asarray(
        _focal_majority(
            a,
            window_shape,
            mask,
            fraction_accepted,
            reduce,
            MajorityMode[majority_mode],
        )
    )

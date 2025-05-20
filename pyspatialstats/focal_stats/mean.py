from typing import Sequence

from numpy.typing import ArrayLike
from pydantic import validate_call

from pyspatialstats.focal_stats.core.mean import _focal_mean, _focal_mean_bootstrap
from pyspatialstats.results import BootstrapMeanResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.types import Fraction, Mask, PositiveInt, RasterFloat64
from pyspatialstats.utils import create_output_array, parse_raster, timeit
from pyspatialstats.windows import (
    Window,
    define_window,
    validate_window,
)


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_mean(
    a: ArrayLike,
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
    a = parse_raster(a)

    window = define_window(window)
    validate_window(window, a.shape, reduce, allow_even=False)
    mask = window.get_mask(2)

    fringe = window.get_fringes(reduce)
    ind_inner = window.get_ind_inner(reduce)
    threshold = fraction_accepted * mask.sum()

    r = create_output_array(a, window.get_shape(), reduce)
    a_windowed = rolling_window(a, window=window, reduce=reduce)

    _focal_mean(
        a=a_windowed,
        mask=window.get_mask(2),
        r=r[ind_inner],
        fringe=fringe,
        threshold=threshold,
        reduce=reduce,
    )

    return r


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_mean_bootstrap(
    a: ArrayLike,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
    n_bootstraps: int = 1000,
    seed: int = 0,
) -> RasterFloat64:
    """
    Bootstrapped focal mean

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

    a = parse_raster(a)

    window = define_window(window)
    validate_window(window, a.shape, reduce, allow_even=False)
    mask = window.get_mask(2)

    fringe = window.get_fringes(reduce)
    ind_inner = window.get_ind_inner(reduce)
    threshold = fraction_accepted * mask.sum()

    mean = create_output_array(a, window.get_shape(2), reduce)
    se = create_output_array(a, window.get_shape(2), reduce)
    a_windowed = rolling_window(a, window=window, reduce=reduce)

    _focal_mean_bootstrap(
        a_windowed,
        window.get_mask(2),
        mean[ind_inner],
        se[ind_inner],
        fringe,
        threshold,
        reduce,
        n_bootstraps,
        seed,
    )

    return BootstrapMeanResult(mean=mean, se=se)

from typing import Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import validate_call

from pyspatialstats.focal_stats.core.correlation import (
    CyFocalCorrelationResult,
    _focal_correlation,
)
from pyspatialstats.results import CorrelationResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.stat_utils import calculate_p_value
from pyspatialstats.types import Fraction, Mask, PositiveInt
from pyspatialstats.utils import create_output_array, parse_raster, timeit
from pyspatialstats.windows import Window, define_window, validate_window


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_correlation(
    a1: NDArray,
    a2: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window = 5,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    p_values: bool = False,
) -> CorrelationResult:
    """
    Focal correlation

    Parameters
    ----------
    a, b : array-like
        Input arrays that will be correlated. If not present in dtype :obj:`~numpy.float64` it will be converted
        internally. They need to have the same shape and have two dimensions.
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a custom ``Window`` object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable

        * ``0``: all views are calculated if at least 1 value is present
        * ``1``: only views completely filled with values are calculated
        * ``0-1``: fraction of acceptability

    reduce : bool, optional
        Reuse all cells exactly once by setting a stepsize of the same size as window_shape. The resulting raster will
        have the raster_shape: ``raster_shape/window_shape``
    verbose : bool, optional
        Verbosity with timing. False by default

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the local correlation. If ``reduce`` is set to False, the output has the same raster_shape as the
        input raster, while if ``reduce`` is True, the output is reduced by the window size:
        ``raster_shape // window_shape``.
    """
    a1 = parse_raster(a1)
    a2 = parse_raster(a2)

    if a1.shape != a2.shape:
        raise ValueError(
            f"Input arrays have different shapes: {a1.shape=}, {a2.shape=}"
        )

    window = define_window(window)
    validate_window(window, a1.shape, reduce, allow_even=False)
    mask = window.get_mask(2)

    fringes = window.get_fringes(reduce)
    ind_inner = window.get_ind_inner(reduce)
    threshold = fraction_accepted * mask.sum()

    df = create_output_array(a1, window.get_shape(), reduce, dtype=np.uintp)
    c = create_output_array(a1, window.get_shape(), reduce)

    r = CyFocalCorrelationResult(df=df[ind_inner], c=c[ind_inner])

    a1_windowed = rolling_window(a1, window=window.get_shape(), reduce=reduce)
    a2_windowed = rolling_window(a2, window=window.get_shape(), reduce=reduce)

    _focal_correlation(
        a1_windowed,
        a2_windowed,
        mask=window.get_mask(),
        r=r,
        fringe=fringes,
        threshold=threshold,
        reduce=reduce,
    )

    if p_values:
        t = c * np.sqrt(df) / np.sqrt(1 - c**2)
        p = calculate_p_value(t, df)
    else:
        p = None

    return CorrelationResult(c=c, p=p)

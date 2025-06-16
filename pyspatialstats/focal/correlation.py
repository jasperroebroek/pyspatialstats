import numpy as np
from numpy.typing import ArrayLike

from pyspatialstats.focal.core.correlation import _focal_correlation
from pyspatialstats.rolling import rolling_window
from pyspatialstats.bootstrap.p_values import calculate_p_value
from pyspatialstats.types.cy_types import CyFocalCorrelationResult
from pyspatialstats.types.results import CorrelationResult
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import create_output_array, parse_raster, timeit
from pyspatialstats.windows import define_window, validate_window


@timeit
def focal_correlation(
    a1: ArrayLike,
    a2: ArrayLike,
    *,
    window: WindowT = 5,
    fraction_accepted: float = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    p_values: bool = False,
) -> CorrelationResult:
    """
    Focal correlation

    Parameters
    ----------
    a1, a2 : array-like
        Input arrays that will be correlated. They need to have the same shape and have two dimensions.
    window : int, array-like, Window
        Window that is applied over `a`. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a boolean array or a :class:`pyspatialstats.window.Window` object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable

        * ``0``: all views are calculated if at least 1 value is present
        * ``1``: only views completely filled with values are calculated
        * ``0-1``: fraction of acceptability

    reduce : bool, optional
        Use all pixels exactly once, without windows overlapping. The resulting array will have the shape:
        ``a_shape / window_shape``
    verbose : bool, optional
        Verbosity with timing

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the focal statistic. If `reduce` is set to False, the output has the same shape as the input,
        while if `reduce` is True, the output is reduced by the window size: ``raster_shape // window_shape``.
    """
    a1 = parse_raster(a1)
    a2 = parse_raster(a2)

    if a1.shape != a2.shape:
        raise ValueError(f'Input arrays have different shapes: {a1.shape=}, {a2.shape=}')

    window = define_window(window)
    validate_window(window, a1.shape, reduce, allow_even=False)
    mask = window.get_mask(2)

    fringes = window.get_fringes(reduce)
    ind_inner = window.get_ind_inner(reduce)
    threshold = window.get_threshold(fraction_accepted=fraction_accepted)

    df = create_output_array(a1, window.get_raster_shape(), reduce, dtype=np.uintp)
    c = create_output_array(a1, window.get_raster_shape(), reduce, dtype=np.float64)

    r = CyFocalCorrelationResult(df=df[ind_inner], c=c[ind_inner])

    a1_windowed = rolling_window(a1, window=window.get_shape(), reduce=reduce)
    a2_windowed = rolling_window(a2, window=window.get_shape(), reduce=reduce)

    _focal_correlation(
        a1_windowed,
        a2_windowed,
        mask=window.get_mask(),
        r=r,
        fringe=np.asarray(fringes, dtype=np.int32),
        threshold=threshold,
        reduce=reduce,
    )

    if p_values:
        t = c * np.sqrt(df) / np.sqrt(1 - c**2)
        p = calculate_p_value(t, df)
    else:
        p = None

    return CorrelationResult(c=c, p=p)

from functools import partial
from typing import Optional

from pyspatialstats.focal._core import focal_stats, focal_stats_base
from pyspatialstats.focal.core.min import _focal_min
from pyspatialstats.types.arrays import Array, RasterFloat64
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import timeit


@timeit
def focal_min(
    a: Array,
    *,
    window: WindowT,
    fraction_accepted: float = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    chunks: Optional[int | tuple[int, int]] = None,
    out: Optional[Array] = None,
) -> RasterFloat64:
    """
    Focal minimum.

    Parameters
    ----------
    a: Array
        Input array. Must be two-dimensional.
    window : int, array-like, or Window
        Window applied over the input array. It can be:

        - An integer (interpreted as a square window),
        - A sequence of integers (interpreted as a rectangular window),
        - A boolean array,
        - Or a :class:`pyspatialstats.windows.Window` object.
    fraction_accepted : float, optional
        Fraction of valid (non-NaN) cells per window required for the statistic to be computed.

        - ``0``: use windows with at least 1 valid value
        - ``1``: use only fully valid windows
        - Between ``0`` and ``1``: minimum acceptable fraction

        Default is 0.7.
    verbose : bool, optional
        If True, print progress message with timing. Default is False.
    reduce : bool, optional
        If True, each pixel is used exactly once without overlapping windows. The resulting array will have shape
        ``a_shape / window_shape``. Default is False.
    chunks : int or tuple of int, optional
        Shape of chunks to split the array into. If None, the array is not split into chunks, which is the default.
    out : :obj:`~numpy.ndarray`, optional
        Output array.

    Returns
    -------
    :obj:`~numpy.ndarray`
    """
    return focal_stats(
        data={'a': a},
        func=partial(focal_stats_base, stat_func=_focal_min),
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        chunks=chunks,
        out=out,
    )

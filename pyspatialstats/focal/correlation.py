from typing import Optional

import numpy as np

from pyspatialstats.focal._core import focal_stats, focal_stats_base
from pyspatialstats.focal.core.correlation import _focal_correlation
from pyspatialstats.focal.result_config import FocalCorrelationResultConfig
from pyspatialstats.results.stats import CorrelationResult
from pyspatialstats.stats.p_values import calculate_p_value
from pyspatialstats.types.arrays import Array
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import timeit


def _focal_correlation_base(
    data: dict[str, Array],
    *,
    window: WindowT,
    fraction_accepted: float,
    reduce: bool,
    result_config: FocalCorrelationResultConfig,
    out: Optional[CorrelationResult],
) -> CorrelationResult:
    r: CorrelationResult = focal_stats_base(
        data=data,
        stat_func=_focal_correlation,
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        result_config=result_config,
        out=out,
    )

    if result_config.p_values:
        t = r.c * np.sqrt(r.df) / np.sqrt(1 - r.c**2)
        r.p = calculate_p_value(t, r.df, out=r.p)

    return r


@timeit
def focal_correlation(
    a1: Array,
    a2: Array,
    *,
    window: WindowT,
    fraction_accepted: float = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    chunks: Optional[int | tuple[int, int]] = None,
    p_values: bool = False,
    out: Optional[CorrelationResult] = None,
) -> CorrelationResult:
    """
    Focal correlation.

    Parameters
    ----------
    a1, a2 : Array
        Input arrays to be correlated. They must have the same shape and be two-dimensional.
    window : int, array-like, or Window, optional
        Window applied over the input arrays. It can be:

        - An integer (interpreted as a square window),
        - A sequence of integers (interpreted as a rectangular window),
        - A boolean array,
        - Or a :class:`pyspatialstats.windows.Window` object.
    fraction_accepted : float, optional
        Fraction of valid cells (i.e., not NaN) per window required for the correlation to be computed.

        - ``0``: include views with at least 1 valid value
        - ``1``: include only fully valid views
        - Between ``0`` and ``1``: minimum fraction of valid values required

        Default is 0.7.
    verbose : bool, optional
        If True, print timing. Default is False.
    reduce : bool, optional
        If True, use each pixel exactly once without overlapping windows. The resulting array will have shape
        ``a_shape / window_shape``. Default is False.
    p_values : bool, optional
        If True, calculate p-values along with correlation coefficients. Default is False.
    chunks : int or tuple of int, optional
        Shape of chunks to split the array into. If None, the array is not split into chunks, which is the default.
    out : CorrelationResult, optional
        CorrelationResult object to write results to.

    Returns
    -------
    CorrelationResult
        Dataclass containing correlation coefficients (and optionally p-values).
    """
    return focal_stats(
        data={'a1': a1, 'a2': a2},
        func=_focal_correlation_base,
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        chunks=chunks,
        result_config=FocalCorrelationResultConfig(p_values=p_values),
        out=out,
    )

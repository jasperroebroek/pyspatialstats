from typing import Optional

from pyspatialstats.focal.core.linear_regression import _focal_linear_regression
from pyspatialstats.focal.focal_core import focal_stats, focal_stats_base
from pyspatialstats.focal.result_config import FocalLinearRegressionResultConfig
from pyspatialstats.types.results import LinearRegressionResult
from pyspatialstats.stats.p_values import calculate_p_value
from pyspatialstats.types.arrays import Array
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import (
    timeit,
)


def _focal_linear_regression_base(
    a1: Array,
    a2: Array,
    *,
    window: WindowT,
    fraction_accepted: float,
    reduce: bool,
    out: Optional[LinearRegressionResult],
    result_config: FocalLinearRegressionResultConfig,
) -> LinearRegressionResult:
    r: LinearRegressionResult = focal_stats_base(
        a1,
        a2,
        cy_func=_focal_linear_regression,
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        result_config=result_config,
        out=out,
    )

    if result_config.p_values:
        r.p_a = calculate_p_value(r.t_a, r.df, out=r.p_a)
        r.p_b = calculate_p_value(r.t_b, r.df, out=r.p_b)

    return r


@timeit
def focal_linear_regression(
    a1: Array,
    a2: Array,
    *,
    window: WindowT,
    fraction_accepted: float = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    p_values: bool = False,
    chunks: Optional[int | tuple[int, int]] = None,
    out: Optional[LinearRegressionResult] = None,
) -> LinearRegressionResult:
    """
    Focal linear regression.

    Parameters
    ----------
    a1, a2 : Array
        Input arrays to be regressed. Must have the same shape and be two-dimensional.
    window : int, array-like, or Window
        Window applied over the input arrays. It can be:

        - An integer (interpreted as a square window),
        - A sequence of integers (interpreted as a rectangular window),
        - A boolean array,
        - Or a :class:`pyspatialstats.window.Window` object.
    fraction_accepted : float, optional
        Fraction of valid (non-NaN) cells per window required for regression to be performed.

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
    p_values : bool, optional
        If True, calculate p-values for the regression coefficients. Default is False.
    out : LinearRegressionResult, optional
        LinearRegressionResult to write results to.

    Returns
    -------
    LinearRegressionResult
        StatResult containing regression coefficients, standard error values, t-statistics, and optionally p-values.
    """
    return focal_stats(
        a1,
        a2,
        func=_focal_linear_regression_base,
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        chunks=chunks,
        result_config=FocalLinearRegressionResultConfig(p_values=p_values),
        out=out,
    )

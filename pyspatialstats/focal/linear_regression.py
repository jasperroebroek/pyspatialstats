from functools import partial
from typing import Optional

import numpy as np

from pyspatialstats.focal._core import focal_stats, focal_stats_base
from pyspatialstats.focal.core.linear_regression import _focal_linear_regression
from pyspatialstats.focal.result_config import FocalLinearRegressionResultConfig
from pyspatialstats.results.stats import RegressionResult
from pyspatialstats.stats.p_values import calculate_p_value
from pyspatialstats.types.arrays import Array
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import timeit


def _focal_linear_regression_base(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    beta: np.ndarray,
    beta_se: np.ndarray,
    r_squared: np.ndarray,
    df: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    fringe: tuple[int, int],
    threshold: float,
    reduce: bool,
):
    _focal_linear_regression(
        x=x,
        y=y,
        mask=mask,
        beta=beta,
        beta_se=beta_se,
        r_squared=r_squared,
        df=df,
        fringe=fringe,
        threshold=threshold,
        reduce=reduce,
    )

    t[:] = beta / beta_se
    p[:] = calculate_p_value(t, df[..., np.newaxis])


@timeit
def focal_linear_regression(
    x: Array,
    y: Array,
    *,
    window: WindowT,
    fraction_accepted: float = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    chunks: Optional[int | tuple[int, int]] = None,
    out: Optional[RegressionResult] = None,
) -> RegressionResult:
    """
    Focal linear regression.

    Parameters
    ----------
    y : Array
        Dependent variable. Must be two-dimensional.
    x : Array
        Independent variables. Must be two or three-dimensional. If two-dimensional, it is interpreted as a single
        feature, internally transformed to three dimensions by adding a singleton dimension.
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
    RegressionResult
        StatResult containing regression coefficients, standard error values, t-statistics, and optionally p-values.
    """
    nf = x.shape[2] + 1 if x.ndim == 3 else 2
    x_ndim = 2 if x.ndim == 2 else 3

    result = focal_stats(
        data={'x': x, 'y': y},
        func=partial(focal_stats_base, stat_func=_focal_linear_regression_base),
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        chunks=chunks,
        result_config=FocalLinearRegressionResultConfig(nf=nf, x_ndim=x_ndim),
        out=out,
    )

    return result

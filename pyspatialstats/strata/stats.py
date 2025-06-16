from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pyspatialstats.grouped.utils import parse_data
from pyspatialstats.strata.core.stats import (
    _strata_correlation,
    _strata_count,
    _strata_linear_regression,
    _strata_max,
    _strata_mean,
    _strata_mean_std,
    _strata_min,
    _strata_std,
)
from pyspatialstats.types.arrays import RasterFloat64, RasterSizeT, RasterT
from pyspatialstats.types.results import CorrelationResult, LinearRegressionResult, Result


def strata_fun(fun: Callable, ind: RasterSizeT, **data) -> RasterFloat64 | Result:
    ind = np.asarray(ind)
    if ind.ndim != 2:
        raise IndexError('Only 2D data is supported')
    rows, cols = ind.shape
    parsed_data = parse_data(ind, **data)
    return fun(rows=rows, cols=cols, **parsed_data)


def strata_count(ind: RasterSizeT, v: RasterT) -> RasterFloat64:
    """
    Calculate the number of occurrences of each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    np.ndarray
        The minimum value in each index.
    """
    return strata_fun(_strata_count, ind=ind, v=v)


def strata_min(ind: RasterSizeT, v: RasterT) -> RasterFloat64:
    """
    Calculate the minimum value in each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    np.ndarray
        The minimum value in each index.
    """
    return strata_fun(_strata_min, ind=ind, v=v)


def strata_max(ind: RasterSizeT, v: RasterT) -> RasterFloat64:
    """
    Calculate the maximum value in each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    np.ndarray
        The maximum value in each index.
    """
    return strata_fun(_strata_max, ind=ind, v=v)


def strata_mean(ind: RasterSizeT, v: RasterT) -> RasterFloat64:
    """
    Calculate the mean value in each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    np.ndarray
        The mean value in each index.
    """
    return strata_fun(_strata_mean, ind=ind, v=v)


def strata_std(ind: RasterSizeT, v: RasterT) -> RasterFloat64:
    """
    Calculate the standard deviation in each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    np.ndarray
        The standard deviation in each index.
    """
    return strata_fun(_strata_std, ind=ind, v=v)


def strata_mean_std(ind: RasterSizeT, v: RasterT) -> RasterFloat64:
    """
    Calculate the mean and standard deviation in each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    np.ndarray
        The mean and standard deviation in each index.
    """
    return strata_fun(_strata_mean_std, ind=ind, v=v)


def strata_correlation(ind: RasterSizeT, v1: NDArray, v2: NDArray) -> CorrelationResult:
    """
    Calculate the correlation coefficient between two variables in each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v1, v2 : array-like
        two-dimensional data

    Returns
    -------
    CorrelationResult
        The correlation coefficient in each index.
        c - the correlation coefficient
        p - the p-value
    """
    return strata_fun(_strata_correlation, ind=ind, v1=v1, v2=v2)


def strata_linear_regression(ind: RasterSizeT, v1: NDArray, v2: NDArray) -> LinearRegressionResult:
    """
    Perform a linear regression in each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v1, v2 : array-like
        two-dimensional data

    Returns
    -------
    LinearRegressionResult
        The result of the linear regression in each index.
        a - the slope
        b - the intercept
        se_a - the standard error of the slope
        se_b - the standard error of the intercept
        p_a - the p-value
        p_b - the p-value
    """
    return strata_fun(_strata_linear_regression, ind=ind, v1=v1, v2=v2)

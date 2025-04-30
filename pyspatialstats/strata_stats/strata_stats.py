from typing import Callable

import numpy as np
from numpydantic import NDArray

from pyspatialstats.grouped_stats.utils import parse_data
from pyspatialstats.strata_stats.core.stats import (
    StrataCorrelationResult,
    StrataLinearRegressionResult,
    _strata_correlation,
    _strata_count,
    _strata_linear_regression,
    _strata_max,
    _strata_mean,
    _strata_mean_std,
    _strata_min,
    _strata_std,
)
from pyspatialstats.types import RasterFloat32


def strata_fun(fun: Callable, ind: NDArray, **data) -> RasterFloat32:
    ind = np.asarray(ind)
    if ind.ndim != 2:
        raise IndexError("Only 2D data is supported")
    rows, cols = ind.shape
    parsed_data = parse_data(ind, **data)
    return fun(rows=rows, cols=cols, **parsed_data)


def strata_count(ind: NDArray, v: NDArray) -> RasterFloat32:
    """
    Calculate the number of occurrences of each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v : array-like
        The data to calculate the minimum of.

    Returns
    -------
    RasterFloat32
        The minimum value in each stratum.
    """
    return strata_fun(_strata_count, ind=ind, v=v)


def strata_min(ind: NDArray, v: NDArray) -> RasterFloat32:
    """
    Calculate the minimum value in each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v : array-like
        The data to calculate the minimum of.

    Returns
    -------
    RasterFloat32
        The minimum value in each stratum.
    """
    return strata_fun(_strata_min, ind=ind, v=v)


def strata_max(ind: NDArray, v: NDArray) -> RasterFloat32:
    """
    Calculate the maximum value in each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v : array-like
        The data to calculate the maximum of.

    Returns
    -------
    RasterFloat32
        The maximum value in each stratum.
    """
    return strata_fun(_strata_max, ind=ind, v=v)


def strata_mean(ind: NDArray, v: NDArray) -> RasterFloat32:
    """
    Calculate the mean value in each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v : array-like
        The data to calculate the mean of.

    Returns
    -------
    RasterFloat32
        The mean value in each stratum.
    """
    return strata_fun(_strata_mean, ind=ind, v=v)


def strata_std(ind: NDArray, v: NDArray) -> RasterFloat32:
    """
    Calculate the standard deviation in each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v : array-like
        The data to calculate the standard deviation of.

    Returns
    -------
    RasterFloat32
        The standard deviation in each stratum.
    """
    return strata_fun(_strata_std, ind=ind, v=v)


def strata_mean_std(ind: NDArray, v: NDArray) -> RasterFloat32:
    """
    Calculate the mean and standard deviation in each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v : array-like
        The data to calculate the mean and standard deviation of.

    Returns
    -------
    RasterFloat32
        The mean and standard deviation in each stratum.
    """
    return strata_fun(_strata_mean_std, ind=ind, v=v)


def strata_correlation(ind: NDArray, v1: NDArray, v2: NDArray) -> StrataCorrelationResult:
    """
    Calculate the correlation coefficient between two variables in each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v1, v2 : array-like
        The data

    Returns
    -------
    StrataCorrelationResult
        The correlation coefficient in each stratum.
        c - the correlation coefficient
        p - the p-value
    """
    return strata_fun(_strata_correlation, ind=ind, v1=v1, v2=v2)


def strata_linear_regression(ind: NDArray, v1: NDArray, v2: NDArray) -> StrataLinearRegressionResult:
    """
    Perform a linear regression in each stratum.

    Parameters
    ----------
    ind : array-like
        The strata index.
    v1, v2 : array-like
        The data

    Returns
    -------
    StrataLinearRegressionResult
        The result of the linear regression in each stratum.
        a - the slope
        b - the intercept
        se_a - the standard error of the slope
        se_b - the standard error of the intercept
        p_a - the p-value
        p_b - the p-value
    """
    return strata_fun(_strata_linear_regression, ind=ind, v1=v1, v2=v2)

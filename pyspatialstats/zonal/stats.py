from dataclasses import fields

import numpy as np

from pyspatialstats.grouped import (
    grouped_correlation,
    grouped_count,
    grouped_linear_regression,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_std,
)
from pyspatialstats.results.stats import CorrelationResult, GroupedStatResult, RegressionResult, StatResult
from pyspatialstats.types.arrays import Array
from pyspatialstats.utils import timeit


def zonal_fun(ind: np.ndarray[tuple[int, ...], np.uintp], grouped_result: GroupedStatResult) -> Array | StatResult:
    if isinstance(grouped_result, StatResult):
        field_values = {}
        for f in fields(grouped_result):
            val = getattr(grouped_result, f.name)
            if val is not None:
                field_values[f.name] = val[ind]
        return type(grouped_result)(**field_values)
    return grouped_result[ind]


@timeit
def zonal_min(
    ind: Array,
    v: Array,
    verbose: bool = False,  # noqa
    **kwargs,
) -> np.ndarray[tuple[int, ...], np.float64]:
    """
    Calculate the minimum value at each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    verbose : bool, optional
        Print timing information
    kwargs
        Keyword arguments for grouped_min

    Returns
    -------
    np.ndarray
        The minimum value at each index.
    """
    ind = np.asarray(ind, dtype=np.uintp)
    min_result = grouped_min(ind, v, filtered=False, **kwargs)
    return zonal_fun(ind, min_result)


@timeit
def zonal_count(
    ind: Array,
    v: Array,
    verbose: bool = False,  # noqa
    **kwargs,
) -> np.ndarray[tuple[int, ...], np.uintp]:
    """
    Calculate the count of each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    verbose : bool, optional
        Print timing information
    kwargs
        Keyword arguments for grouped_count

    Returns
    -------
    np.ndarray
        The count of each index.
    """
    ind = np.asarray(ind, dtype=np.uintp)
    count_result = grouped_count(ind, v, filtered=False, **kwargs)
    return zonal_fun(ind, count_result)


@timeit
def zonal_max(
    ind: Array,
    v: Array,
    verbose: bool = False,  # noqa
    **kwargs,
) -> np.ndarray[tuple[int, ...], np.float64]:
    """
    Calculate the maximum value at each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    verbose : bool, optional
        Print timing information
    kwargs
        Keyword arguments for grouped_max

    Returns
    -------
    np.ndarray
        The maximum value at each index.
    """
    ind = np.asarray(ind, dtype=np.uintp)
    max_result = grouped_max(ind, v, filtered=False, **kwargs)
    return zonal_fun(ind, max_result)


@timeit
def zonal_std(
    ind: Array,
    v: Array,
    verbose: bool = False,  # noqa
    **kwargs,
) -> np.ndarray[tuple[int, ...], np.float64]:
    """
    Calculate the standard deviation at each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    verbose : bool, optional
        Print timing information
    kwargs
        Keyword arguments for grouped_std

    Returns
    -------
    np.ndarray
        The standard deviation at each index.
    """
    ind = np.asarray(ind, dtype=np.uintp)
    std_result = grouped_std(ind, v, filtered=False, **kwargs)
    return zonal_fun(ind, std_result)


@timeit
def zonal_mean(
    ind: Array,
    v: Array,
    verbose: bool = False,  # noqa
    **kwargs,
) -> np.ndarray[tuple[int, ...], np.float64]:
    """
    Calculate the mean value in each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    verbose : bool, optional
        Print timing information
    kwargs
        Keyword arguments for grouped_mean

    Returns
    -------
    np.ndarray
        The mean value in each index.
    """
    ind = np.asarray(ind, dtype=np.uintp)
    mean_result = grouped_mean(ind, v, filtered=False, **kwargs)
    return zonal_fun(ind, mean_result)


def zonal_correlation(
    ind: Array,
    v1: Array,
    v2: Array,
    verbose: bool = False,  # noqa
    **kwargs,
) -> CorrelationResult:
    """
    Calculate the correlation coefficient between two variables in each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v1, v2 : array-like
        data
    verbose : bool, optional
        Print timing information
    kwargs
        Keyword arguments for grouped_correlation

    Returns
    -------
    CorrelationResult
    """
    ind = np.asarray(ind, dtype=np.uintp)
    correlation_result = grouped_correlation(ind, v1, v2, filtered=False, **kwargs)
    return zonal_fun(ind, correlation_result)


def zonal_linear_regression(
    ind: Array,
    x: Array,
    y: Array,
    verbose: bool = False,  # noqa
    **kwargs,
) -> RegressionResult:
    """
    Perform a linear regression in each index.

    Parameters
    ----------
    ind : array-like
        index labels
    y : array-like
        dependent data
    x : array-like
        independent data. Must have one dimension more than `y`, with the first dimension corresponding to the index
        labels and the last dimension corresponding to the independent variables.
    verbose : bool, optional
        Print timing information
    kwargs
        Keyword arguments for grouped_linear_regression

    Returns
    -------
    RegressionResult
    """
    ind = np.asarray(ind, dtype=np.uintp)
    regression_results = grouped_linear_regression(ind, x, y, filtered=False, **kwargs)
    return zonal_fun(ind, regression_results)

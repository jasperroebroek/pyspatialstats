import numpy as np
import pandas as pd

from pyspatialstats.grouped.core.correlation import (
    grouped_correlation_npy,
    grouped_correlation_npy_filtered,
)
from pyspatialstats.grouped.core.count import (
    grouped_count_npy,
    grouped_count_npy_filtered,
)
from pyspatialstats.grouped.core.linear_regression import (
    grouped_linear_regression_npy,
    grouped_linear_regression_npy_filtered,
)
from pyspatialstats.grouped.core.max import (
    grouped_max_npy,
    grouped_max_npy_filtered,
)
from pyspatialstats.grouped.core.mean import (
    grouped_mean_npy,
    grouped_mean_npy_filtered,
)
from pyspatialstats.grouped.core.min import (
    grouped_min_npy,
    grouped_min_npy_filtered,
)
from pyspatialstats.grouped.core.std import (
    grouped_std_npy,
    grouped_std_npy_filtered,
)
from pyspatialstats.grouped.utils import (
    generate_index,
    grouped_fun,
    grouped_fun_pd,
    parse_array,
)
from pyspatialstats.types.results import CorrelationResult, LinearRegressionResult


def grouped_max(ind: np.ndarray, v: np.ndarray) -> np.ndarray[tuple[int], np.float64]:
    """
    Compute the maximum of each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    maxima : np.ndarray
        The maximum of each index.
    """
    return grouped_fun(grouped_max_npy, ind=ind, v=v)


def grouped_max_pd(ind: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """
    Compute the maximum of each stratum in a pandas DataFrame

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    maxima : pd.DataFrame
        The maximum of each stratum.
    """
    return grouped_fun_pd(grouped_max_npy_filtered, name='maximum', ind=ind, v=v)


def grouped_min(ind: np.ndarray, v: np.ndarray) -> np.ndarray[tuple[int], np.float64]:
    """
    Compute the minimum of each stratum.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    minima : np.ndarray
        The minimum of each stratum.
    """
    return grouped_fun(grouped_min_npy, ind=ind, v=v)


def grouped_min_pd(ind: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """
    Compute the minimum of each stratum in a pandas DataFrame.

        Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    minima : pd.DataFrame
        The minimum of each index.
    """
    return grouped_fun_pd(grouped_min_npy_filtered, name='minimum', ind=ind, v=v)


def grouped_count(ind: np.ndarray, v: np.ndarray) -> np.ndarray[tuple[int], np.int64]:
    """
    Compute the count of each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    counts : np.ndarray
        The count of each index.
    """
    return grouped_fun(grouped_count_npy, ind=ind, v=v)


def grouped_count_pd(ind: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """
    Compute the count of each index in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    counts : pd.DataFrame
        The count of each index.
    """
    return grouped_fun_pd(grouped_count_npy_filtered, name='count', ind=ind, v=v)


def grouped_mean(ind: np.ndarray, v: np.ndarray) -> np.ndarray[tuple[int], np.float64]:
    """
    Compute the mean of each index.

        Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    means : np.ndarray
        The mean of each index.
    """
    return grouped_fun(grouped_mean_npy, ind=ind, v=v)


def grouped_mean_pd(ind: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """
    Compute the mean of each index in a pandas DataFrame.

        Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    means : pd.DataFrame
        The mean of each index.
    """
    return grouped_fun_pd(grouped_mean_npy_filtered, name='mean', ind=ind, v=v)


def grouped_std(ind: np.ndarray, v: np.ndarray) -> np.ndarray[tuple[int], np.float64]:
    """
    Compute the standard deviation of each index.

        Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    stds : np.ndarray
        The standard deviation of each index.
    """
    return grouped_fun(grouped_std_npy, ind=ind, v=v)


def grouped_std_pd(ind: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """
    Compute the standard deviation of each index in a pandas DataFrame.

        Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    stds : pd.DataFrame
        The standard deviation of each index.
    """
    return grouped_fun_pd(grouped_std_npy_filtered, name='std', ind=ind, v=v)


def grouped_mean_std_pd(ind: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """
    Compute the mean and standard deviation of each index in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data

    Returns
    -------
    pd.DataFrame
        The mean and standard deviation of each index.
    """
    ind = parse_array('ind', ind).ravel()
    v = parse_array('v', v).ravel()

    if ind.size != v.size:
        raise IndexError(f'Arrays are not all of the same size: {ind.size=}, {v.size=}')

    index = generate_index(ind, v)
    mean_v = grouped_mean_npy_filtered(ind, v)
    std_v = grouped_std_npy_filtered(ind, v)

    return pd.DataFrame(data={'mean': mean_v, 'std': std_v}, index=index)


def grouped_correlation(ind: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> CorrelationResult:
    """
    Compute the correlation of each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v1, v2 : array-like
        two-dimensional data

    Returns
    -------
    GroupedCorrelationResult
        The correlation of each index. The Namedtuple will include the following attributes:
            c: the correlation coefficient
            p: the p-value
    """
    return grouped_fun(grouped_correlation_npy, ind=ind, v1=v1, v2=v2)


def grouped_correlation_pd(ind: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> pd.DataFrame:
    """
    Compute the correlation of each index in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v1, v2 : array-like
        two-dimensional data

    Returns
    -------
    pd.DataFrame
        The correlation of each index. The DataFrame will include the following columns:
        * c: the correlation coefficient
        * p: the p-value
    """
    return grouped_fun_pd(grouped_correlation_npy_filtered, name='correlation', ind=ind, v1=v1, v2=v2)


def grouped_linear_regression(ind: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> LinearRegressionResult:
    """
    Compute the linear regression of each index.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v1, v2 : array-like
        two-dimensional data

    Returns
    -------
    GroupedLinearRegressionResult
        The linear regression of each index.
        LinearRegressionResult is a named tuple with the following attributes:
        * a: the slope
        * b: the intercept
        * se_a: the standard error of the slope
        * se_b: the standard error of the intercept
        * t_a: the t-statistic of the slope
        * t_b: the t-statistic of the intercept
        * p_a: the p-value of the slope
        * p_b: the p-value of the intercept
    """
    return grouped_fun(grouped_linear_regression_npy, ind=ind, v1=v1, v2=v2)


def grouped_linear_regression_pd(ind: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> pd.DataFrame:
    """
    Compute the linear regression of each index in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v1, v2 : array-like
        two-dimensional data

    Returns
    -------
    pd.DataFrame
        The linear regression of each index. The DataFrame will include the following columns:
        * a: the slope
        * b: the intercept
        * se_a: the standard error of the slope
        * se_b: the standard error of the intercept
        * t_a: the t-statistic of the slope
        * t_b: the t-statistic of the intercept
        * p_a: the p-value of the slope
        * p_b: the p-value of the intercept
    """
    return grouped_fun_pd(grouped_linear_regression_npy_filtered, name='lr', ind=ind, v1=v1, v2=v2)

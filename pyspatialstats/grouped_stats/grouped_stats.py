import pandas as pd
from numpydantic import Shape
from numpydantic.dtype import Float64, Int64
from numpydantic.ndarray import NDArray

from pyspatialstats.grouped_stats.core.correlation import (
    CorrelationResult,
    grouped_correlation_npy,
    grouped_correlation_npy_filtered,
)
from pyspatialstats.grouped_stats.core.count import (
    grouped_count_npy,
    grouped_count_npy_filtered,
)
from pyspatialstats.grouped_stats.core.linear_regression import (
    LinearRegressionResult,
    grouped_linear_regression_npy,
    grouped_linear_regression_npy_filtered,
)
from pyspatialstats.grouped_stats.core.max import (
    grouped_max_npy,
    grouped_max_npy_filtered,
)
from pyspatialstats.grouped_stats.core.mean import (
    grouped_mean_npy,
    grouped_mean_npy_filtered,
)
from pyspatialstats.grouped_stats.core.min import (
    grouped_min_npy,
    grouped_min_npy_filtered,
)
from pyspatialstats.grouped_stats.core.std import (
    grouped_std_npy,
    grouped_std_npy_filtered,
)
from pyspatialstats.grouped_stats.utils import (
    generate_index,
    grouped_fun,
    grouped_fun_pd,
    parse_array,
)


def grouped_max(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float64]:
    """
    Compute the maximum of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    maximums : 1D np.ndarray
        The maximum of each stratum.
    """
    return grouped_fun(grouped_max_npy, ind=ind, v=v)


def grouped_max_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    """
    Compute the maximum of each stratum in a pandas DataFrame

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    maximums : pd.DataFrame
        The maximum of each stratum.
    """
    return grouped_fun_pd(grouped_max_npy_filtered, name="maximum", ind=ind, v=v)


def grouped_min(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float64]:
    """
    Compute the minimum of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    minimums : 1D np.ndarray
        The minimum of each stratum.
    """
    return grouped_fun(grouped_min_npy, ind=ind, v=v)


def grouped_min_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    """
    Compute the minimum of each stratum in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    minimums : pd.DataFrame
        The minimum of each stratum.
    """
    return grouped_fun_pd(grouped_min_npy_filtered, name="minimum", ind=ind, v=v)


def grouped_count(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Int64]:
    """
    Compute the count of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    counts : 1D np.ndarray
        The count of each stratum.
    """
    return grouped_fun(grouped_count_npy, ind=ind, v=v)


def grouped_count_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    """
    Compute the count of each stratum in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    counts : pd.DataFrame
        The count of each stratum.
    """
    return grouped_fun_pd(grouped_count_npy_filtered, name="count", ind=ind, v=v)


def grouped_mean(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float64]:
    """
    Compute the mean of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    means : 1D np.ndarray
        The mean of each stratum.
    """
    return grouped_fun(grouped_mean_npy, ind=ind, v=v)


def grouped_mean_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    """
    Compute the mean of each stratum in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    means : pd.DataFrame
        The mean of each stratum.
    """
    return grouped_fun_pd(grouped_mean_npy_filtered, name="mean", ind=ind, v=v)


def grouped_std(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float64]:
    """
    Compute the standard deviation of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    stds : 1D np.ndarray
        The standard deviation of each stratum.
    """
    return grouped_fun(grouped_std_npy, ind=ind, v=v)


def grouped_std_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    """
    Compute the standard deviation of each stratum in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    stds : pd.DataFrame
        The standard deviation of each stratum.
    """
    return grouped_fun_pd(grouped_std_npy_filtered, name="std", ind=ind, v=v)


def grouped_mean_std_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    """
    Compute the mean and standard deviation of each stratum in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    means_stds : pd.DataFrame
        The mean and standard deviation of each stratum.
    """
    ind = parse_array("ind", ind).ravel()
    v = parse_array("v", v).ravel()

    if ind.size != v.size:
        raise IndexError(f"Arrays are not all of the same size: {ind.size=}, {v.size=}")

    index = generate_index(ind, v)
    mean_v = grouped_mean_npy_filtered(ind, v)
    std_v = grouped_std_npy_filtered(ind, v)

    return pd.DataFrame(data={"mean": mean_v, "std": std_v}, index=index)


def grouped_correlation(
    ind: NDArray, v1: NDArray, v2: NDArray
) -> CorrelationResult:
    """
    Compute the correlation of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v1, v2 : array-like
        Data.

    Returns
    -------
    maximums : GroupedCorrelationResult
        The correlation of each stratum. The NamedTuple will include the following attributes:
            c: the correlation coefficient
            p: the p-value
    """
    return grouped_fun(grouped_correlation_npy, ind=ind, v1=v1, v2=v2)


def grouped_correlation_pd(ind: NDArray, v1: NDArray, v2: NDArray) -> pd.DataFrame:
    """
    Compute the correlation of each stratum in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v1, v2 : array-like
        Data.

    Returns
    -------
    correlations : pd.DataFrame
        The correlation of each stratum. The DataFrame will include the following columns:
        * c: the correlation coefficient
        * p: the p-value
    """
    return grouped_fun_pd(
        grouped_correlation_npy_filtered, name="correlation", ind=ind, v1=v1, v2=v2
    )


def grouped_linear_regression(
    ind: NDArray, v1: NDArray, v2: NDArray
) -> LinearRegressionResult:
    """
    Compute the linear regression of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v1, v2 : array-like
        Data.

    Returns
    -------
    linear_regressions : GroupedLinearRegressionResult
        The linear regression of each stratum.
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


def grouped_linear_regression_pd(
    ind: NDArray, v1: NDArray, v2: NDArray
) -> pd.DataFrame:
    """
    Compute the linear regression of each stratum in a pandas DataFrame.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v1, v2 : array-like
        Data.

    Returns
    -------
    linear_regressions : pd.DataFrame
        The linear regression of each stratum. The DataFrame will include the following columns:
        * a: the slope
        * b: the intercept
        * se_a: the standard error of the slope
        * se_b: the standard error of the intercept
        * t_a: the t-statistic of the slope
        * t_b: the t-statistic of the intercept
        * p_a: the p-value of the slope
        * p_b: the p-value of the intercept
    """
    return grouped_fun_pd(
        grouped_linear_regression_npy_filtered, name="lr", ind=ind, v1=v1, v2=v2
    )

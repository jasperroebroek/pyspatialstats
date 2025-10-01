from typing import Literal, Optional

import numpy as np
import pandas as pd

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.grouped.accumulators import (
    GroupedBootstrapMeanAccumulator,
    GroupedCorrelationAccumulator,
    GroupedCountAccumulator,
    GroupedLinearRegressionAccumulator,
    GroupedMaxAccumulator,
    GroupedMinAccumulator,
    GroupedSumAccumulator,
    GroupedWelfordAccumulator,
)
from pyspatialstats.grouped.accumulators.linear_regression import GroupedBootstrapLinearRegressionAccumulator
from pyspatialstats.grouped.base import grouped_stats
from pyspatialstats.grouped.config import GroupedResultConfig
from pyspatialstats.grouped.utils import parse_data_linear_regression
from pyspatialstats.results.stats import RegressionResult, MeanResult
from pyspatialstats.types.arrays import Array
from pyspatialstats.utils import timeit


@timeit
def grouped_max(
    ind: Array,
    v: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    verbose: bool = False,  # noqa
) -> np.ndarray[tuple[int], np.float64] | pd.DataFrame:
    """
    Compute the maximum at each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    verbose : bool, optional
        Print timing

    Returns
    -------
    maxima : np.ndarray or pd.DataFrame
        The maximum at each index.
    """
    return grouped_stats(
        ind=ind, v=v, filtered=filtered, chunks=chunks, config=GroupedResultConfig(GroupedMaxAccumulator)
    )


@timeit
def grouped_min(
    ind: Array,
    v: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    verbose: bool = False,  # noqa
) -> np.ndarray[tuple[int], np.float64] | pd.DataFrame:
    """
    Compute the minimum at each index.

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    verbose : bool, optional
        Print timing

    Returns
    -------
    minima : np.ndarray or pd.DataFrame
        The minimum at each index.
    """
    return grouped_stats(
        ind=ind, v=v, filtered=filtered, chunks=chunks, config=GroupedResultConfig(GroupedMinAccumulator)
    )


@timeit
def grouped_count(
    ind: Array,
    v: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    verbose: bool = False,  # noqa
) -> np.ndarray[tuple[int], np.uintp] | pd.DataFrame:
    """
    Compute the count of each index. NaN values in v are ignored

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    verbose : bool, optional
        Print timing

    Returns
    -------
    counts : np.ndarray or pd.DataFrame
        The count of each index.
    """
    return grouped_stats(
        ind=ind, v=v, filtered=filtered, chunks=chunks, config=GroupedResultConfig(GroupedCountAccumulator)
    )


@timeit
def grouped_sum(
    ind: Array,
    v: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    verbose: bool = False,  # noqa
) -> np.ndarray[tuple[int], np.uintp] | pd.DataFrame:
    """
    Compute the sum at each index

    Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    verbose : bool, optional
        Print timing

    Returns
    -------
    counts : np.ndarray or pd.DataFrame
        The count of each index.
    """
    return grouped_stats(
        ind=ind, v=v, filtered=filtered, chunks=chunks, config=GroupedResultConfig(GroupedSumAccumulator)
    )


@timeit
def grouped_mean(
    ind: Array,
    v: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    std_df: Literal[0, 1] = 1,
    error: Optional[Literal['bootstrap', 'parametric']] = None,
    bootstrap_config: Optional[BootstrapConfig] = None,
    verbose: bool = False,  # noqa
) -> MeanResult | pd.DataFrame:
    """
    Compute the mean of each index.

        Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    std_df : {0, 1}, optional
        Degrees of freedom for the standard deviation if error is 'parametric'
    error : {'bootstrap', 'parametric'}, optional
        Compute the uncertainty of the mean using either a bootstrap or parametric method. If not set, the function only
        returns the mean. With bootstrap, the standard error is returned, and with parametric, the standard error and
        standard deviation are returned
    bootstrap_config : BootstrapConfig, optional
        Configuration for the bootstrap if error is 'bootstrap'
    verbose : bool, optional
        Print timing

    Returns
    -------
    MeanStatResult or pd.DataFrame
        The mean of each index. If filtered is True, this function returns a pandas DataFrame otherwise it returns a
        MeanStatResult object.
    """
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig()

    match error:
        case 'bootstrap':
            config = GroupedResultConfig(
                GroupedBootstrapMeanAccumulator,
                kwargs={'n_boot': bootstrap_config.n_bootstraps, 'seed': bootstrap_config.seed},
            )
        case 'parametric':
            config = GroupedResultConfig(
                GroupedWelfordAccumulator,
                kwargs={'std_df': std_df},
                to_result_func='to_mean_std_result',
                to_filtered_result_func='to_mean_std_filtered_result',
            )
        case None:
            config = GroupedResultConfig(
                GroupedSumAccumulator,
                to_result_func='to_mean_result',
                to_filtered_result_func='to_mean_filtered_result',
            )
        case _:
            raise ValueError('error must be either "bootstrap" or "parametric"')

    return grouped_stats(ind=ind, v=v, filtered=filtered, chunks=chunks, config=config)


@timeit
def grouped_std(
    ind: Array,
    v: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    std_df: Literal[0, 1] = 1,
    verbose: bool = False,  # noqa
) -> np.ndarray[tuple[int], np.float64]:
    """
    Compute the standard deviation at each index.

        Parameters
    ----------
    ind : array-like
        index labels
    v : array-like
        data
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    verbose : bool, optional
        Print timing

    Returns
    -------
    stds : np.ndarray
        The standard deviation at each index.
    """
    return grouped_stats(
        ind=ind,
        v=v,
        filtered=filtered,
        chunks=chunks,
        config=GroupedResultConfig(
            GroupedWelfordAccumulator,
            to_result_func='to_std_result',
            to_filtered_result_func='to_std_filtered_result',
            kwargs={'std_df': std_df},
        ),
    )


@timeit
def grouped_correlation(
    ind: Array,
    v1: Array,
    v2: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    verbose: bool = False,  # noqa
) -> np.ndarray[tuple[int], np.float64]:
    """
    Compute the standard deviation at each index.

        Parameters
    ----------
    ind : array-like
        index labels
    v1, v2 : array-like
        data
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    verbose : bool, optional
        Print timing

    Returns
    -------
    CorrelationResult
        The pearson correlation at each index.
        * r: the correlation coefficient
        * df: degrees of freedom
        * p: the p-value
    """
    return grouped_stats(
        ind=ind,
        v1=v1,
        v2=v2,
        filtered=filtered,
        chunks=chunks,
        config=GroupedResultConfig(GroupedCorrelationAccumulator),
    )


@timeit
def grouped_linear_regression(
    ind: Array,
    x: Array,
    y: Array,
    filtered: bool = True,
    chunks: Optional[int | tuple[int, ...]] = None,
    error: Optional[Literal['bootstrap', 'parametric']] = 'parametric',
    bootstrap_config: Optional[BootstrapConfig] = None,
    verbose: bool = False,  # noqa
) -> RegressionResult:
    """
    Compute the linear regression at each index.

    Parameters
    ----------
    ind : array-like
        index labels
    x : array-like
        Independent variables. Must have more one more dimension than y, the last dimension is the number of independent
        variables
    y: array-like
        dependent variable
    filtered : bool, optional
        Filter the output in a pandas dataframe, which is the default. If False, this function returns the raw output,
        where the index of the value corresponds to the index labels.
    chunks : int or tuple of ints, optional
        Optional chunking of the data, which can be run in parallel in a joblib context
    error : {'bootstrap', 'parametric'}, optional
        Compute the uncertainty of the linear regression parameters using either a bootstrap or parametric method. If
        not set, the function does not return the uncertainty.
    bootstrap_config : BootstrapConfig, optional
        Configuration for the bootstrap if error is 'bootstrap'
    verbose : bool, optional
        Print timing

    Returns
    -------
    RegressionResult
        The linear regression at each index.
        * df: degrees of freedom
        * beta: the slope
        * se: the standard error of the slope
        * t: the t-statistic
        * p: the p-value
    """
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig()

    match error:
        case 'bootstrap':
            config = GroupedResultConfig(
                GroupedBootstrapLinearRegressionAccumulator,
                parse_data_fun=parse_data_linear_regression,
                kwargs={'n_boot': bootstrap_config.n_bootstraps, 'seed': bootstrap_config.seed},
            )
        case 'parametric':
            config = GroupedResultConfig(
                GroupedLinearRegressionAccumulator, parse_data_fun=parse_data_linear_regression,
                kwargs={'calc_se': True, 'calc_r2': True},
            )
        case None:
            config = GroupedResultConfig(
                GroupedLinearRegressionAccumulator, parse_data_fun=parse_data_linear_regression,
                kwargs={'calc_se': False, 'calc_r2': False},
            )
        case _:
            raise ValueError('error must be either "bootstrap" or "parametric"')
    return grouped_stats(ind=ind, x=x, y=y, filtered=filtered, chunks=chunks, config=config)

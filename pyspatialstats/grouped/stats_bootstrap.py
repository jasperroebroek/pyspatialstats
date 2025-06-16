import pandas as pd
from numpy.typing import NDArray

from pyspatialstats.grouped.core.mean import grouped_mean_bootstrap_npy
from pyspatialstats.grouped.utils import (
    grouped_fun,
    grouped_fun_pd,
)
from pyspatialstats.types.results import BootstrapMeanResult


def grouped_mean_bootstrap(ind: NDArray, v: NDArray, n_bootstraps: int, seed: int) -> BootstrapMeanResult:
    """
    Compute the bootstrapped mean of each stratum.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data
    n_bootstraps : int
        number of bootstraps
    seed : int
        random seed

    Returns
    -------
    BootstrapMeanResult
        * mean : np.ndarray
        * se : np.ndarray
    """
    return grouped_fun(grouped_mean_bootstrap_npy, ind=ind, v=v, n_bootstraps=n_bootstraps, seed=seed)


def grouped_mean_bootstrap_pd(ind: NDArray, v: NDArray, n_bootstraps: int, seed: int) -> pd.DataFrame:
    """
    Compute the bootstrapped mean of each stratum.

    Parameters
    ----------
    ind : array-like
        two-dimensional index labels
    v : array-like
        two-dimensional data
    n_bootstraps : int
        number of bootstraps
    seed : int
        random seed

    Returns
    -------
    BootstrapMeanResult
        * mean : np.ndarray
        * se : np.ndarray
    """
    return grouped_fun_pd(
        grouped_mean_bootstrap_npy,
        name='mean',
        ind=ind,
        v=v,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

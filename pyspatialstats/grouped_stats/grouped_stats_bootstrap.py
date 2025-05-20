from numpydantic import NDArray
import pandas as pd

from pyspatialstats.grouped_stats.core.mean import grouped_mean_bootstrap_npy
from pyspatialstats.grouped_stats.utils import (
    grouped_fun,
    grouped_fun_pd,
)
from pyspatialstats.results import BootstrapMeanResult


def grouped_mean_bootstrap(
    ind: NDArray, v: NDArray, n_bootstraps: int, seed: int
) -> BootstrapMeanResult:
    """
    Compute the bootstrapped mean of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    BootstrapMeanResult
        * mean : 1D np.ndarray
        * se : 1D np.ndarray
    """
    return grouped_fun(
        grouped_mean_bootstrap_npy, ind=ind, v=v, n_bootstraps=n_bootstraps, seed=seed
    )


def grouped_mean_bootstrap_pd(
    ind: NDArray, v: NDArray, n_bootstraps: int, seed: int
) -> pd.DataFrame:
    """
    Compute the bootstrapped mean of each stratum.

    Parameters
    ----------
    ind : array-like
        Strata labels.
    v : array-like
        Data.

    Returns
    -------
    BootstrapMeanResult
        * mean : 1D np.ndarray
        * se : 1D np.ndarray
    """
    return grouped_fun_pd(
        grouped_mean_bootstrap_npy, ind=ind, v=v, n_bootstraps=n_bootstraps, seed=seed
    )

import numpy as np
import pytest

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.grouped import grouped_mean
from pyspatialstats.random.random import Random
from pyspatialstats.results.stats import MeanResult


def grouped_mean_bootstrap_simple(v: np.ndarray, n_bootstraps: int, seed: int) -> MeanResult:
    rng = Random(seed)
    sums = np.zeros(n_bootstraps)
    weights = np.zeros(n_bootstraps)

    for i in range(len(v)):
        for j in range(n_bootstraps):
            sample_idx = rng.np_randpoisson(lam=1, n=1).item()
            sums[j] += v[i] * sample_idx
            weights[j] += sample_idx

    mask = weights > 0
    means = sums[mask] / weights[mask]

    return MeanResult(mean=means.mean(), se=means.std(ddof=1))


@pytest.mark.parametrize('error', ['bootstrap', 'parametric', None])
def test_grouped_mean_shape_and_type(ind, v, error):
    result = grouped_mean(ind=ind, v=v, error=error, filtered=False)
    num_unique_ind = len(np.unique(ind))
    assert isinstance(result, MeanResult)
    assert result.shape == (num_unique_ind,)
    assert result.size == num_unique_ind


@pytest.mark.parametrize('error', ['bootstrap', 'parametric', None])
def test_grouped_mean_nan_handling(error):
    ind = np.array([0, 0, 1, 1, 2, 2])
    v = np.array([1.0, np.nan, 2.0, 4.0, 3.0, np.nan])
    result = grouped_mean(ind=ind, v=v, error=error, filtered=False)
    assert not np.isnan(result.mean[0])
    assert not np.isnan(result.mean[1])
    assert not np.isnan(result.mean[2])


def test_grouped_mean_bootstrap_estimate_correctness():
    rng = np.random.default_rng(42)
    ind = np.repeat(np.arange(10), 100)
    v = rng.normal(loc=5.0, scale=2.0, size=1000)

    result = grouped_mean(ind=ind, v=v, error='bootstrap', filtered=False)

    assert np.allclose(result.mean, 5.0, atol=0.5)
    assert np.all(result.se > 0.0)
    assert np.all(result.se < 1.0)


def test_grouped_mean_bootstrap_estimate_correctness_against_numpy():
    ind = np.ones(100)
    v = np.random.default_rng(42).normal(loc=5.0, scale=2.0, size=100)

    result = grouped_mean(
        ind=ind, v=v, error='bootstrap', filtered=False, bootstrap_config=BootstrapConfig(n_bootstraps=1000, seed=42)
    )
    result_expected = grouped_mean_bootstrap_simple(v, n_bootstraps=1000, seed=42)

    assert np.isclose(result.mean[1], result_expected.mean)
    assert np.isclose(result.se[1], result_expected.se)


def test_grouped_mean_bootstrap_filtered_result():
    ind = np.array([0, 0, 2, 2])
    v = np.array([1.0, 2.0, 4.0, 6.0])

    df = grouped_mean(ind=ind, v=v, error='bootstrap', filtered=True)
    assert set(df.columns) == {'mean', 'se'}
    assert set(df.index) == {0, 2}


def test_grouped_mean_parametric_filtered_result():
    ind = np.array([0, 0, 2, 2])
    v = np.array([1.0, 2.0, 4.0, 6.0])

    df = grouped_mean(ind=ind, v=v, error='parametric', filtered=True)
    assert set(df.columns) == {'mean', 'se', 'std'}
    assert set(df.index) == {0, 2}

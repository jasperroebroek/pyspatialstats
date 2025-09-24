import numpy as np
import pytest

from pyspatialstats.bootstrap.mean import py_bootstrap_mean
from tests.bootstrap.np_stats import np_bootstrap_mean


def test_bootstrap_mean():
    n_bootstraps = 1000
    data = np.random.randn(10).astype(np.float64)

    result = py_bootstrap_mean(data, n_bootstraps, seed=42)

    assert isinstance(result.mean, float)
    assert isinstance(result.se, float)

    assert -10 < result.mean < 10
    assert 0 < result.se < 10


def test_bootstrap_mean_single_bootstrap():
    data = np.random.randn(10)

    with pytest.raises(ValueError):
        py_bootstrap_mean(data, 1)


def test_bootstrap_mean_empty():
    data = np.array([], dtype=np.float64)

    # Check for exception or cy_result handling
    with pytest.raises(ValueError):
        py_bootstrap_mean(data, 1000, seed=42)


def test_bootstrap_mean_zero_variance():
    data = np.full(10, 1.0, dtype=np.float64)
    n_bootstraps = 1000

    result = py_bootstrap_mean(data, n_bootstraps, seed=42)

    assert result.mean == 1.0
    assert result.se == 0


def test_bootstrap_mean_seed_consistency():
    seed = 42
    data = np.random.randn(10).astype(np.float64)
    n_bootstraps = 1000

    r1 = py_bootstrap_mean(data, n_bootstraps, seed=seed)
    r2 = py_bootstrap_mean(data, n_bootstraps, seed=seed)

    assert r1.mean == r2.mean
    assert r1.se == r2.se


def test_bootstrap_mean_comparison_to_numpy():
    data = np.random.randn(100)
    n_bootstraps = 1000
    seed = 0

    cy_result = py_bootstrap_mean(data, n_bootstraps, seed)
    np_result = np_bootstrap_mean(data, n_bootstraps, seed)

    np.testing.assert_array_almost_equal(cy_result.mean, np_result.mean)
    np.testing.assert_array_almost_equal(cy_result.se, np_result.se.item(), decimal=3)

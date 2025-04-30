import numpy as np
import pytest

from pyspatialstats.bootstrap.mean import py_bootstrap_mean


def np_bootstrap_mean(v, n_bootstraps, seed):
    np.random.seed(seed)
    means = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        sample = np.random.choice(v, size=v.size, replace=True)
        means[i] = np.mean(sample)

    return means.mean(), means.std()


def test_bootstrap_mean():
    n_bootstraps = 1000
    data = np.random.randn(10).astype(np.float32)

    mean, se = py_bootstrap_mean(data, n_bootstraps, seed=42)

    assert isinstance(mean, float)
    assert isinstance(se, float)

    assert -10 < mean < 10
    assert 0 < se < 10


def test_bootstrap_mean_single_bootstrap():
    data = np.random.randn(10).astype(np.float32)

    with pytest.raises(ValueError):
        py_bootstrap_mean(data, 1)


def test_bootstrap_mean_empty():
    data = np.array([], dtype=np.float32)

    # Check for exception or result handling
    with pytest.raises(ValueError):
        py_bootstrap_mean(data, 1000, seed=42)


def test_bootstrap_mean_zero_variance():
    data = np.full(10, 1.0, dtype=np.float32)
    n_bootstraps = 1000

    mean, se = py_bootstrap_mean(data, n_bootstraps, seed=42)

    assert mean == 1.0
    assert se == 0


def test_bootstrap_mean_seed_consistency():
    seed = 42
    data = np.random.randn(10).astype(np.float32)
    n_bootstraps = 1000

    mean1, se1 = py_bootstrap_mean(data, n_bootstraps, seed=seed)
    mean2, se2 = py_bootstrap_mean(data, n_bootstraps, seed=seed)

    assert mean1 == mean2
    assert se1 == se2


def test_bootstrap_mean_comparison_to_numpy():
    data = np.random.randn(100).astype(np.float32)
    n_bootstraps = 1000
    seed = 0

    cy_mean, cy_se = py_bootstrap_mean(data, n_bootstraps, seed)
    np_mean, np_se = np_bootstrap_mean(data, n_bootstraps, seed)

    assert np.isclose(cy_mean, np_mean, atol=1e-2)
    assert np.isclose(cy_se, np_se, atol=1e-2)

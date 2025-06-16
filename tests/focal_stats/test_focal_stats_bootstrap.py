import numpy as np

from pyspatialstats.focal import focal_mean_bootstrap
from pyspatialstats.bootstrap.mean import py_bootstrap_mean


def test_basic_bootstrap():
    a = np.arange(100).reshape(10, 10)
    result = focal_mean_bootstrap(a, window=3, n_bootstraps=100, seed=42)
    mean, se = result.mean, result.se

    assert mean.shape == a.shape
    assert se.shape == a.shape
    assert np.nanmean(mean) > 0
    assert np.nanmean(se) > 0


def test_nan_handling():
    a = np.full((5, 5), 10.0)
    a[2, 2] = np.nan  # Introduce a NaN

    result = focal_mean_bootstrap(a, window=3, n_bootstraps=100, seed=123)
    assert np.isnan(result.mean[2, 2])
    assert np.isnan(result.se[2, 2])


def test_threshold():
    a = np.full((5, 5), 1.0)
    a[0, 0] = np.nan

    # Set threshold to 1.0, expect top-left window to be skipped
    result = focal_mean_bootstrap(
        a, window=3, fraction_accepted=1.0, n_bootstraps=10, seed=1
    )
    assert np.isnan(result.mean[1, 1])
    assert np.isnan(result.se[1, 1])

    # Set threshold to 0.0, expect top-left window to be accepted
    result = focal_mean_bootstrap(
        a, window=3, fraction_accepted=0.0, n_bootstraps=10, seed=1
    )
    assert not np.isnan(result.mean[1, 1])
    assert not np.isnan(result.se[1, 1])


def test_reduce_mode(rs):
    a = rs.random((6, 6))
    result = focal_mean_bootstrap(a, window=3, reduce=True, n_bootstraps=50, seed=0)
    assert result.mean.shape == (2, 2)
    assert result.se.shape == (2, 2)


def test_deterministic_output(rs):
    a = rs.random((10, 10))
    res1 = focal_mean_bootstrap(a, window=3, n_bootstraps=100, seed=123)
    res2 = focal_mean_bootstrap(a, window=3, n_bootstraps=100, seed=123)

    np.testing.assert_allclose(res1.mean, res2.mean, equal_nan=True)
    np.testing.assert_allclose(res1.se, res2.se, equal_nan=True)


def test_values(rs):
    a = rs.random((3, 3))
    result = focal_mean_bootstrap(a, window=3, n_bootstraps=100, seed=123)
    mean, se = py_bootstrap_mean(a.flatten(), 100, seed=123)

    assert result.mean[1, 1] == mean
    assert result.se[1, 1] == se

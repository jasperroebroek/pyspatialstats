import numpy as np
import pytest
import xarray as xr

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.bootstrap.mean import py_bootstrap_mean
from pyspatialstats.focal import focal_mean
from pyspatialstats.results.stats import MeanResult
from pyspatialstats.windows import define_window


def test_basic_bootstrap():
    a = np.arange(100).reshape(10, 10)
    result = focal_mean(a, window=3, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=42))
    mean, se = result.mean, result.se

    assert mean.shape == a.shape
    assert se.shape == a.shape
    assert np.nanmean(mean) > 0
    assert np.nanmean(se) > 0


def test_nan_handling():
    a = np.full((5, 5), 10.0)
    a[2, 2] = np.nan  # Introduce a NaN

    result = focal_mean(a, window=3, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=123, seed=42))
    assert np.isnan(result.mean[2, 2])
    assert np.isnan(result.se[2, 2])


def test_threshold():
    a = np.full((5, 5), 1.0)
    a[0, 0] = np.nan

    # Set threshold to 1.0, expect top-left window to be skipped
    result = focal_mean(
        a,
        window=3,
        fraction_accepted=1.0,
        error='bootstrap',
        bootstrap_config=BootstrapConfig(n_bootstraps=10, seed=1),
    )
    assert np.isnan(result.mean[1, 1])
    assert np.isnan(result.se[1, 1])

    # Set threshold to 0.0, expect top-left window to be accepted
    result = focal_mean(
        a,
        window=3,
        fraction_accepted=0.0,
        error='bootstrap',
        bootstrap_config=BootstrapConfig(n_bootstraps=10, seed=1),
    )
    assert not np.isnan(result.mean[1, 1])
    assert not np.isnan(result.se[1, 1])


def test_reduce_mode(rs):
    a = rs.random((6, 6))
    result = focal_mean(
        a, window=3, reduce=True, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=50, seed=0)
    )
    assert result.mean.shape == (2, 2)
    assert result.se.shape == (2, 2)


def test_deterministic_output(rs):
    a = rs.random((10, 10))
    res1 = focal_mean(a, window=3, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=123))
    res2 = focal_mean(a, window=3, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=123))

    np.testing.assert_allclose(res1.mean, res2.mean, equal_nan=True)
    np.testing.assert_allclose(res1.se, res2.se, equal_nan=True)


def test_values(rs):
    a = rs.random((3, 3))
    focal_result = focal_mean(
        a, window=3, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=123)
    )
    bootstrap_result = py_bootstrap_mean(a.flatten(), 100, seed=123)

    assert focal_result.mean[1, 1] == bootstrap_result.mean
    assert focal_result.se[1, 1] == bootstrap_result.se


def test_parallel(rs):
    a = rs.random((100, 100))

    linear_r = focal_mean(a, window=5, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=123))
    parallel_r = focal_mean(
        a, window=5, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=123), chunks=75
    )

    np.testing.assert_allclose(linear_r.mean, parallel_r.mean, equal_nan=True, atol=0.1)
    np.testing.assert_allclose(linear_r.se, parallel_r.se, equal_nan=True, atol=0.1)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
def test_xarray_output(rs, reduce, chunks):
    a = xr.DataArray(rs.random((100, 100)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a)

    output = MeanResult(
        mean=xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=np.float64)),
        se=xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=np.float64)),
    )

    bootstrap_config = BootstrapConfig(n_bootstraps=100, seed=123)
    r = focal_mean(
        a,
        window=5,
        out=output,
        reduce=reduce,
        chunks=chunks,
        error='bootstrap',
        bootstrap_config=bootstrap_config,
    )
    r_expected = focal_mean(
        a,
        window=5,
        out=None,
        reduce=reduce,
        chunks=chunks,
        error='bootstrap',
        bootstrap_config=bootstrap_config,
    )

    assert id(r.mean) == id(output.mean)
    assert id(r.se) == id(output.se)

    np.testing.assert_allclose(r.mean.values, r_expected.mean, equal_nan=True)
    np.testing.assert_allclose(r.se.values, r_expected.se, equal_nan=True)

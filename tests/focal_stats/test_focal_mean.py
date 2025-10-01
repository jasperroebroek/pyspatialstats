import numpy as np
import pytest
import xarray as xr

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.bootstrap.mean import py_bootstrap_mean
from pyspatialstats.focal import focal_mean
from pyspatialstats.results.stats import MeanResult
from pyspatialstats.rolling import rolling_mean, rolling_window
from pyspatialstats.windows import define_window

errors = ('parametric', 'bootstrap', None)


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', errors)
def test_focal_mean_center_value(rs, error, reduce):
    """Check mean at the center equals np.mean of the array."""
    a = rs.random((5, 5))
    idx = define_window(5).get_fringes(reduce, ndim=2)
    assert np.allclose(focal_mean(a, window=5, error=error, reduce=reduce).mean[idx], a.mean(), atol=0.01)


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', errors)
def test_focal_mean_full_array_values(rs, error, reduce):
    """Check focal_mean matches rolling_window for larger arrays."""
    a = rs.random((100, 100))
    ind_inner = define_window(5).get_ind_inner(reduce, ndim=2)
    np.testing.assert_allclose(
        focal_mean(a, window=5, error=error, reduce=reduce).mean[ind_inner],
        rolling_mean(a, window=5, reduce=reduce),
        atol=0.01,
    )


@pytest.mark.parametrize('error', errors)
def test_focal_mean_nan_handling(error):
    a = np.full((5, 5), 10.0)
    a[2, 2] = np.nan
    r = focal_mean(a, window=3, error=error, bootstrap_config=BootstrapConfig(n_bootstraps=123, seed=42))
    assert np.isnan(r.mean[2, 2])

    a = np.full((5, 5), 10.0)
    a[2, 1] = np.nan
    r = focal_mean(a, window=3, error=error, bootstrap_config=BootstrapConfig(n_bootstraps=123, seed=42))
    assert not np.isnan(r.mean[2, 2])


@pytest.mark.parametrize('error', errors)
@pytest.mark.parametrize('fraction', (0.0, 1.0))
def test_focal_mean_threshold(fraction, error):
    a = np.full((5, 5), 1.0)
    a[0, 0] = np.nan

    r = focal_mean(
        a,
        window=3,
        fraction_accepted=fraction,
        error='bootstrap',
        bootstrap_config=BootstrapConfig(n_bootstraps=10, seed=1),
    )

    assert np.isnan(r.mean[1, 1]) == (fraction == 1)


@pytest.mark.parametrize('error', errors)
def test_focal_mean_masking(rs, error):
    a = rs.random((5, 5))
    a[1, 2] = np.nan

    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = mask[-1, -1] = mask[0, -1] = mask[-1, 0] = False

    np.testing.assert_allclose(
        focal_mean(a, window=mask, fraction_accepted=0, reduce=True, error=error).mean,
        np.nanmean(a[mask]),
        atol=0.01,
    )


@pytest.mark.parametrize('reduce', (True, False))
def test_focal_mean_parametric_std(rs, reduce):
    """Check parametric std results match numpy std for both modes."""
    a = rs.random((5, 5))
    window = define_window(5)
    ind_inner = window.get_ind_inner(reduce, ndim=2)
    result = focal_mean(a, window=window, reduce=reduce, error='parametric').std
    expected = np.std(rolling_window(a, window=window, reduce=reduce, flatten=True), axis=-1)
    np.testing.assert_allclose(result[ind_inner], expected)


@pytest.mark.parametrize('reduce', (True, False))
def test_focal_mean_bootstrap_basic(reduce):
    a = np.arange(100).reshape(10, 10)
    window = define_window(5)
    shape = window.define_windowed_shape(reduce=reduce, a=a)
    r = focal_mean(
        a, window=window, error='bootstrap', reduce=reduce, bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=42)
    )

    assert r.std is None
    assert r.mean.shape == shape
    assert r.se.shape == shape
    assert np.nanmean(r.mean) > 0
    assert np.nanmean(r.se) > 0


def test_focal_mean_bootstrap_deterministic(rs):
    a = rs.random((10, 10))
    cfg = BootstrapConfig(n_bootstraps=100, seed=123)
    r1 = focal_mean(a, window=3, error='bootstrap', bootstrap_config=cfg)
    r2 = focal_mean(a, window=3, error='bootstrap', bootstrap_config=cfg)
    np.testing.assert_allclose(r1.mean, r2.mean, equal_nan=True)
    np.testing.assert_allclose(r1.se, r2.se, equal_nan=True)


def test_focal_mean_bootstrap_matches_py_bootstrap(rs):
    a = rs.random((3, 3))
    focal_r = focal_mean(a, window=3, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100, seed=123))
    boot_r = py_bootstrap_mean(a.flatten(), 100, seed=123)

    assert focal_r.mean[1, 1] == boot_r.mean
    assert focal_r.se[1, 1] == boot_r.se


@pytest.mark.parametrize('error, field', [('parametric', 'std'), ('bootstrap', 'se')])
@pytest.mark.parametrize('reduce', (True, False))
def test_focal_mean_parallel_consistency(rs, error, field, reduce):
    """Check linear vs parallel consistency for parametric std and bootstrap se."""
    a = rs.random((100, 100))
    cfg = BootstrapConfig(n_bootstraps=100, seed=123) if error == 'bootstrap' else None

    linear = focal_mean(a, window=5, reduce=reduce, error=error, bootstrap_config=cfg)
    parallel = focal_mean(a, window=5, reduce=reduce, error=error, bootstrap_config=cfg, chunks=75)

    np.testing.assert_allclose(linear.mean, parallel.mean, equal_nan=True, atol=0.1)
    np.testing.assert_allclose(getattr(linear, field), getattr(parallel, field), equal_nan=True, atol=0.1)


@pytest.mark.parametrize('error, field', [('parametric', 'std'), ('bootstrap', 'se')])
@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
def test_focal_mean_xarray_output(rs, error, field, reduce, chunks):
    a = xr.DataArray(rs.random((100, 100)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a)

    if error == 'parametric':
        output = MeanResult(
            mean=xr.DataArray(np.full(output_shape, np.nan)),
            std=xr.DataArray(np.full(output_shape, np.nan)),
        )
        cfg = None
    else:
        output = MeanResult(
            mean=xr.DataArray(np.full(output_shape, np.nan)),
            se=xr.DataArray(np.full(output_shape, np.nan)),
        )
        cfg = BootstrapConfig(n_bootstraps=100, seed=123)

    r = focal_mean(a, window=5, out=output, reduce=reduce, chunks=chunks, error=error, bootstrap_config=cfg)
    r_expected = focal_mean(a, window=5, out=None, reduce=reduce, chunks=chunks, error=error, bootstrap_config=cfg)

    assert id(r.mean) == id(output.mean)
    assert id(getattr(r, field)) == id(getattr(output, field))
    np.testing.assert_allclose(r.mean.values, r_expected.mean, equal_nan=True)
    np.testing.assert_allclose(getattr(r, field).values, getattr(r_expected, field), equal_nan=True)

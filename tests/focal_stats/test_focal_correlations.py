import numpy as np
import pytest
import xarray as xr
from scipy.stats import pearsonr

from pyspatialstats.focal import focal_correlation
from pyspatialstats.results.stats import CorrelationResult
from pyspatialstats.windows import define_window
from tests.focal_stats.utils import focal_correlation_simple


def test_correlation_shape(v1, v2):
    assert focal_correlation(v1, v2, window=3).c.shape == v1.shape
    assert focal_correlation(v1, v2, window=10, reduce=True).c.shape == (1, 1)


def test_correlation_values(rs):
    a1 = rs.random((5, 5))
    a2 = rs.random((5, 5))

    # Cython implementation
    r = focal_correlation(a1, a2, window=5, reduce=True, p_values=True)
    assert np.allclose(pearsonr(a1.flatten(), a2.flatten()).statistic, r.c)
    assert np.allclose(pearsonr(a1.flatten(), a2.flatten()).pvalue, r.p)

    # Local implementation
    assert np.allclose(
        pearsonr(a1.flatten(), a2.flatten())[0],
        focal_correlation_simple(a1, a2, window=5).c[2, 2],
    )


def test_correlation_values_mask(rs):
    a1 = rs.random((15, 15))
    a2 = rs.random((15, 15))
    window = define_window(rs.random((5, 7)) > 0.5)

    assert np.allclose(
        focal_correlation(a1, a2, window=window, fraction_accepted=0).c,
        focal_correlation_simple(a1, a2, window=window).c,
        equal_nan=True,
    )


def test_correlation_values_mask_reduce(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    window = define_window(rs.random((5, 5)) > 0.5)

    assert np.allclose(
        focal_correlation(a, b, window=window, fraction_accepted=0, reduce=True).c,
        focal_correlation_simple(a, b, window=window).c[2, 2],
        equal_nan=True,
    )


def test_nan_behaviour(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[2, 2] = np.nan
    assert np.allclose(focal_correlation(a, b, window=5).c, focal_correlation_simple(a, b, window=5).c, equal_nan=True)
    assert np.isnan(focal_correlation(a, b, window=5).c[2, 2])

    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[1, 1] = np.nan
    assert np.allclose(focal_correlation(a, b, window=5).c, focal_correlation_simple(a, b, window=5).c, equal_nan=True)
    assert not np.isnan(focal_correlation(a, b, window=5).c[2, 2])
    assert np.isnan(focal_correlation(a, b, fraction_accepted=1, window=5).c[2, 2])
    assert not np.isnan(focal_correlation(a, b, fraction_accepted=0, window=5).c[2, 2])


def test_correlation_dtype(rs):
    a = rs.random((5, 5)).astype(np.int32)
    b = rs.random((5, 5)).astype(np.int32)
    assert focal_correlation(a, b, window=5).c.dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    b = rs.random((5, 5)).astype(np.float64)
    assert focal_correlation(a, b, window=5).c.dtype == np.float64


def test_parallel(rs):
    a1 = rs.random((100, 100))
    a2 = rs.random((100, 100))

    linear_r = focal_correlation(a1, a2, window=5, p_values=True)
    parallel_r = focal_correlation(a1, a2, window=5, p_values=True, chunks=75)

    np.testing.assert_allclose(linear_r.c, parallel_r.c, equal_nan=True)
    np.testing.assert_allclose(linear_r.p, parallel_r.p, equal_nan=True)
    np.testing.assert_allclose(linear_r.df, parallel_r.df, equal_nan=True)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
def test_xarray_output(rs, reduce, chunks):
    a1 = xr.DataArray(rs.random((100, 100)))
    a2 = xr.DataArray(rs.random((100, 100)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a1)

    output = CorrelationResult(
        c=xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=np.float64)),
        df=xr.DataArray(np.full(output_shape, fill_value=0, dtype=np.uintp)),
        p=xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=np.float64)),
    )

    r = focal_correlation(a1, a2, window=5, out=output, reduce=reduce, chunks=chunks, p_values=True)
    r_expected = focal_correlation(a1, a2, window=5, out=None, reduce=reduce, chunks=chunks, p_values=True)

    assert id(r.c) == id(output.c)
    assert id(r.df) == id(output.df)
    assert id(r.p) == id(output.p)

    np.testing.assert_allclose(r.c.values, r_expected.c, equal_nan=True)
    np.testing.assert_allclose(r.df.values, r_expected.df, equal_nan=True)
    np.testing.assert_allclose(r.p.values, r_expected.p, equal_nan=True)

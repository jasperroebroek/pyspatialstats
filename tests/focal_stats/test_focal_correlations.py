import numpy as np
import pytest
from scipy.stats import pearsonr

from pyspatialstats.focal_stats import focal_correlation
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


def test_correlation_errors(rs):
    with pytest.raises(ValueError):
        focal_correlation(rs.random((10, 10)), rs.random((10, 10)), window=5, verbose=2)

    with pytest.raises(ValueError):
        focal_correlation(rs.random((10, 10)), rs.random((10, 10)), window=5, reduce=2)

    # not 2D
    with pytest.raises(IndexError):
        a = rs.random((10, 10, 10))
        b = rs.random((10, 10, 10))
        focal_correlation(a, b)

    # different shapes
    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        b = rs.random((10, 15))
        focal_correlation(a, b)

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        b = rs.random((10, 10))
        focal_correlation(a, b, window="x")

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        b = rs.random((10, 10))
        focal_correlation(a, b, window=1)

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        b = rs.random((10, 10))
        focal_correlation(a, b, window=11)

    # uneven window_shape is not supported
    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        b = rs.random((10, 10))
        focal_correlation(a, b, window=4)

    # Not exactly divided in reduce mode
    with pytest.raises((NotImplementedError, ValueError)):
        a = rs.random((10, 10))
        b = rs.random((10, 10))
        focal_correlation(a, b, window=4, reduce=True)

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        b = rs.random((10, 10))
        focal_correlation(a, b, fraction_accepted=-0.1)

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        b = rs.random((10, 10))
        focal_correlation(a, b, fraction_accepted=1.1)


def test_nan_behaviour(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[2, 2] = np.nan
    assert np.allclose(
        focal_correlation(a, b).c, focal_correlation_simple(a, b), equal_nan=True
    )
    assert np.isnan(focal_correlation(a, b).c[2, 2])

    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[1, 1] = np.nan
    assert np.allclose(
        focal_correlation(a, b).c, focal_correlation_simple(a, b).c, equal_nan=True
    )
    assert not np.isnan(focal_correlation(a, b).c[2, 2])
    assert np.isnan(focal_correlation(a, b, fraction_accepted=1).c[2, 2])
    assert not np.isnan(focal_correlation(a, b, fraction_accepted=0).c[2, 2])


def test_correlation_dtype(rs):
    a = rs.random((5, 5)).astype(np.int32)
    b = rs.random((5, 5)).astype(np.int32)
    assert focal_correlation(a, b).c.dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    b = rs.random((5, 5)).astype(np.float64)
    assert focal_correlation(a, b).c.dtype == np.float64

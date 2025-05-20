import numpy as np
import pytest
from scipy.stats import linregress, pearsonr

from pyspatialstats.strata_stats import strata_min
from pyspatialstats.strata_stats.strata_stats import (
    strata_correlation,
    strata_linear_regression,
    strata_max,
    strata_mean,
    strata_mean_std,
    strata_std,
)


def test_strata_min(ind, v):
    r = strata_min(ind, v)
    expected_result = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i
        values = v[mask]
        minimum = np.nanmin(values)
        expected_result[mask] = minimum

    assert np.allclose(expected_result, r, equal_nan=True)


def test_strata_min_1D():
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float64)
    with pytest.raises(IndexError):
        min_v = strata_min(ind, v)


# Test with empty arrays
def test_strata_min_empty():
    ind = np.array([[]], dtype=np.uintp)
    v = np.array([[]], dtype=np.float64)
    min_v = strata_min(ind, v)
    assert min_v.size == 0


# Test with all NaNs
def test_strata_min_all_nans():
    ind = np.ones((10, 10), dtype=np.uintp)
    v = np.full((10, 10), np.nan, dtype=np.float64)
    min_v = strata_min(ind, v)
    assert (~np.isnan(min_v)).sum() == 0


# Test with a single group
def test_strata_min_single_group():
    ind = np.ones((10, 10), dtype=np.uintp)
    v = np.arange(100, dtype=np.float64).reshape((10, 10))
    min_v = strata_min(ind, v)
    assert np.all(min_v == v.min())


def test_strata_max(ind, v):
    r = strata_max(ind, v)
    expected_result = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i
        values = v[mask]
        maximum = np.nanmax(values)
        expected_result[mask] = maximum

    assert np.allclose(expected_result, r, equal_nan=True)


def test_strata_mean(ind, v):
    r = strata_mean(ind, v)
    expected_result = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i
        values = v[mask]
        m = np.nanmean(values)
        expected_result[mask] = m

    assert np.allclose(expected_result, r, equal_nan=True)


def test_strata_std(ind, v):
    r = strata_std(ind, v)
    expected_result = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i
        values = v[mask]
        m = np.nanstd(values)
        expected_result[mask] = m

    assert np.allclose(expected_result, r, equal_nan=True)


def test_strata_mean_std(ind, v):
    r = strata_mean_std(ind, v)
    expected_mean = np.full_like(ind, np.nan, dtype=np.float64)
    expected_std = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i
        values = v[mask]
        expected_mean[mask] = np.nanmean(values)
        expected_std[mask] = np.nanstd(values)

    assert np.allclose(expected_mean, r.mean, equal_nan=True)
    assert np.allclose(expected_std, r.std, equal_nan=True)


def test_strata_correlation(ind, v1, v2):
    r = strata_correlation(ind, v1, v2)
    expected_c = np.full_like(ind, np.nan, dtype=np.float64)
    expected_p = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i

        c_v1 = v1[mask]
        c_v2 = v2[mask]

        scipy_corr = pearsonr(c_v1, c_v2)

        expected_c[mask] = scipy_corr.statistic
        expected_p[mask] = scipy_corr.pvalue

    assert np.allclose(expected_c, r.c, equal_nan=True, atol=1e-5)
    assert np.allclose(expected_p, r.p, equal_nan=True, atol=1e-5)


def test_strata_linear_regression(ind, v1, v2):
    r = strata_linear_regression(ind, v1, v2)
    expected_a = np.full_like(ind, np.nan, dtype=np.float64)
    expected_b = np.full_like(ind, np.nan, dtype=np.float64)
    expected_p_a = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i

        c_v1 = v1[mask]
        c_v2 = v2[mask]

        scipy_corr = linregress(c_v1, c_v2)

        expected_a[mask] = scipy_corr.slope
        expected_b[mask] = scipy_corr.intercept
        expected_p_a[mask] = scipy_corr.pvalue

    assert np.allclose(expected_a, r.a, equal_nan=True, atol=1e-5)
    assert np.allclose(expected_b, r.b, equal_nan=True, atol=1e-5)
    assert np.allclose(expected_p_a, r.p_a, equal_nan=True, atol=1e-5)

import numpy as np
import pytest
from scipy.stats import linregress, pearsonr

from pyspatialstats.strata import strata_min
from pyspatialstats.strata.stats import (
    strata_correlation,
    strata_count,
    strata_linear_regression,
    strata_max,
    strata_mean,
    strata_mean_std,
    strata_std,
)


@pytest.mark.parametrize(
    "sst,nps,fill_value", [
        (strata_min, np.nanmin, np.nan),
        (strata_max, np.nanmax, np.nan),
        (strata_mean, np.nanmean, np.nan),
        (strata_std, np.nanstd, np.nan),
        (strata_count, np.count_nonzero, 0)
    ]
)
def test_strata_stats(sst, nps, fill_value, ind, v):
    r = sst(ind, v)
    expected_result = np.full_like(ind, fill_value, dtype=np.float64)

    for i in range(1, int(ind.max()) + 1):
        mask = ind == i
        values = v[mask]
        expected_result[mask] = nps(values)

    assert np.allclose(expected_result, r, equal_nan=True)


@pytest.mark.parametrize(
    "sst", [
        strata_min,
        strata_max,
        strata_mean,
        strata_std,
        strata_count
    ]
)
def test_strata_stats_1D(sst):
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float64)
    with pytest.raises(IndexError):
        sst(ind, v)


@pytest.mark.parametrize(
    "sst", [
        strata_min,
        strata_max,
        strata_mean,
        strata_std,
        strata_count
    ]
)
def test_strata_min_empty(sst):
    ind = np.array([[]], dtype=np.uintp)
    v = np.array([[]], dtype=np.float64)
    min_v = sst(ind, v)
    assert min_v.size == 0


@pytest.mark.parametrize(
    "sst,f", [
        (strata_min, lambda x: (~np.isnan(x))),
        (strata_max, lambda x: (~np.isnan(x))),
        (strata_mean, lambda x: (~np.isnan(x))),
        (strata_std, lambda x: (~np.isnan(x))),
        (strata_count, lambda x: x > 0),
    ]
)
def test_strata_stats_all_nans(sst, f):
    ind = np.ones((10, 10), dtype=np.uintp)
    v = np.full((10, 10), np.nan, dtype=np.float64)
    strata_v = sst(ind, v)
    assert f(strata_v).sum() == 0


@pytest.mark.parametrize(
    "sst,nps", [
        (strata_min, np.nanmin),
        (strata_max, np.nanmax),
        (strata_mean, np.nanmean),
        (strata_std, np.nanstd),
        (strata_count, np.count_nonzero)
    ]
)
def test_strata_min_single_group(sst,nps,rs):
    ind = np.ones((10, 10), dtype=np.uintp)
    v = rs.random((10, 10))
    diff = sst(ind, v) - nps(v)
    assert np.allclose(diff, 0)


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

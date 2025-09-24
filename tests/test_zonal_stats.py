from functools import partial

import numpy as np
import pytest
from scipy.stats import linregress, pearsonr

from pyspatialstats.zonal.stats import (
    zonal_correlation,
    zonal_count,
    zonal_linear_regression,
    zonal_max,
    zonal_mean,
    zonal_min,
    zonal_std,
)


def zonal_mean_simple(*args, **kwargs):
    return zonal_mean(*args, **kwargs).mean


ZONAL_STAT_FUNCTIONS = [
    zonal_min,
    zonal_max,
    zonal_mean,
    zonal_std,
    zonal_count,
    zonal_correlation,
    zonal_linear_regression,
]

ZONAL_STAT_FUNCTIONS_SIMPLE = [
    zonal_min,
    zonal_max,
    zonal_mean_simple,
    zonal_std,
    zonal_count,
]


NPY_STAT_FUNCTIONS = {
    zonal_min: np.nanmin,
    zonal_max: np.nanmax,
    zonal_mean: np.nanmean,
    zonal_mean_simple: np.nanmean,
    zonal_std: partial(np.nanstd, ddof=1),
    zonal_count: np.count_nonzero,
    zonal_correlation: lambda x: np.corrcoef(x[:, 0], x[:, 1])[0, 1],
    zonal_linear_regression: lambda x: np.polyfit(x[:, 0], x[:, 1], 1)[0],
}


@pytest.mark.parametrize('sst', ZONAL_STAT_FUNCTIONS_SIMPLE)
def test_zonal_stats(sst, ind, v):
    nps = NPY_STAT_FUNCTIONS[sst]

    r = sst(ind, v)
    expected_result = np.zeros_like(ind, dtype=np.float64)

    for i in range(int(ind.max()) + 1):
        mask = ind == i
        values = v[mask]
        expected_result[mask] = nps(values)

    assert np.allclose(expected_result, r, equal_nan=True)


@pytest.mark.parametrize(
    'sst',
    [
        zonal_min,
        zonal_max,
        zonal_mean,
        zonal_std,
        zonal_count,
    ],
)
def test_zonal_min_empty(sst):
    ind = np.array([[]], dtype=np.uintp)
    v = np.array([[]], dtype=np.float64)
    min_v = sst(ind, v)
    assert min_v.size == 0


@pytest.mark.parametrize(
    'sst,f',
    [
        (zonal_min, lambda x: (~np.isnan(x))),
        (zonal_max, lambda x: (~np.isnan(x))),
        (zonal_mean_simple, lambda x: (~np.isnan(x))),
        (zonal_std, lambda x: (~np.isnan(x))),
        (zonal_count, lambda x: x > 0),
    ],
)
def test_zonal_stats_all_nans(sst, f):
    ind = np.ones((10, 10), dtype=np.uintp)
    v = np.full((10, 10), np.nan, dtype=np.float64)
    zonal_v = sst(ind, v)
    assert f(zonal_v).sum() == 0


@pytest.mark.parametrize('sst', ZONAL_STAT_FUNCTIONS_SIMPLE)
def test_zonal_min_single_group(sst, rs):
    nps = NPY_STAT_FUNCTIONS[sst]
    ind = np.ones((10, 10), dtype=np.uintp)
    v = rs.random((10, 10))
    diff = sst(ind, v) - nps(v)
    assert np.allclose(diff, 0)


def test_zonal_correlation(ind, v1, v2):
    r = zonal_correlation(ind, v1, v2)
    expected_c = np.full_like(ind, np.nan, dtype=np.float64)
    expected_p = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(int(ind.max()) + 1):
        mask = ind == i

        c_v1 = v1[mask]
        c_v2 = v2[mask]

        scipy_corr = pearsonr(c_v1, c_v2, alternative='two-sided')

        expected_c[mask] = scipy_corr.statistic
        expected_p[mask] = scipy_corr.pvalue

    assert np.allclose(expected_c, r.c, equal_nan=True, atol=1e-5)
    assert np.allclose(expected_p, r.p, equal_nan=True, atol=1e-5)


def test_zonal_linear_regression(ind, v1, v2):
    r = zonal_linear_regression(ind, v1, v2)
    expected_intercept = np.full_like(ind, np.nan, dtype=np.float64)
    expected_slope = np.full_like(ind, np.nan, dtype=np.float64)
    expected_p_slope = np.full_like(ind, np.nan, dtype=np.float64)

    for i in range(int(ind.max()) + 1):
        mask = ind == i

        c_v1 = v1[mask]
        c_v2 = v2[mask]

        scipy_corr = linregress(c_v1, c_v2)

        expected_slope[mask] = scipy_corr.slope
        expected_intercept[mask] = scipy_corr.intercept
        expected_p_slope[mask] = scipy_corr.pvalue

    assert np.allclose(expected_intercept, r.beta[:, :, 0], equal_nan=True, atol=1e-5)
    assert np.allclose(expected_slope, r.beta[:, :, 1], equal_nan=True, atol=1e-5)
    assert np.allclose(expected_p_slope, r.p[:, :, 1], equal_nan=True, atol=1e-5)

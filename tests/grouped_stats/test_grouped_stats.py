from functools import partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest

from pyspatialstats.grouped import grouped_correlation as grouped_correlation_internal
from pyspatialstats.grouped import grouped_count, grouped_max, grouped_min, grouped_std, grouped_sum
from pyspatialstats.grouped import grouped_linear_regression as grouped_linear_regression_internal
from pyspatialstats.grouped import grouped_mean as grouped_mean_internal


def double_v(v):
    rs = np.random.RandomState(0)
    if (~np.isnan(v)).sum() == 0:
        v1 = np.full(v.shape, np.nan)
        v2 = np.full(v.shape, np.nan)
    else:
        v1 = rs.normal(size=v.shape)
        v2 = rs.normal(size=v.shape)
    return v1, v2


def grouped_mean(ind, v, filtered=True, **kwargs):
    r = grouped_mean_internal(ind, v, filtered=filtered, **kwargs)
    if filtered:
        return r[['mean']]
    else:
        return r.mean


def grouped_correlation(ind, v, filtered=True, **kwargs):
    v1, v2 = double_v(v)
    r = grouped_correlation_internal(ind, v1, v2, filtered=filtered, **kwargs)
    if filtered:
        return r[['c']]
    else:
        return r.c


def grouped_linear_regression(ind, v, filtered=True, **kwargs):
    v1, v2 = double_v(v)
    v1 = v1[..., None]
    r = grouped_linear_regression_internal(ind, v1, v2, filtered=filtered, **kwargs)
    if filtered:
        return r[['beta_0', 'beta_1']]
    return r.beta


def numpy_correlation(v, mask: Optional[np.ndarray] = None):
    v1, v2 = double_v(v)
    if mask is not None:
        v1 = v1[mask]
        v2 = v2[mask]
    return np.corrcoef(v1, v2)[0, 1]


def numpy_regression(v, mask: Optional[np.ndarray] = None):
    v1, v2 = double_v(v)
    if mask is not None:
        if mask.size == 0:
            return np.array([np.nan, np.nan])
        v1 = v1[mask]
        v2 = v2[mask]
    return np.polyfit(v1, v2, 1)[::-1]


def numpy_func(v, fun: Callable, mask: Optional[np.ndarray] = None, **kwargs):
    if mask is not None:
        v = v[mask]
    return fun(v, **kwargs)


GROUPED_STAT_FUNCTIONS = [
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_std,
    grouped_count,
    grouped_sum,
    grouped_correlation,
    grouped_linear_regression,
]


NPY_STAT_FUNCTIONS = {
    grouped_count: partial(numpy_func, fun=np.count_nonzero),
    grouped_max: partial(numpy_func, fun=np.max),
    grouped_mean: partial(numpy_func, fun=np.mean),
    grouped_min: partial(numpy_func, fun=np.min),
    grouped_std: partial(numpy_func, fun=np.std, ddof=1),
    grouped_sum: partial(numpy_func, fun=np.sum),
    grouped_correlation: numpy_correlation,
    grouped_linear_regression: numpy_regression,
}


CHECK_NULL = {
    grouped_count: lambda x: x == 0,
    grouped_sum: lambda x: x == 0,
}

DEFAULT_CHECK_NULL = lambda x: np.isnan(x)  # noqa


@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
def test_grouped_stats_single_group(gs):
    np_gs = NPY_STAT_FUNCTIONS[gs]
    ind = np.ones(10, dtype=np.uintp)
    v = np.arange(1, 11, dtype=np.float64)
    r = gs(ind, v, filtered=False)
    assert np.allclose(r[1], np_gs(v), atol=1e-5)


@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
@pytest.mark.parametrize('chunks', (None, 10))
def test_grouped_stats(ind, v, gs, chunks):
    np_gs = NPY_STAT_FUNCTIONS[gs]
    r = gs(ind, v, chunks=chunks, filtered=False)

    for i in range(int(ind.max()) + 1):
        expected_r = np_gs(v, mask=ind == i)
        assert np.allclose(r[i], expected_r, atol=1e-5)


@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
def test_grouped_stats_pd(ind, v, gs):
    np_gs = NPY_STAT_FUNCTIONS[gs]
    result_df = gs(ind, v)
    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        mask = ind == i
        expected_r = np_gs(v, mask=mask)
        assert np.allclose(np.asarray(result_df.loc[i]), expected_r, atol=1e-5)


@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
def test_grouped_stats_empty(gs):
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float64)
    r = gs(ind, v, filtered=False)
    assert r.size == 0


@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
def test_grouped_stats_all_nans(gs):
    ind = np.ones(10, dtype=np.uintp)
    v = np.full(10, np.nan, dtype=np.float64)
    r = gs(ind, v, filtered=True)
    assert r.empty


@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
def test_grouped_stats_large_values(gs):
    np_gs = NPY_STAT_FUNCTIONS[gs]
    ind = np.array([1, 1, 1, 1, 1, 1], dtype=np.uintp)
    v = np.array([1e38, 1e-38, 1e38, 1e-38, 1e38, 1e-38], dtype=np.float64)
    r = gs(ind, v, filtered=False)
    assert np.allclose(r[1], np_gs(v), atol=1e-5)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('filtered', (True, False))
@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
def test_parallel_against_linear(rs, gs, chunks, filtered):
    v = rs.random((100, 100))
    ind = rs.integers(1, 15, size=(100, 100))

    r = gs(ind=ind, v=v, chunks=chunks, filtered=filtered)
    r_expected = gs(ind=ind, v=v, chunks=None, filtered=filtered)

    np.testing.assert_allclose(r, r_expected, equal_nan=True)


@pytest.mark.parametrize('gs', GROUPED_STAT_FUNCTIONS)
def test_different_sized_cy_result_objects(rs, gs):
    a = rs.random((100, 100))
    ind = np.ones((100, 100))

    ind[:50, :] += 1

    r = gs(ind=ind, v=a, chunks=50, filtered=False)
    r_expected = gs(ind=ind, v=a, chunks=None, filtered=False)

    np.testing.assert_allclose(r, r_expected, equal_nan=True)

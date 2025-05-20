import numpy as np
import pandas as pd
import pytest

from pyspatialstats.grouped_stats import (
    grouped_max,
    grouped_max_pd,
    grouped_mean,
    grouped_mean_pd,
    grouped_mean_std_pd,
    grouped_min,
    grouped_min_pd,
    grouped_std,
    grouped_std_pd,
)


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (grouped_max, np.max),
        (grouped_mean, np.mean),
        (grouped_min, np.min),
        (grouped_std, np.std),
    ],
)
def test_grouped_stats(ind, v, fs, np_fs):
    r = fs(ind, v)

    for i in range(1, int(ind.max()) + 1):
        values_in_group = v[ind == i]
        expected_r = np_fs(values_in_group)
        assert np.isclose(r[i], expected_r, atol=1e-5)

    assert np.isnan(r[0])


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (grouped_max_pd, np.max),
        (grouped_mean_pd, np.mean),
        (grouped_min_pd, np.min),
        (grouped_std_pd, np.std),
    ],
)
def test_grouped_stats_pd(ind, v, fs, np_fs):
    result_df = fs(ind, v)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_r = np_fs(values_in_group)
        assert np.isclose(result_df.loc[i, result_df.columns[0]], expected_r, atol=1e-5)

    assert 0 not in result_df.index


def test_grouped_mean_std_pd(ind, v):
    result_df = grouped_mean_std_pd(ind, v)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_mean = np.nanmean(values_in_group)
        expected_std = np.nanstd(values_in_group)
        assert np.isclose(result_df.loc[i, "mean"], expected_mean, atol=1e-5)
        assert np.isclose(result_df.loc[i, "std"], expected_std, atol=1e-5)

    assert 0 not in result_df.index


@pytest.mark.parametrize(
    "fs",
    [grouped_max, grouped_mean, grouped_min, grouped_std],
)
def test_grouped_stats_empty(fs):
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float64)
    r = fs(ind, v)
    assert r.size == 1
    assert np.isnan(r[0])


@pytest.mark.parametrize(
    "fs",
    [grouped_max, grouped_mean, grouped_min, grouped_std],
)
def test_grouped_stats_all_nans(fs):
    ind = np.ones(10, dtype=np.uintp)
    v = np.full(10, np.nan, dtype=np.float64)
    r = fs(ind, v)
    assert np.all(np.isnan(r))


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (grouped_max, np.max),
        (grouped_mean, np.mean),
        (grouped_min, np.min),
        (grouped_std, np.std),
    ],
)
def test_grouped_stats_single_group(fs, np_fs):
    ind = np.ones(10, dtype=np.uintp)
    v = np.arange(10, dtype=np.float64)
    r = fs(ind, v)
    assert np.isclose(r[1], np_fs(v), atol=1e-5)


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (grouped_max, np.max),
        (grouped_mean, np.mean),
        (grouped_min, np.min),
        (grouped_std, np.std),
    ],
)
def test_grouped_stats_large_values(fs, np_fs):
    ind = np.array([1, 1], dtype=np.uintp)
    v = np.array([1e38, 1e-38], dtype=np.float64)
    r = fs(ind, v)
    assert np.isclose(r[1], np_fs(v), atol=1e-5)

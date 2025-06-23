import numpy as np
import pytest

from pyspatialstats.grouped.stats_bootstrap import grouped_mean_bootstrap
from pyspatialstats.random.random import RandomInts
from tests.bootstrap.np_stats import np_bootstrap_mean


@pytest.mark.parametrize(
    "fs,np_fs,stat",
    [
        (grouped_mean_bootstrap, np_bootstrap_mean, "mean"),
    ],
)
def test_grouped_stats_one_group(ind, v, fs, np_fs, stat):
    ind = np.ones_like(ind)
    r = fs(ind, v, n_bootstraps=1000, seed=0)

    expected_r = np_fs(v.flatten(), n_bootstraps=1000, seed=0)
    assert np.isclose(getattr(r, stat)[1], getattr(expected_r, stat))
    assert np.isclose(r.se[1], expected_r.se)

    assert np.isnan(r.mean[0])


@pytest.mark.parametrize(
    "fs,np_fs,stat",
    [
        (grouped_mean_bootstrap, np_bootstrap_mean, "mean"),
    ],
)
def test_grouped_stats(ind, v, fs, np_fs, stat):
    seed = 0
    n_bootstraps = 1000

    r = fs(ind, v, n_bootstraps=n_bootstraps, seed=seed)
    rng = RandomInts(seed)

    for i in range(int(ind.max()) + 1):
        expected_r = np_fs(v[ind == i], n_bootstraps=n_bootstraps, rng=rng)
        assert np.isclose(getattr(r, stat)[i], getattr(expected_r, stat), atol=0.01)
        assert np.isclose(r.se[i], expected_r.se, atol=0.01)


# todo
# @pytest.mark.parametrize(
#     "fs,np_fs",
#     [
#         (grouped_max_pd, np.max),
#         (grouped_mean_pd, np.mean),
#         (grouped_min_pd, np.min),
#         (grouped_std_pd, np.std),
#     ],
# )
# def test_grouped_stats_pd(ind, v, fs, np_fs):
#     result_df = fs(ind, v)
#
#     assert isinstance(result_df, pd.DataFrame)
#
#     for i in result_df.index:
#         values_in_group = v[ind == i]
#         expected_r = np_fs(values_in_group)
#         assert np.isclose(result_df.loc[i, result_df.columns[0]], expected_r, atol=1e-5)



@pytest.mark.parametrize("fs,stat", [(grouped_mean_bootstrap, "mean")])
def test_grouped_stats_empty(fs, stat):
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float64)
    r = fs(ind, v, n_bootstraps=2, seed=0)
    assert getattr(r, stat).size == 1
    assert np.isnan(getattr(r, stat)[0])


@pytest.mark.parametrize("fs,stat", [(grouped_mean_bootstrap, "mean")])
def test_grouped_stats_all_nans(fs, stat):
    ind = np.ones(10, dtype=np.uintp)
    v = np.full(10, np.nan, dtype=np.float64)
    r = fs(ind, v, n_bootstraps=2, seed=0)
    assert np.all(np.isnan(getattr(r, stat)))

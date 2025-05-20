import numpy as np

from pyspatialstats.grouped_stats import define_max_ind, grouped_count, grouped_count_pd


def test_count(ind):
    v = np.ones_like(ind, dtype=np.float64)
    max_ind = define_max_ind(ind)

    count_v = grouped_count(ind, v)

    for i in range(1, max_ind + 1):
        assert count_v[i] == (ind == i).sum()

    assert count_v[0] == 0


def test_count_all_group_0():
    # Edge case: all elements in group 0
    ind = np.zeros((10, 10), dtype=np.uintp)
    count = grouped_count(ind, v=np.ones_like(ind))
    assert count[0] == 0


def test_count_all_group_1():
    # Edge case: all elements in group 1
    ind = np.ones((10, 10), dtype=np.uintp)
    count = grouped_count(ind, v=np.ones_like(ind))
    assert count[1] == ind.size


def test_count_nans(rs, ind):
    v = rs.random(ind.shape)
    for i in (0, 2, 8):
        v[i, i] = np.nan

    max_ind = define_max_ind(ind)

    count_v = grouped_count(ind, v)

    # Check that the counts are correct
    for i in range(1, max_ind + 1):
        assert count_v[i] == np.logical_and(ind == i, ~np.isnan(v)).sum()

    # Check that counts for group 0 are zero
    assert count_v[0] == 0


def test_count_df(rs, ind):
    v = rs.random(ind.shape)
    for i in (0, 2, 8):
        v[i, i] = np.nan

    max_ind = define_max_ind(ind)

    df = grouped_count_pd(ind, v)
    assert df.columns == ["count"]

    # Check that the counts are correct
    for i in range(1, max_ind + 1):
        assert df.loc[i].item() == np.logical_and(ind == i, ~np.isnan(v)).sum()

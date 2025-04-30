import numpy as np
import pytest
from scipy.stats import pearsonr
import pandas as pd

from pyspatialstats.grouped_stats import (
    grouped_correlation,
    grouped_correlation_pd,
)

rs = np.random.RandomState(0)


@pytest.fixture
def ind():
    return rs.randint(0, 5, size=(10, 10), dtype=np.uintp)


@pytest.fixture
def v1():
    return rs.rand(10, 10)


@pytest.fixture
def v2():
    return rs.rand(10, 10)


def test_grouped_correlation_with_example_data(v1, v2):
    ind = np.ones_like(v1, dtype=np.uintp)
    result = grouped_correlation(ind, v1, v2)
    assert result.c[1] == pytest.approx(np.corrcoef(v1.ravel(), v2.ravel())[0, 1], rel=1e-4)


def test_grouped_correlation_with_empty_arrays():
    ind = np.array([[]], dtype=np.uintp)
    v1 = np.array([[]], dtype=np.float32)
    v2 = np.array([[]], dtype=np.float32)
    result = grouped_correlation(ind, v1, v2)
    assert np.allclose(result.c, [np.nan], equal_nan=True)
    assert np.allclose(result.p, [np.nan], equal_nan=True)


def test_grouped_correlation_with_different_length_arrays():
    ind = np.array([[1, 1, 2, 2]], dtype=np.uintp)
    v1 = np.array([[10.0, 15.0, 20.0]], dtype=np.float32)
    v2 = np.array([[5.0, 10.0, 15.0, 20.0]], dtype=np.float32)
    with pytest.raises(IndexError):
        grouped_correlation(ind, v1, v2)


def test_grouped_correlation_against_scipy():
    ind = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=np.uintp)
    v1 = rs.rand(9).reshape(1, 9)
    v2 = rs.rand(9).reshape(1, 9)

    # Call the Cython function
    result = grouped_correlation(ind, v1, v2)

    # Iterate over unique groups and test each group separately
    unique_groups = np.unique(ind)
    for group in unique_groups:
        group_indices = ind == group
        group_v1 = v1[group_indices]
        group_v2 = v2[group_indices]

        # Calculate expected result using NumPy/SciPy
        expected_result = pearsonr(group_v1, group_v2)

        # Compare the results.py
        assert result.c[group] == pytest.approx(expected_result.statistic, rel=1e-4)
        assert result.p[group] == pytest.approx(expected_result.pvalue, rel=1e-4)


def test_grouped_correlation_all_values_in_single_group():
    ind = np.array([[1, 1, 1]], dtype=np.uintp)
    v1 = np.array([[10.0, 15.0, 20.0]], dtype=np.float32)
    v2 = np.array([[5.0, 10.0, 15.0]], dtype=np.float32)

    # Call the Cython function
    result = grouped_correlation(ind, v1, v2)

    # Calculate expected result using NumPy/SciPy
    expected_result = np.corrcoef(v1, v2)[1, 0]

    # Compare the results.py
    np.testing.assert_allclose(result.c[1], expected_result, rtol=1e-5, atol=1e-8)


def test_grouped_correlation_with_nan_values():
    ind = np.ones(10).reshape(1, 10)
    v1 = rs.rand(10).reshape(1, 10)
    v2 = rs.rand(10).reshape(1, 10)
    v1[0, 1] = np.nan
    v2[0, 2] = np.nan

    mask = ~np.isnan(v1) & ~np.isnan(v2)

    # Call the Cython function
    result = grouped_correlation(ind, v1, v2)

    # Calculate expected result using NumPy/SciPy
    expected_result = np.corrcoef(v1[mask], v2[mask])[1, 0]

    # Compare the results.py
    np.testing.assert_allclose(result.c[1], expected_result, rtol=1e-5)


def test_grouped_correlation_one_group_random(v1, v2):
    ind = np.ones_like(v1, dtype=np.uintp)

    # Call the Cython function
    result = grouped_correlation(ind, v1, v2)

    # Calculate expected result using NumPy/SciPy
    expected_result = np.corrcoef(v1.flatten(), v2.flatten())[1, 0]

    # Compare the results.py
    np.testing.assert_allclose(result.c[1], expected_result, rtol=1e-4)


def test_grouped_correlation(ind, v1, v2):
    r = grouped_correlation(ind, v1, v2)

    for i in range(1, int(ind.max()) + 1):
        values_in_group_1 = v1[ind == i]
        values_in_group_2 = v2[ind == i]
        expected_correlation = np.corrcoef(values_in_group_1, values_in_group_2)[0, 1]
        assert np.isclose(r.c[i], expected_correlation, atol=1e-5)

    # Assuming the function returns NaN for groups with no entries
    assert np.isnan(r.c[0])


def test_grouped_correlation_pd(ind, v1, v2):
    result_df = grouped_correlation_pd(ind, v1, v2)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group_1 = v1[ind == i]
        values_in_group_2 = v2[ind == i]
        expected_correlation = np.corrcoef(values_in_group_1, values_in_group_2)[0, 1]
        assert np.isclose(
            result_df.loc[i, "c"], expected_correlation, atol=1e-5
        )

    assert 0 not in result_df.index


def test_grouped_correlation_all_nans():
    ind = np.ones(10, dtype=np.uintp)
    v1 = np.full(10, np.nan, dtype=np.float32)
    v2 = np.full(10, np.nan, dtype=np.float32)
    r = grouped_correlation(ind, v1, v2)

    # Assuming the function returns NaN for groups with no valid (non-NaN) entries
    assert np.isnan(r.c[1])


def test_grouped_correlation_single_group():
    ind = np.ones(10, dtype=np.uintp)
    v1 = np.arange(10, dtype=np.float32)
    v2 = np.arange(10, dtype=np.float32)
    r = grouped_correlation(ind, v1, v2)
    expected_correlation = np.corrcoef(v1, v2)[0, 1]
    assert np.isclose(r.c[1], expected_correlation, atol=1e-5)


def test_grouped_correlation_non_contiguous_groups():
    ind = np.array([1, 3, 5], dtype=np.uintp)
    v1 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    v2 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    r = grouped_correlation(ind, v1, v2)
    assert np.isnan(r.c[1])
    assert np.isnan(r.c[3])
    assert np.isnan(r.c[5])


def test_grouped_correlation_large_values():
    ind = np.array([1, 1, 1], dtype=np.uintp)
    v1 = np.array([1e10, 1e-10, -1e10], dtype=np.float32)
    v2 = np.array([1e10, -1e-10, 1e10], dtype=np.float32)
    r = grouped_correlation(ind, v1, v2)
    expected_correlation = np.corrcoef(v1, v2)[0, 1]
    assert np.isclose(r.c[1], expected_correlation, atol=1e-5)


def test_grouped_correlation_non_overlapping_data():
    ind = np.ones(10)
    v1 = rs.rand(10)
    v2 = rs.rand(10)

    v1[:5] = np.nan
    v2[5:] = np.nan

    r_df = grouped_correlation_pd(ind, v1, v2)
    assert r_df.size == 0

    r = grouped_correlation(ind, v1, v2)
    assert r.c.size == 2

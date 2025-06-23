import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from pyspatialstats.grouped import (
    grouped_correlation,
    grouped_correlation_pd,
)


def test_grouped_correlation_with_example_data(v1, v2):
    ind = np.ones_like(v1, dtype=np.uintp)
    result = grouped_correlation(ind, v1, v2)
    assert result.c[1] == pytest.approx(
        np.corrcoef(v1.ravel(), v2.ravel())[0, 1], rel=1e-4
    )


def test_grouped_correlation_with_empty_arrays():
    ind = np.array([[]], dtype=np.uintp)
    v1 = np.array([[]], dtype=np.float64)
    v2 = np.array([[]], dtype=np.float64)
    result = grouped_correlation(ind, v1, v2)
    assert np.allclose(result.c, [np.nan], equal_nan=True)
    assert np.allclose(result.p, [np.nan], equal_nan=True)


def test_grouped_correlation_with_different_length_arrays():
    ind = np.array([[1, 1, 2, 2]], dtype=np.uintp)
    v1 = np.array([[10.0, 15.0, 20.0]], dtype=np.float64)
    v2 = np.array([[5.0, 10.0, 15.0, 20.0]], dtype=np.float64)
    with pytest.raises(IndexError):
        grouped_correlation(ind, v1, v2)


def test_grouped_correlation_against_scipy(rs):
    ind = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=np.uintp)
    a = rs.random((1, 9))
    b = rs.random((1, 9))

    # Call the Cython function
    result = grouped_correlation(ind, a, b)

    # Iterate over unique groups and test each group separately
    unique_groups = np.unique(ind)
    for group in unique_groups:
        group_indices = ind == group
        group_v1 = a[group_indices]
        group_v2 = b[group_indices]

        # Calculate expected cy_result using NumPy/SciPy
        expected_result = pearsonr(group_v1, group_v2)

        # Compare the focal.py
        assert result.c[group] == pytest.approx(expected_result.statistic, rel=1e-4)
        assert result.p[group] == pytest.approx(expected_result.pvalue, rel=1e-4)


def test_grouped_correlation_all_values_in_single_group():
    ind = np.array([[1, 1, 1]], dtype=np.uintp)
    v1 = np.array([[10.0, 15.0, 20.0]], dtype=np.float64)
    v2 = np.array([[5.0, 10.0, 15.0]], dtype=np.float64)

    # Call the Cython function
    result = grouped_correlation(ind, v1, v2)

    # Calculate expected cy_result using NumPy/SciPy
    expected_result = np.corrcoef(v1, v2)[1, 0]

    # Compare the focal.py
    np.testing.assert_allclose(result.c[1], expected_result, rtol=1e-5, atol=1e-8)


def test_grouped_correlation_with_nan_values(rs):
    ind = np.ones((1, 10))

    a = rs.random((1, 10))
    b = rs.random((1, 10))

    a[0, 1] = np.nan
    b[0, 2] = np.nan

    mask = ~np.isnan(a) & ~np.isnan(b)

    # Call the Cython function
    result = grouped_correlation(ind, a, b)

    # Calculate expected cy_result using NumPy/SciPy
    expected_result = np.corrcoef(a[mask], b[mask])[1, 0]

    # Compare the focal.py
    np.testing.assert_allclose(result.c[1], expected_result, rtol=1e-5)


def test_grouped_correlation_one_group_random(v1, v2):
    ind = np.ones_like(v1, dtype=np.uintp)

    # Call the Cython function
    result = grouped_correlation(ind, v1, v2)

    # Calculate expected cy_result using NumPy/SciPy
    expected_result = np.corrcoef(v1.flatten(), v2.flatten())[1, 0]

    # Compare the focal.py
    np.testing.assert_allclose(result.c[1], expected_result, rtol=1e-4)


def test_grouped_correlation(ind, v1, v2):
    r = grouped_correlation(ind, v1, v2)

    for i in range(int(ind.max()) + 1):
        values_in_group_1 = v1[ind == i]
        values_in_group_2 = v2[ind == i]
        expected_correlation = np.corrcoef(values_in_group_1, values_in_group_2)[0, 1]
        assert np.isclose(r.c[i], expected_correlation, atol=1e-5)


def test_grouped_correlation_pd(ind, v1, v2):
    result_df = grouped_correlation_pd(ind, v1, v2)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group_1 = v1[ind == i]
        values_in_group_2 = v2[ind == i]
        expected_correlation = np.corrcoef(values_in_group_1, values_in_group_2)[0, 1]
        assert np.isclose(result_df.loc[i, "c"], expected_correlation, atol=1e-5)


def test_grouped_correlation_all_nans():
    ind = np.ones(10, dtype=np.uintp)
    v1 = np.full(10, np.nan, dtype=np.float64)
    v2 = np.full(10, np.nan, dtype=np.float64)
    r = grouped_correlation(ind, v1, v2)

    # Assuming the function returns NaN for groups with no valid (non-NaN) entries
    assert np.isnan(r.c[1])


def test_grouped_correlation_single_group():
    ind = np.ones(10, dtype=np.uintp)
    v1 = np.arange(10, dtype=np.float64)
    v2 = np.arange(10, dtype=np.float64)
    r = grouped_correlation(ind, v1, v2)
    expected_correlation = np.corrcoef(v1, v2)[0, 1]
    assert np.isclose(r.c[1], expected_correlation, atol=1e-5)


def test_grouped_correlation_non_contiguous_groups():
    ind = np.array([1, 3, 5], dtype=np.uintp)
    v1 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    v2 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    r = grouped_correlation(ind, v1, v2)
    assert np.isnan(r.c[1])
    assert np.isnan(r.c[3])
    assert np.isnan(r.c[5])


def test_grouped_correlation_large_values():
    ind = np.array([1, 1, 1], dtype=np.uintp)
    v1 = np.array([1e10, 1e-10, -1e10], dtype=np.float64)
    v2 = np.array([1e10, -1e-10, 1e10], dtype=np.float64)
    r = grouped_correlation(ind, v1, v2)
    expected_correlation = np.corrcoef(v1, v2)[0, 1]
    assert np.isclose(r.c[1], expected_correlation, atol=1e-5)


def test_grouped_correlation_non_overlapping_data(rs):
    ind = np.ones(10)
    a = rs.random(10)
    b = rs.random(10)

    a[:5] = np.nan
    b[5:] = np.nan

    r_df = grouped_correlation_pd(ind, a, b)
    assert r_df.size == 0

    r = grouped_correlation(ind, a, b)
    assert r.c.size == 2

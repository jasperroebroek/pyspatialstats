import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import linregress

from pyspatialstats.grouped import (
    grouped_linear_regression,
    grouped_linear_regression_pd,
)


def test_py_grouped_linear_regression_with_example_data():
    ind = np.array([[1, 1, 1, 1]], dtype=np.uintp)
    v1 = np.array([[10.0, 15.0, 20.0, 25.0]], dtype=np.float64)
    v2 = np.array([[26.1, 38.6, 51.1, 63.6]], dtype=np.float64)
    result = grouped_linear_regression(ind, v1, v2)

    assert np.isnan(result.a[0])
    assert np.isnan(result.b[0])
    assert np.isnan(result.se_a[0])
    assert np.isnan(result.se_b[0])
    assert np.isnan(result.t_a[0])
    assert np.isnan(result.t_b[0])
    assert np.isnan(result.p_a[0])
    assert np.isnan(result.p_b[0])

    assert result.a[1] == pytest.approx(2.5, rel=1e-5)
    assert result.b[1] == pytest.approx(1.1, rel=1e-5)
    assert np.isclose(result.se_a[1], 0, atol=1e-5)
    assert np.isclose(result.se_b[1], 0, atol=1e-5)
    assert np.isclose(result.p_a[1], 0, atol=1e-5)
    assert np.isclose(result.p_b[1], 0, atol=1e-5)


def test_py_grouped_linear_regression_with_empty_arrays():
    ind = np.array([], dtype=np.uintp)
    v1 = np.array([], dtype=np.float64)
    v2 = np.array([], dtype=np.float64)
    result = grouped_linear_regression(ind, v1, v2)
    assert np.isnan(result.a[0])
    assert np.isnan(result.b[0])
    assert np.isnan(result.se_a[0])
    assert np.isnan(result.se_b[0])
    assert np.isnan(result.t_a[0])
    assert np.isnan(result.t_b[0])
    assert np.isnan(result.p_a[0])
    assert np.isnan(result.p_b[0])


def test_py_grouped_correlation_with_different_length_arrays():
    ind = np.array([1, 1, 2, 2], dtype=np.uintp)
    a = np.array([10.0, 15.0, 20.0], dtype=np.float64)
    b = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float64)
    with pytest.raises(IndexError, match="Arrays are not all of the same shape"):
        grouped_linear_regression(ind, a, b)


def test_py_grouped_correlation_against_scipy(rs):
    ind = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.uintp).reshape(3, 3)
    a = rs.random((3, 3))
    b = rs.random((3, 3))

    # Call the Cython function
    result = grouped_linear_regression(ind, a, b)

    # Iterate over unique groups and test each group separately
    unique_groups = np.unique(ind)
    for group in unique_groups:
        group_indices = ind == group
        group_v1 = a[group_indices]
        group_v2 = b[group_indices]

        # Calculate expected result using NumPy/SciPy
        expected_result = linregress(group_v1, group_v2)

        # Compare the results.py
        assert result.a[group] == pytest.approx(expected_result.slope, rel=1e-4)
        assert result.b[group] == pytest.approx(expected_result.intercept, rel=1e-4)
        assert result.p_a[group] == pytest.approx(expected_result.pvalue, rel=1e-4)
        assert result.se_a[group] == pytest.approx(expected_result.stderr, rel=1e-4)
        assert result.se_b[group] == pytest.approx(
            expected_result.intercept_stderr, rel=1e-4
        )


def test_py_grouped_correlation_against_statsmodels(rs):
    ind = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.uintp)
    a = rs.random(9)
    b = rs.random(9)

    # Call the Cython function
    result = grouped_linear_regression(ind, a, b)

    # Iterate over unique groups and test each group separately
    unique_groups = np.unique(ind)
    for group in unique_groups:
        group_indices = ind == group
        group_a = a[group_indices]
        group_b = b[group_indices]

        # statsmodels implementation
        group_a_with_intercept = sm.add_constant(group_a)
        model = sm.OLS(group_b, group_a_with_intercept)
        result_statsmodels = model.fit()

        # Compare coefficients
        np.testing.assert_allclose(
            result.a[group], result_statsmodels.params[1], rtol=1e-4
        )
        np.testing.assert_allclose(
            result.b[group], result_statsmodels.params[0], rtol=1e-4
        )

        # Compare standard errors
        np.testing.assert_allclose(
            result.se_a[group], result_statsmodels.bse[1], rtol=1e-4
        )
        np.testing.assert_allclose(
            result.se_b[group], result_statsmodels.bse[0], rtol=1e-4
        )

        # Compare t-values
        np.testing.assert_allclose(
            result.t_a[group], result_statsmodels.tvalues[1], rtol=1e-4
        )
        np.testing.assert_allclose(
            result.t_b[group], result_statsmodels.tvalues[0], rtol=1e-4
        )

        # Compare p-values
        np.testing.assert_allclose(
            result.p_a[group], result_statsmodels.pvalues[1], rtol=1e-4
        )
        np.testing.assert_allclose(
            result.p_b[group], result_statsmodels.pvalues[0], rtol=1e-4
        )


def test_py_grouped_correlation_with_nan_values(rs):
    ind = np.ones(10)
    a = rs.random(10)
    b = rs.random(10)
    a[1] = np.nan
    b[2] = np.nan

    mask = ~np.isnan(a) & ~np.isnan(b)

    # Call the Cython function
    result = grouped_linear_regression(ind, a, b)

    # Calculate expected result using NumPy/SciPy
    expected_result = linregress(a[mask], b[mask])

    assert result.a[1] == pytest.approx(expected_result.slope, rel=1e-4)
    assert result.b[1] == pytest.approx(expected_result.intercept, rel=1e-4)
    assert result.p_a[1] == pytest.approx(expected_result.pvalue, rel=1e-4)
    assert result.se_a[1] == pytest.approx(expected_result.stderr)
    assert result.se_b[1] == pytest.approx(expected_result.intercept_stderr)


def test_grouped_correlation_pd(ind, v1, v2):
    result_df = grouped_linear_regression_pd(ind, v1, v2)

    assert isinstance(result_df, pd.DataFrame)

    assert np.all(
        result_df.columns == ["a", "b", "se_a", "se_b", "t_a", "t_b", "p_a", "p_b"]
    )

    for i in result_df.index:
        values_in_group_1 = v1[ind == i]
        values_in_group_2 = v2[ind == i]
        expected_result = linregress(values_in_group_1, values_in_group_2)

        assert np.isclose(result_df.loc[i, "a"], expected_result.slope, atol=1e-5)
        assert np.isclose(result_df.loc[i, "b"], expected_result.intercept, atol=1e-5)


def test_grouped_linear_regression_non_overlapping_data(rs):
    ind = np.ones(10)
    a = rs.random(10)
    b = rs.random(10)

    a[:5] = np.nan
    b[5:] = np.nan

    r_df = grouped_linear_regression_pd(ind, a, b)
    assert r_df.size == 0

    r = grouped_linear_regression(ind, a, b)
    assert r.a.size == 2

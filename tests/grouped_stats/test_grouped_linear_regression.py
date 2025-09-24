import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import linregress
from statsmodels.regression.linear_model import OLS

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.grouped import grouped_linear_regression as grouped_linear_regression_internal
from pyspatialstats.results.stats import RegressionResult
from pyspatialstats.stats.p_values import calculate_p_value


def grouped_linear_regression_bootstrap_simple(x, y, n_bootstraps=1000):
    df_resid = 0
    seed = 42
    rng = np.random.default_rng(seed)
    n_features = x.shape[1] + 1
    betas = np.zeros((n_bootstraps, n_features))
    r2s = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        sample_idx = rng.choice(len(y), len(y), replace=True)
        x_s, y_s = x[sample_idx], y[sample_idx]
        x_s = sm.add_constant(x_s)
        model = sm.OLS(y_s, x_s).fit()
        betas[i, :] = model.params
        r2s[i] = model.rsquared
        df_resid += model.df_resid

    beta = betas.mean(axis=0)
    beta_se = betas.std(axis=0, ddof=1)
    r2 = r2s.mean()
    r2_se = r2s.std(ddof=1)
    df = df_resid / n_bootstraps

    return RegressionResult(
        beta=beta,
        beta_se=beta_se,
        r_squared=r2,
        r_squared_se=r2_se,
        df=df,
        t=beta / beta_se,
        p=calculate_p_value(beta / beta_se, df),
    )


def grouped_linear_regression(error='parametric', n_bootstraps=1000, *args, **kwargs):
    if error == 'bootstrap':
        kwargs['bootstrap_config'] = BootstrapConfig(seed=42, n_bootstraps=n_bootstraps)
    return grouped_linear_regression_internal(*args, error=error, **kwargs)


@pytest.mark.parametrize('error', ['bootstrap', 'parametric'])
def test_py_grouped_linear_regression_with_example_data(rs, error):
    n = 1000
    ind = np.ones(n, dtype=np.uintp)

    x = rs.normal(size=(n, 1))
    y = x.flatten() * 2 + rs.normal(size=n)

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False, error=error)

    assert result.df[0] == 0
    assert np.all(np.isnan(result.beta[0]))
    assert np.all(np.isnan(result.beta_se[0]))
    assert np.all(np.isnan(result.t[0]))
    assert np.all(np.isnan(result.p[0]))
    assert np.isnan(result.r_squared[0])

    if error == 'parametric':
        assert result.r_squared_se is None
    else:
        assert np.isnan(result.r_squared_se[0])

    assert np.allclose(result.beta[1], [0, 2], atol=0.1)
    assert np.allclose(result.beta_se[1], [0.03, 0.03], atol=0.01)
    assert np.allclose(result.p[1], [1, 0], atol=0.2)
    assert np.allclose(result.r_squared[1], 0.8, atol=0.8)

    if error == 'bootstrap':
        assert np.isclose(result.r_squared_se[1], 0.01, atol=0.1)


@pytest.mark.parametrize('error', ['bootstrap', 'parametric'])
def test_py_grouped_linear_regression_with_empty_arrays(error):
    ind = np.array([], dtype=np.uintp)
    v1 = np.array([], dtype=np.float64)
    v2 = np.array([], dtype=np.float64)
    result = grouped_linear_regression(ind=ind, x=v1, y=v2, filtered=False, error=error)

    assert result.beta.size == 0
    assert result.beta_se.size == 0
    assert result.t.size == 0
    assert result.p.size == 0
    assert result.r_squared.size == 0


@pytest.mark.parametrize('error', ['bootstrap', 'parametric'])
def test_py_grouped_linear_regression_with_different_length_arrays(error):
    ind = np.array([1, 1, 1, 2, 2, 2], dtype=np.uintp)
    x = np.array([10.0, 15.0, 20.0], dtype=np.float64)
    y = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float64)
    with pytest.raises(IndexError):
        grouped_linear_regression(ind=ind, x=x, y=y, error=error)


def test_grouped_linear_regression_against_simple(rs):
    n = 100
    ind = np.zeros(n)
    x = rs.random(size=(n, 1))
    y = rs.random(size=n)

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False)
    result_simple = grouped_linear_regression_bootstrap_simple(x, y)

    assert np.allclose(result.beta[0], result_simple.beta, atol=0.1)
    assert np.allclose(result.beta_se[0], result_simple.beta_se, atol=0.1)
    assert np.allclose(result.p[0], result_simple.p, atol=0.1)
    assert np.allclose(result.r_squared, result_simple.r_squared, atol=0.1)


def test_grouped_linear_regression_against_bootstrap_non_correlated(rs):
    n = 100
    ind = np.zeros(n)
    x = rs.random(size=(n, 1))
    y = rs.random(size=n)

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False)
    result_bootstrap = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False, error='bootstrap')

    assert np.allclose(result.beta, result_bootstrap.beta, atol=0.1)
    assert np.allclose(result.beta_se, result_bootstrap.beta_se, atol=0.1)
    assert np.allclose(result.p, result_bootstrap.p, atol=0.1)
    assert np.allclose(result.r_squared, result_bootstrap.r_squared, atol=0.1)


def test_grouped_linear_regression_against_bootstrap_correlated(rs):
    n = 100
    ind = np.zeros(n)
    x = rs.normal(size=(n, 1))
    y = x.flatten() * 2 + rs.normal(size=n)

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False)
    result_bootstrap = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False, error='bootstrap')

    assert np.allclose(result.beta, result_bootstrap.beta, atol=0.1)
    assert np.allclose(result.beta_se, result_bootstrap.beta_se, atol=0.1)
    assert np.allclose(result.p, result_bootstrap.p, atol=0.1)
    assert np.allclose(result.r_squared, result_bootstrap.r_squared, atol=0.1)


def test_py_grouped_linear_regression_against_scipy(rs):
    ind = np.arange(4).repeat(6).reshape(4, 6)
    x = rs.random((4, 6))
    y = rs.random((4, 6))

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False)

    for group in range(4):
        group_indices = ind == group
        group_x = x[group_indices]
        group_y = y[group_indices]

        expected_result = linregress(group_x, group_y)

        np.testing.assert_allclose(result.beta[group, 0], expected_result.intercept, rtol=1e-4)
        np.testing.assert_allclose(result.beta[group, 1], expected_result.slope, rtol=1e-4)
        np.testing.assert_allclose(result.beta_se[group, 0], expected_result.intercept_stderr, rtol=1e-4)
        np.testing.assert_allclose(result.beta_se[group, 1], expected_result.stderr, rtol=1e-4)
        np.testing.assert_allclose(result.p[group, 1], expected_result.pvalue, rtol=1e-4)
        np.testing.assert_allclose(result.r_squared[group], expected_result.rvalue**2, rtol=1e-4)


def test_py_grouped_linear_regression_against_statsmodels(rs):
    ind = np.arange(4).repeat(10).reshape(4, 10)
    x = rs.random((4, 10, 5))
    y = rs.random((4, 10))

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False)

    for group in range(4):
        group_indices = ind == group
        group_x = x[group_indices]
        group_y = y[group_indices]

        group_x_with_intercept = sm.add_constant(group_x)
        model = sm.OLS(group_y, group_x_with_intercept)
        result_statsmodels = model.fit()

        np.testing.assert_allclose(result.beta[group], result_statsmodels.params, rtol=1e-4)
        np.testing.assert_allclose(result.beta_se[group], result_statsmodels.bse, rtol=1e-4)
        np.testing.assert_allclose(result.p[group], result_statsmodels.pvalues, rtol=1e-4)
        np.testing.assert_allclose(result.t[group], result_statsmodels.tvalues, rtol=1e-4)
        np.testing.assert_allclose(result.r_squared[group], result_statsmodels.rsquared, rtol=1e-4)


def test_grouped_linear_regression_against_statsmodels_3D(rs):
    x = rs.random((35, 3))
    y = rs.random((35))
    ind = np.zeros(35)

    cy_result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False)

    x_with_intercept = sm.add_constant(x)
    sm_result = OLS(y, x_with_intercept).fit()

    np.testing.assert_allclose(cy_result.beta[0], sm_result.params, rtol=1e-4)
    np.testing.assert_allclose(cy_result.beta_se[0], sm_result.bse, rtol=1e-4)
    np.testing.assert_allclose(cy_result.p[0], sm_result.pvalues, rtol=1e-4)
    np.testing.assert_allclose(cy_result.t[0], sm_result.tvalues, rtol=1e-4)
    np.testing.assert_allclose(cy_result.df[0], sm_result.df_resid, rtol=1e-4)
    np.testing.assert_allclose(cy_result.r_squared[0], sm_result.rsquared, rtol=1e-4)


def test_py_grouped_linear_regression_with_nan_values(rs):
    ind = np.ones(10)
    x = rs.random(10)
    y = rs.random(10)
    x[1] = np.nan
    y[2] = np.nan

    mask = ~np.isnan(x) & ~np.isnan(y)

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False)
    expected_result = linregress(x[mask], y[mask])

    np.testing.assert_allclose(result.beta[1, 0], expected_result.intercept, rtol=1e-4)
    np.testing.assert_allclose(result.beta[1, 1], expected_result.slope, rtol=1e-4)
    np.testing.assert_allclose(result.beta_se[1, 0], expected_result.intercept_stderr, rtol=1e-4)
    np.testing.assert_allclose(result.beta_se[1, 1], expected_result.stderr, rtol=1e-4)
    np.testing.assert_allclose(result.p[1, 1], expected_result.pvalue, rtol=1e-4)
    np.testing.assert_allclose(result.r_squared[1], expected_result.rvalue**2, rtol=1e-4)


@pytest.mark.parametrize('error', ['bootstrap', 'parametric'])
def test_grouped_linear_regression_pd(ind, v1, v2, error):
    result_df = grouped_linear_regression(ind=ind, x=v1, y=v2, error=error)

    assert isinstance(result_df, pd.DataFrame)

    assert set(result_df.columns).issuperset(
        {
            'df',
            'beta_0',
            'beta_1',
            'beta_se_0',
            'beta_se_1',
            'p_0',
            'p_1',
            't_0',
            't_1',
            'r_squared',
        }
    )

    for i in result_df.index:
        values_in_group_1 = v1[ind == i]
        values_in_group_2 = v2[ind == i]
        expected_result = linregress(values_in_group_1, values_in_group_2)

        assert np.isclose(result_df.loc[i, 'beta_1'], expected_result.slope, atol=0.1)
        assert np.isclose(result_df.loc[i, 'beta_0'], expected_result.intercept, atol=0.1)


@pytest.mark.parametrize('error', ['bootstrap', 'parametric'])
def test_grouped_linear_regression_non_overlapping_data(rs, error):
    ind = np.ones(10)
    x = rs.random(10)
    y = rs.random(10)

    x[:5] = np.nan
    y[5:] = np.nan

    result = grouped_linear_regression(ind=ind, x=x, y=y, filtered=False, error=error)
    assert np.allclose(result.df, 0)
    assert np.all(np.isnan(result.beta))

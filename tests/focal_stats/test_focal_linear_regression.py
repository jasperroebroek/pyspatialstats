import numpy as np
import pytest
import statsmodels.api as sm
import xarray as xr
from statsmodels.regression.linear_model import OLS

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.focal import focal_linear_regression
from pyspatialstats.focal.result_config import FocalLinearRegressionResultConfig
from pyspatialstats.focal.utils import create_output_array
from pyspatialstats.results.stats import RegressionResult
from pyspatialstats.windows import define_window
from tests.bootstrap.np_stats import np_bootstrap_linear_regression
from tests.focal_stats.utils import (
    focal_linear_regression_simple,
)


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_focal_linear_regression_values(v1, v2, rs, reduce, error):
    cy_result = focal_linear_regression(v1, v2, window=5, reduce=reduce, error=error)
    sm_result = focal_linear_regression_simple(v1, v2, window=5, reduce=reduce)

    np.testing.assert_allclose(cy_result.df, sm_result.df, equal_nan=True, atol=0.01)
    np.testing.assert_allclose(cy_result.beta, sm_result.beta, equal_nan=True, atol=0.1)

    if error is not None:
        np.testing.assert_allclose(cy_result.beta_se, sm_result.beta_se, equal_nan=True, atol=0.1)
        np.testing.assert_allclose(cy_result.p, sm_result.p, equal_nan=True, atol=0.2)
        np.testing.assert_allclose(cy_result.r_squared, sm_result.r_squared, equal_nan=True, atol=0.1)


@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_focal_linear_regression_values_against_statsmodels(rs, error):
    x = rs.random((50, 50, 3))
    y = rs.random((50, 50))

    cy_result = focal_linear_regression(x, y, window=50, reduce=True, error=error)

    x_with_intercept = sm.add_constant(x.reshape(-1, 3))
    sm_result = OLS(y.reshape(-1), x_with_intercept).fit()

    np.testing.assert_allclose(cy_result.beta[0, 0], sm_result.params, atol=0.1)

    if error is not None:
        np.testing.assert_allclose(cy_result.beta_se[0, 0], sm_result.bse, atol=0.1)
        np.testing.assert_allclose(cy_result.p[0, 0], sm_result.pvalues, atol=0.1)
        np.testing.assert_allclose(cy_result.r_squared[0, 0], sm_result.rsquared, atol=0.1)
        np.testing.assert_allclose(cy_result.df[0, 0], sm_result.df_resid, atol=0.1)


def test_focal_linear_regression_bootstrap_values(rs):
    x = rs.random((50, 50, 3))
    y = rs.random((50, 50))

    result = focal_linear_regression(
        x,
        y,
        window=50,
        reduce=True,
        error='bootstrap',
        bootstrap_config=BootstrapConfig(n_bootstraps=1000, seed=0),
    )
    expected_result = np_bootstrap_linear_regression(x.reshape(-1, 3), y.flatten(), 1000, seed=0)

    np.testing.assert_allclose(result.beta[0, 0], expected_result.beta)
    np.testing.assert_allclose(result.beta_se[0, 0], expected_result.beta_se)
    np.testing.assert_allclose(result.p[0, 0], expected_result.p)
    np.testing.assert_allclose(result.r_squared[0, 0], expected_result.r_squared)
    np.testing.assert_allclose(result.t[0, 0], expected_result.t)
    np.testing.assert_allclose(result.r_squared_se[0, 0], expected_result.r_squared_se)
    np.testing.assert_allclose(result.df[0, 0], expected_result.df)


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_regression_values_mask(rs, reduce, error):
    a = rs.random((15, 15))
    b = rs.random((15, 15))
    window = define_window(rs.random((5, 5)) > 0.5)

    np.testing.assert_allclose(
        focal_linear_regression(a, b, window=window, fraction_accepted=0, reduce=reduce, error=error).beta,
        focal_linear_regression_simple(a, b, window=window, reduce=reduce).beta,
        equal_nan=True,
        atol=0.1,
    )


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_nan_behaviour_center(rs, reduce, error):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[2, 2] = np.nan
    fringes = define_window(5).get_fringes(reduce=reduce, ndim=2)
    np.testing.assert_allclose(
        focal_linear_regression(a, b, window=5, fraction_accepted=0, reduce=reduce, error=error).beta,
        focal_linear_regression_simple(a, b, window=5, reduce=reduce).beta,
        equal_nan=True,
        atol=0.1,
    )

    assert np.all(
        ~np.isnan(
            focal_linear_regression(a, b, window=5, fraction_accepted=0, reduce=reduce, error=error).beta[
                fringes[0], fringes[1]
            ]
        )
        == reduce
    )


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_nan_behaviour_non_center(rs, error, reduce):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[1, 1] = np.nan
    fringes = define_window(5).get_fringes(reduce=reduce, ndim=2)

    np.testing.assert_allclose(
        focal_linear_regression(a, b, window=5, fraction_accepted=0, error=error, reduce=reduce).beta,
        focal_linear_regression_simple(a, b, window=5, reduce=reduce).beta,
        equal_nan=True,
        atol=0.1,
    )

    assert np.all(
        ~np.isnan(
            focal_linear_regression(a, b, window=5, fraction_accepted=0, error=error, reduce=reduce).beta[*fringes]
        )
    )


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_regression_dtype(rs, reduce, error):
    a = rs.random((5, 5)).astype(np.int32)
    b = rs.random((5, 5)).astype(np.int32)
    assert focal_linear_regression(a, b, window=5, error=error, reduce=reduce).beta.dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    b = rs.random((5, 5)).astype(np.float64)
    assert focal_linear_regression(a, b, window=5, error=error, reduce=reduce).beta.dtype == np.float64


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_parallel(rs, error, reduce):
    x = rs.random((20, 20, 3))
    y = rs.random((20, 20))

    linear_r = focal_linear_regression(
        x,
        y,
        window=5,
        error=error,
        reduce=reduce,
        bootstrap_config=BootstrapConfig(n_bootstraps=1000, seed=0),
    )
    parallel_r = focal_linear_regression(
        x,
        y,
        window=5,
        chunks=10,
        error=error,
        reduce=reduce,
        bootstrap_config=BootstrapConfig(n_bootstraps=1000, seed=0),
    )

    np.testing.assert_allclose(linear_r.beta, parallel_r.beta, equal_nan=True, atol=0.2)

    if error is not None:
        np.testing.assert_allclose(linear_r.beta_se, parallel_r.beta_se, equal_nan=True, atol=0.1)
        np.testing.assert_allclose(linear_r.p, parallel_r.p, equal_nan=True, atol=0.15)
        np.testing.assert_allclose(linear_r.r_squared, parallel_r.r_squared, equal_nan=True, atol=0.1)
        np.testing.assert_allclose(linear_r.df, parallel_r.df, equal_nan=True, atol=0.1)


@pytest.mark.parametrize('chunks', (None, 15))
@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('error', ('parametric', 'bootstrap', None))
def test_xarray_output(rs, reduce, chunks, error):
    config = FocalLinearRegressionResultConfig(error=error)

    x = xr.DataArray(rs.random((20, 20, 3)))
    y = xr.DataArray(rs.random((20, 20)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=y) + (4,)

    output = RegressionResult(
        **{
            field: xr.DataArray(create_output_array(output_shape[: config.get_ndim(field)], dtype=np.float64))
            for field in config.active_fields
        }
    )

    r = focal_linear_regression(
        x,
        y,
        window=5,
        out=output,
        reduce=reduce,
        chunks=chunks,
        error=error,
        bootstrap_config=BootstrapConfig(n_bootstraps=1000, seed=0),
    )
    r_expected = focal_linear_regression(
        x,
        y,
        window=5,
        out=None,
        reduce=reduce,
        chunks=chunks,
        error=error,
        bootstrap_config=BootstrapConfig(n_bootstraps=1000, seed=0),
    )

    for field in config.active_fields:
        assert id(getattr(r, field)) == id(getattr(output, field))
        np.testing.assert_allclose(getattr(r, field).values, getattr(r_expected, field), equal_nan=True)

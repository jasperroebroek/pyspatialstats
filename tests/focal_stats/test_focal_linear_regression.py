import numpy as np
import pytest
import statsmodels.api as sm
import xarray as xr
from statsmodels.regression.linear_model import OLS

from pyspatialstats.focal import focal_linear_regression
from pyspatialstats.focal.result_config import FocalLinearRegressionResultConfig
from pyspatialstats.focal.utils import create_output_array
from pyspatialstats.results.stats import RegressionResult
from pyspatialstats.windows import define_window
from tests.focal_stats.utils import (
    focal_linear_regression_simple,
)


def test_lr_values(v1, v2):
    cy_result = focal_linear_regression(v1, v2, window=5)
    sm_result = focal_linear_regression_simple(v1, v2, window=5)

    # np.testing.assert_allclose(cy_result.df, sm_result.df, equal_nan=True)
    np.testing.assert_allclose(cy_result.beta, sm_result.beta, equal_nan=True)
    np.testing.assert_allclose(cy_result.beta_se, sm_result.beta_se, equal_nan=True)
    np.testing.assert_allclose(cy_result.t, sm_result.t, equal_nan=True, rtol=1e-5)
    np.testing.assert_allclose(cy_result.p, sm_result.p, equal_nan=True, rtol=1e-5)
    np.testing.assert_allclose(cy_result.r_squared, sm_result.r_squared, equal_nan=True, rtol=1e-5)


def test_linear_regression_values_against_statsmodels(rs):
    x = rs.random((5, 5, 3))
    y = rs.random((5, 5))

    cy_result = focal_linear_regression(x, y, window=5, reduce=True)

    x_with_intercept = sm.add_constant(x.reshape(-1, 3))
    sm_result = OLS(y.reshape(-1), x_with_intercept).fit()

    np.testing.assert_allclose(cy_result.beta[0, 0], sm_result.params)
    np.testing.assert_allclose(cy_result.beta_se[0, 0], sm_result.bse)
    np.testing.assert_allclose(cy_result.t[0, 0], sm_result.tvalues)
    np.testing.assert_allclose(cy_result.p[0, 0], sm_result.pvalues)
    np.testing.assert_allclose(cy_result.r_squared[0, 0], sm_result.rsquared)
    np.testing.assert_allclose(cy_result.df[0, 0], sm_result.df_resid)


def test_regression_values_mask(rs):
    a = rs.random((15, 15))
    b = rs.random((15, 15))
    window = define_window(rs.random((5, 5)) > 0.5)

    assert np.allclose(
        focal_linear_regression(a, b, window=window, fraction_accepted=0).beta,
        focal_linear_regression_simple(a, b, window=window).beta,
        equal_nan=True,
    )


def test_regression_values_mask_reduce(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    window = define_window(rs.random((5, 5)) > 0.5)

    assert np.allclose(
        focal_linear_regression(a, b, window=window, fraction_accepted=0, reduce=True).beta,
        focal_linear_regression_simple(a, b, window=window).beta[2, 2],
        equal_nan=True,
    )


def test_regression_shape(v1, v2):
    assert focal_linear_regression(v1, v2, window=3).beta.shape == v2.shape + (2,)
    assert focal_linear_regression(v1, v2, window=10, reduce=True).beta.shape == (1, 1, 2)


def test_nan_behaviour_center(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[2, 2] = np.nan
    assert np.allclose(
        focal_linear_regression(a, b, window=5).beta,
        focal_linear_regression_simple(a, b, window=5).beta,
        equal_nan=True,
    )
    assert np.all(np.isnan(focal_linear_regression(a, b, window=5).beta[2, 2]))

    a = rs.random((5, 5))
    b = rs.random((5, 5))
    b[2, 2] = np.nan
    assert np.allclose(
        focal_linear_regression(a, b, window=5).beta,
        focal_linear_regression_simple(a, b, window=5).beta,
        equal_nan=True,
    )
    assert np.all(np.isnan(focal_linear_regression(a, b, window=5).beta[2, 2]))


def test_nan_behaviour_non_center(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[1, 1] = np.nan
    assert np.allclose(
        focal_linear_regression(a, b, window=5).beta,
        focal_linear_regression_simple(a, b, window=5).beta,
        equal_nan=True,
    )
    assert not np.all(np.isnan(focal_linear_regression(a, b, window=5).beta[2, 2]))
    assert np.all(np.isnan(focal_linear_regression(a, b, fraction_accepted=1, window=5).beta[2, 2]))
    assert not np.all(np.isnan(focal_linear_regression(a, b, fraction_accepted=0, window=5).beta[2, 2]))

    a = rs.random((5, 5))
    b = rs.random((5, 5))
    b[1, 1] = np.nan
    assert np.allclose(
        focal_linear_regression(a, b, window=5).beta,
        focal_linear_regression_simple(a, b, window=5).beta,
        equal_nan=True,
    )
    assert not np.all(np.isnan(focal_linear_regression(a, b, window=5).beta[2, 2]))
    assert np.all(np.isnan(focal_linear_regression(a, b, fraction_accepted=1, window=5).beta[2, 2]))
    assert not np.all(np.isnan(focal_linear_regression(a, b, fraction_accepted=0, window=5).beta[2, 2]))


def test_regression_dtype(rs):
    a = rs.random((5, 5)).astype(np.int32)
    b = rs.random((5, 5)).astype(np.int32)
    assert focal_linear_regression(a, b, window=5).beta.dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    b = rs.random((5, 5)).astype(np.float64)
    assert focal_linear_regression(a, b, window=5).beta.dtype == np.float64


def test_parallel(rs):
    a1 = rs.random((100, 100))
    a2 = rs.random((100, 100))

    linear_r = focal_linear_regression(a1, a2, window=5)
    parallel_r = focal_linear_regression(a1, a2, window=5, chunks=75)

    np.testing.assert_allclose(linear_r.beta, parallel_r.beta, equal_nan=True)
    np.testing.assert_allclose(linear_r.beta_se, parallel_r.beta_se, equal_nan=True)
    np.testing.assert_allclose(linear_r.t, parallel_r.t, equal_nan=True)
    np.testing.assert_allclose(linear_r.p, parallel_r.p, equal_nan=True)
    np.testing.assert_allclose(linear_r.r_squared, parallel_r.r_squared, equal_nan=True)
    np.testing.assert_allclose(linear_r.df, parallel_r.df, equal_nan=True)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
def test_xarray_output(rs, reduce, chunks):
    config = FocalLinearRegressionResultConfig()

    a1 = xr.DataArray(rs.random((100, 100)))
    a2 = xr.DataArray(rs.random((100, 100)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a1) + (2,)

    output = RegressionResult(
        **{
            field: xr.DataArray(create_output_array(output_shape[: config.get_ndim(field)], dtype=np.float64))
            for field in config.active_fields
        }
    )

    r = focal_linear_regression(a1, a2, window=5, out=output, reduce=reduce, chunks=chunks)
    r_expected = focal_linear_regression(a1, a2, window=5, out=None, reduce=reduce, chunks=chunks)

    for field in config.active_fields:
        assert id(getattr(r, field)) == id(getattr(output, field))
        np.testing.assert_allclose(getattr(r, field).values, getattr(r_expected, field), equal_nan=True)

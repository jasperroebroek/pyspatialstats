import numpy as np
import pytest
import xarray as xr

from pyspatialstats.focal import focal_linear_regression
from pyspatialstats.focal.result_config import FocalLinearRegressionResultConfig
from pyspatialstats.focal.utils import create_output_raster
from pyspatialstats.types.results import LinearRegressionResult
from pyspatialstats.utils import get_dtype
from pyspatialstats.windows import define_window
from tests.focal_stats.utils import (
    focal_linear_regression_simple,
)


def test_lr_values(v1, v2):
    cy_result = focal_linear_regression(v1, v2, window=5, p_values=True)
    sm_result = focal_linear_regression_simple(v1, v2, window=5)

    assert np.allclose(cy_result.a, sm_result.a, equal_nan=True)
    assert np.allclose(cy_result.b, sm_result.b, equal_nan=True)
    assert np.allclose(cy_result.se_a, sm_result.se_a, equal_nan=True)
    assert np.allclose(cy_result.se_b, sm_result.se_b, equal_nan=True)
    assert np.allclose(cy_result.t_a, sm_result.t_a, equal_nan=True)
    assert np.allclose(cy_result.t_b, sm_result.t_b, equal_nan=True)
    assert np.allclose(cy_result.p_a, sm_result.p_a, equal_nan=True)
    assert np.allclose(cy_result.p_b, sm_result.p_b, equal_nan=True)


def test_correlation_values_mask(rs):
    a = rs.random((15, 15))
    b = rs.random((15, 15))
    window = define_window(rs.random((5, 5)) > 0.5)

    assert np.allclose(
        focal_linear_regression(a, b, window=window, fraction_accepted=0).a,
        focal_linear_regression_simple(a, b, window=window).a,
        equal_nan=True,
    )


def test_correlation_values_mask_reduce(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    window = define_window(rs.random((5, 5)) > 0.5)

    assert np.allclose(
        focal_linear_regression(a, b, window=window, fraction_accepted=0, reduce=True).a,
        focal_linear_regression_simple(a, b, window=window).a[2, 2],
        equal_nan=True,
    )


def test_correlation_shape(v1, v2):
    assert focal_linear_regression(v1, v2, window=3).a.shape == v1.shape
    assert focal_linear_regression(v1, v2, window=10, reduce=True).a.shape == (1, 1)


def test_nan_behaviour(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[2, 2] = np.nan
    assert np.allclose(
        focal_linear_regression(a, b, window=5).a, focal_linear_regression_simple(a, b, window=5).a, equal_nan=True
    )
    assert np.isnan(focal_linear_regression(a, b, window=5).a[2, 2])

    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[1, 1] = np.nan
    assert np.allclose(
        focal_linear_regression(a, b, window=5).a, focal_linear_regression_simple(a, b, window=5).a, equal_nan=True
    )
    assert not np.isnan(focal_linear_regression(a, b, window=5).a[2, 2])
    assert np.isnan(focal_linear_regression(a, b, fraction_accepted=1, window=5).a[2, 2])
    assert not np.isnan(focal_linear_regression(a, b, fraction_accepted=0, window=5).a[2, 2])


def test_correlation_dtype(rs):
    a = rs.random((5, 5)).astype(np.int32)
    b = rs.random((5, 5)).astype(np.int32)
    assert focal_linear_regression(a, b, window=5).a.dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    b = rs.random((5, 5)).astype(np.float64)
    assert focal_linear_regression(a, b, window=5).a.dtype == np.float64


def test_parallel(rs):
    a1 = rs.random((100, 100))
    a2 = rs.random((100, 100))

    linear_r = focal_linear_regression(a1, a2, window=5, p_values=True)
    parallel_r = focal_linear_regression(a1, a2, window=5, p_values=True, chunks=75)

    np.testing.assert_allclose(linear_r.a, parallel_r.a, equal_nan=True)
    np.testing.assert_allclose(linear_r.b, parallel_r.b, equal_nan=True)
    np.testing.assert_allclose(linear_r.se_a, parallel_r.se_a, equal_nan=True)
    np.testing.assert_allclose(linear_r.se_b, parallel_r.se_b, equal_nan=True)
    np.testing.assert_allclose(linear_r.t_a, parallel_r.t_a, equal_nan=True)
    np.testing.assert_allclose(linear_r.t_b, parallel_r.t_b, equal_nan=True)
    np.testing.assert_allclose(linear_r.p_a, parallel_r.p_a, equal_nan=True)
    np.testing.assert_allclose(linear_r.p_b, parallel_r.p_b, equal_nan=True)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
def test_xarray_output(rs, reduce, chunks):
    config = FocalLinearRegressionResultConfig(p_values=True)

    a1 = xr.DataArray(rs.random((100, 100)))
    a2 = xr.DataArray(rs.random((100, 100)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a1)

    output = LinearRegressionResult(
        **{
            field: xr.DataArray(create_output_raster(output_shape, dtype=get_dtype(field)))
            for field in config.active_fields
        }
    )

    r = focal_linear_regression(a1, a2, window=5, out=output, reduce=reduce, chunks=chunks, p_values=True)
    r_expected = focal_linear_regression(a1, a2, window=5, out=None, reduce=reduce, chunks=chunks, p_values=True)

    for field in config.active_fields:
        assert id(getattr(r, field)) == id(getattr(output, field))
        np.testing.assert_allclose(getattr(r, field).values, getattr(r_expected, field), equal_nan=True)

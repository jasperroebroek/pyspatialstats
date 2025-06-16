import numpy as np
import pytest

from pyspatialstats.focal import focal_correlation, focal_linear_regression
from pyspatialstats.windows import define_window
from tests.focal_stats.utils import (
    focal_correlation_simple,
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
        focal_correlation(a, b, window=window, fraction_accepted=0).c,
        focal_correlation_simple(a, b, window=window).c,
        equal_nan=True,
    )


def test_correlation_values_mask_reduce(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    window = define_window(rs.random((5, 5)) > 0.5)

    assert np.allclose(
        focal_correlation(a, b, window=window, fraction_accepted=0, reduce=True).c,
        focal_correlation_simple(a, b, window=window).c[2, 2],
        equal_nan=True,
    )


def test_correlation_shape(v1, v2):
    assert focal_correlation(v1, v2, window=3).c.shape == v1.shape
    assert focal_correlation(v1, v2, window=10, reduce=True).c.shape == (1, 1)


def test_correlation_errors(rs, v1):
    # not 2D
    with pytest.raises(IndexError):
        a = rs.random((10, 10, 10))
        b = rs.random((10, 10, 10))
        focal_correlation(a, b)

    # different shapes
    with pytest.raises(ValueError):
        a = rs.random((15, 15))
        focal_correlation(a, v1)

    with pytest.raises(TypeError):
        focal_correlation(v1, v1, window="x")

    with pytest.raises(ValueError):
        focal_correlation(v1, v1, window=1)

    with pytest.raises(ValueError):
        focal_correlation(v1, v1, window=11)

    # uneven window_shape is not supported
    with pytest.raises(ValueError):
        focal_correlation(v1, v1, window=4)

    # Not exactly divided in reduce mode
    with pytest.raises((NotImplementedError, ValueError)):
        focal_correlation(v1, v1, window=4, reduce=True)

    with pytest.raises(ValueError):
        focal_correlation(v1, v1, fraction_accepted=-0.1)

    with pytest.raises(ValueError):
        focal_correlation(v1, v1, fraction_accepted=1.1)



def test_nan_behaviour(rs):
    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[2, 2] = np.nan
    assert np.allclose(
        focal_correlation(a, b).c, focal_correlation_simple(a, b).c, equal_nan=True
    )
    assert np.isnan(focal_correlation(a, b).c[2, 2])

    a = rs.random((5, 5))
    b = rs.random((5, 5))
    a[1, 1] = np.nan
    assert np.allclose(
        focal_correlation(a, b).c, focal_correlation_simple(a, b).c, equal_nan=True
    )
    assert not np.isnan(focal_correlation(a, b).c[2, 2])
    assert np.isnan(focal_correlation(a, b, fraction_accepted=1).c[2, 2])
    assert not np.isnan(focal_correlation(a, b, fraction_accepted=0).c[2, 2])


def test_correlation_dtype(rs):
    a = rs.random((5, 5)).astype(np.int32)
    b = rs.random((5, 5)).astype(np.int32)
    assert focal_correlation(a, b).c.dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    b = rs.random((5, 5)).astype(np.float64)
    assert focal_correlation(a, b).c.dtype == np.float64

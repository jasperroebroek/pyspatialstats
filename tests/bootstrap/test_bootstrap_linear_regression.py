import numpy as np
import pytest

from pyspatialstats.bootstrap.linear_regression import py_bootstrap_linear_regression
from tests.bootstrap.np_stats import np_bootstrap_linear_regression


def test_bootstrap_lr_basic():
    rs = np.random.default_rng(42)
    n = 50
    n_params = 3
    x = rs.random((n, n_params))
    y = x @ np.array([1.5, -2.0, 0.5]) + rs.random(n) * 0.1

    result = py_bootstrap_linear_regression(x, y, 1000, seed=13)

    assert isinstance(result.r_squared, float)
    assert isinstance(result.r_squared_se, float)
    assert result.beta.shape == (n_params + 1,)
    assert result.beta_se.shape == (n_params + 1,)
    assert result.status == 0


def test_bootstrap_lr_invalid_bootstrap_size():
    x = np.random.randn(10, 2)
    y = np.random.randn(10)
    with pytest.raises(ValueError):
        py_bootstrap_linear_regression(x, y, 1)


def test_bootstrap_lr_invalid_sample_size():
    x = np.zeros((1, 2))
    y = np.zeros(1)
    with pytest.raises(ValueError):
        py_bootstrap_linear_regression(x, y, 10)


def test_bootstrap_lr_seed_consistency():
    x = np.random.randn(30, 2)
    y = np.random.randn(30)

    r1 = py_bootstrap_linear_regression(x, y, 50, seed=123)
    r2 = py_bootstrap_linear_regression(x, y, 50, seed=123)

    np.testing.assert_array_equal(r1.beta, r2.beta)
    np.testing.assert_array_equal(r1.beta_se, r2.beta_se)
    assert r1.r_squared == r2.r_squared
    assert r1.r_squared_se == r2.r_squared_se


def test_bootstrap_lr_comparison_to_numpy():
    x = np.random.randn(40, 2)
    y = x @ np.array([2.0, -1.0]) + np.random.randn(40) * 0.1

    cy_result = py_bootstrap_linear_regression(x, y, 200, seed=0)
    np_result = np_bootstrap_linear_regression(x, y, 200, seed=0)

    np.testing.assert_allclose(cy_result.beta, np_result.beta, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(cy_result.beta_se, np_result.beta_se, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(cy_result.r_squared, np_result.r_squared, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(cy_result.r_squared_se, np_result.r_squared_se, rtol=1e-1, atol=1e-1)

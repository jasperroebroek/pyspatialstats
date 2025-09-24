import numpy as np
import pytest
from scipy import stats

from pyspatialstats.stats.correlation import Correlation


@pytest.fixture
def simple_data():
    rs = np.random.default_rng(42)
    n = 200
    x = rs.random(n)
    y = 2.0 * x + 0.5 * rs.random(n)
    return x, y


def test_initialization():
    corr = Correlation()
    assert corr.count == 0
    assert np.isnan(corr.corr)


def test_single_addition():
    corr = Correlation()
    corr.add(1.0, 2.0)
    assert corr.count == 1
    # corr is undefined with only one point
    assert np.isnan(corr.corr)


def test_against_numpy_and_scipy(simple_data):
    x, y = simple_data
    corr = Correlation()
    for xi, yi in zip(x, y):
        corr.add(xi, yi)
    assert corr.count == len(x)

    # Compare against numpy/scipy
    expected_np = np.corrcoef(x, y)[0, 1]
    expected_sp, _ = stats.pearsonr(x, y)

    np.testing.assert_allclose(corr.corr, expected_np, rtol=1e-12)
    np.testing.assert_allclose(corr.corr, expected_sp, rtol=1e-12)


def test_reset(simple_data):
    x, y = simple_data
    corr = Correlation()
    corr.add(x[0], y[0])
    assert corr.count == 1
    corr.reset()
    assert corr.count == 0
    assert np.isnan(corr.corr)


def test_merge_equivalence(simple_data):
    x, y = simple_data
    corr1 = Correlation()
    corr2 = Correlation()

    # Split data into two halves
    half = len(x) // 2
    for xi, yi in zip(x[:half], y[:half]):
        corr1.add(xi, yi)
    for xi, yi in zip(x[half:], y[half:]):
        corr2.add(xi, yi)

    # Merge should equal combined
    corr1.merge(corr2)

    corr_full = Correlation()
    for xi, yi in zip(x, y):
        corr_full.add(xi, yi)

    np.testing.assert_allclose(corr1.corr, corr_full.corr, rtol=1e-15)


def test_constant_variable():
    x = np.ones(100)
    y = np.random.randn(100)
    corr = Correlation()
    for xi, yi in zip(x, y):
        corr.add(xi, yi)
    assert np.isnan(corr.corr)


def test_perfect_correlation():
    x = np.arange(10, dtype=float)
    y = 2.0 * x + 1.0
    corr = Correlation()
    for xi, yi in zip(x, y):
        corr.add(xi, yi)
    assert np.isclose(corr.corr, 1.0)


def test_perfect_anticorrelation():
    x = np.arange(10, dtype=float)
    y = -3.0 * x + 5.0
    corr = Correlation()
    for xi, yi in zip(x, y):
        corr.add(xi, yi)
    assert np.isclose(corr.corr, -1.0)

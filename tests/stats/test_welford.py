import numpy as np
import pytest

from pyspatialstats.stats.welford import Welford


@pytest.fixture
def simple_data():
    return np.random.default_rng(200).random(200)


def test_initialization():
    w = Welford()
    assert w.count == 0
    assert np.isnan(w.mean)
    assert np.isnan(w.std)


def test_single_addition():
    w = Welford()
    w.add(3.0)
    assert w.count == 1
    assert np.isclose(w.mean, 3.0)
    # std is undefined with only one point
    assert np.isnan(w.std)


def test_against_numpy(simple_data):
    w = Welford()
    for xi in simple_data:
        w.add(xi)

    assert w.count == len(simple_data)

    # Compare against numpy
    expected_mean = np.mean(simple_data)
    expected_std = np.std(simple_data, ddof=1)  # sample std

    np.testing.assert_allclose(w.mean, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(w.std, expected_std, rtol=1e-12)


def test_reset(simple_data):
    w = Welford()
    w.add(simple_data[0])
    assert w.count == 1
    w.reset()
    assert w.count == 0
    assert np.isnan(w.mean)
    assert np.isnan(w.std)


def test_merge_equivalence(simple_data):
    half = len(simple_data) // 2
    w1 = Welford()
    w2 = Welford()

    for xi in simple_data[:half]:
        w1.add(xi)
    for xi in simple_data[half:]:
        w2.add(xi)

    w1.merge(w2)

    w_full = Welford()
    for xi in simple_data:
        w_full.add(xi)

    np.testing.assert_allclose(w1.mean, w_full.mean, rtol=1e-15)
    np.testing.assert_allclose(w1.std, w_full.std, rtol=1e-15)


def test_constant_variable():
    x = np.ones(100)
    w = Welford()
    for xi in x:
        w.add(xi)
    assert np.isclose(w.mean, 1.0)
    assert np.isclose(w.std, 0.0)


def test_single_value():
    w = Welford()
    w.add(42.0)
    assert w.count == 1
    assert np.isclose(w.mean, 42.0)
    assert np.isnan(w.std)

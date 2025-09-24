import numpy as np

from pyspatialstats.random.random import Random


def test_random_functionality():
    bound = 100
    n = 1000

    gen = Random(123)

    values = gen.np_randints(bound, n)

    assert values.shape == (n,)
    assert values.dtype == np.uint64
    assert np.all(values >= 0)
    assert np.all(values <= bound)


def test_random_seed_consistency():
    """Test that same seed produces same sequence"""
    bound = 1000
    n = 100

    v1 = Random(seed=42).np_randints(bound, n)
    v2 = Random(seed=42).np_randints(bound, n)

    np.testing.assert_array_equal(v1, v2)


def test_randints_with_numpy():
    bound = 1000
    n = 10000

    cy_values = Random(seed=0).np_randints(bound, n)
    np_values = np.random.default_rng(seed=0).integers(low=0, high=bound, size=n)

    np.testing.assert_array_equal(cy_values, np_values)


def test_randpoisson_shape_and_type():
    r = Random(seed=42)
    lam = 5.0
    n = 1000

    samples = r.np_randpoisson(lam, n)

    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.int64
    assert samples.shape == (n,)


def test_randpoisson_values_are_nonnegative():
    r = Random(seed=42)
    samples = r.np_randpoisson(3.5, 500)
    assert np.all(samples >= 0)


def test_randpoisson_mean_close_to_lambda():
    r = Random(seed=123)
    lam = 10.0
    n = 10000
    samples = r.np_randpoisson(lam, n)
    assert np.isclose(np.mean(samples), lam, rtol=0.01)


def test_randpoisson_randomness():
    r1 = Random(seed=1)
    r2 = Random(seed=2)

    samples1 = r1.np_randpoisson(7.0, 1000)
    samples2 = r2.np_randpoisson(7.0, 1000)

    assert not np.array_equal(samples1, samples2)


def test_single_poisson_call_is_valid():
    r = Random(seed=101)
    for lam in [0.5, 1.0, 5.0, 20.0]:
        val = r.np_randpoisson(lam, 1).item()
        assert isinstance(val, int)
        assert val >= 0

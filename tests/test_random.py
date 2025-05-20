import numpy as np

from pyspatialstats.random.random import RandomInts


def test_random_functionality():
    bound = 100
    n = 1000

    gen = RandomInts(123)

    values = gen.np_randints(bound, n)

    assert values.shape == (n,)
    assert values.dtype == np.uint64
    assert np.all(values >= 0)
    assert np.all(values <= bound)


def test_random_seed_consistency():
    """Test that same seed produces same sequence"""
    bound = 1000
    n = 100

    v1 = RandomInts(seed=42).np_randints(bound, n)
    v2 = RandomInts(seed=42).np_randints(bound, n)

    np.testing.assert_array_equal(v1, v2)


def test_randints_with_numpy():
    bound = 1000
    n = 10000

    cy_values = RandomInts(seed=0).np_randints(bound, n)
    np_values = np.random.default_rng(seed=0).integers(low=0, high=bound, size=n)

    np.testing.assert_array_equal(cy_values, np_values)

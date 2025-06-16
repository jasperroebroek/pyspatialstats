import numpy as np
import pytest
from numpy.ma.testutils import assert_array_approx_equal

from pyspatialstats.grouped import define_max_ind as cy_define_max_ind, generate_index


def define_max_ind(ind):
    if ind.size == 0:
        return 0
    return int(ind.max())


def test_max_ind_empty_input():
    # Edge case: empty input arrays
    ind_empty = np.array([[]], dtype=np.uintp)
    max_ind_empty = cy_define_max_ind(ind_empty)
    assert max_ind_empty == 0


@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_max_ind(ndim):
    # Edge case: empty input arrays
    ind = np.random.randint(0, 10, size=(10,) * ndim)
    assert cy_define_max_ind(ind) == define_max_ind(ind)


@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_index(ndim):
    shape = (10,) * ndim
    ind = np.random.randint(0, 10, size=shape)
    v = np.ones(shape)

    unique = np.unique(ind)
    assert_array_approx_equal(unique, generate_index(ind, v))


def test_index_empty_ind():
    ind = np.zeros(10)
    v = np.ones(10)

    assert_array_approx_equal(generate_index(ind, v), [0])


def test_index_shape_mismatch():
    ind = np.zeros(10)
    v = np.ones(11)

    with pytest.raises(IndexError):
        generate_index(ind, v)
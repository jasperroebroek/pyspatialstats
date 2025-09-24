import numpy as np
import pytest

from pyspatialstats.grouped import define_max_ind as cy_define_max_ind


def define_max_ind(ind):
    if ind.size == 0:
        return 0
    return int(ind.max())


def test_max_ind_empty_input():
    ind_empty = np.array([[]], dtype=np.uintp)
    max_ind_empty = cy_define_max_ind(ind_empty)
    assert max_ind_empty == -1


@pytest.mark.parametrize('ndim', (1, 2, 3))
def test_max_ind(ndim):
    ind = np.random.randint(0, 10, size=(10,) * ndim)
    assert cy_define_max_ind(ind) == define_max_ind(ind)

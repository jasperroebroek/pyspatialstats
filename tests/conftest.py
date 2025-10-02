import sys

import numpy as np
import pytest

print('>>> pytest using NumPy', np.__version__, 'from', np.__file__, file=sys.stderr)


@pytest.fixture
def rs():
    return np.random.default_rng(0)


@pytest.fixture
def ind():
    return np.random.default_rng(40).integers(0, 5, size=(10, 10), dtype=np.uintp)


@pytest.fixture
def v():
    return np.random.default_rng(41).random(size=(10, 10))


@pytest.fixture
def v1():
    return np.random.default_rng(42).random(size=(10, 10))


@pytest.fixture
def v2():
    return np.random.default_rng(43).random(size=(10, 10))

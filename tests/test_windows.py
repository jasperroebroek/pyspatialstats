import numpy as np
import pytest

from pyspatialstats.windows import MaskedWindow, RectangularWindow, define_window


@pytest.fixture
def basic_shape():
    return np.array([10, 10])


@pytest.fixture
def square_mask():
    return np.ones((3, 3), dtype=bool)


@pytest.fixture
def non_square_mask():
    return np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)


def test_rectangular_window_shape_from_int():
    w = RectangularWindow(window_size=3)
    assert w.get_shape(2) == (3, 3)
    assert not w.masked


def test_window_default_ndim():
    w = define_window(5)

    assert w.get_shape() == (5, 5)
    assert w.get_mask().shape == (5, 5)
    assert np.array_equal(w.get_fringes(True), (0, 0))
    assert np.array_equal(w.get_fringes(False), (2, 2))


def test_rectangular_window_shape_from_tuple():
    w = RectangularWindow(window_size=(3, 5))
    assert w.get_shape(2) == (3, 5)


def test_rectangular_window_shape_invalid_dim():
    w = RectangularWindow(window_size=(3, 5))
    with pytest.raises(IndexError):
        w.get_shape(3)


def test_rectangular_window_mask():
    w = RectangularWindow(window_size=3)
    mask = w.get_mask(2)
    assert mask.shape == (3, 3)
    assert np.all(mask)


def test_masked_window_get_shape(square_mask):
    w = MaskedWindow(mask=square_mask)
    assert w.get_shape(2) == (3, 3)
    assert w.masked


def test_masked_window_mismatched_ndim(square_mask):
    w = MaskedWindow(mask=square_mask)
    with pytest.raises(IndexError):
        w.get_shape(3)


def test_masked_window_get_mask(square_mask):
    w = MaskedWindow(mask=square_mask)
    assert np.array_equal(w.get_mask(2), square_mask)


def test_types(square_mask):
    assert isinstance(define_window(3), RectangularWindow)
    assert isinstance(define_window((3, 3)), RectangularWindow)
    assert isinstance(define_window(square_mask), MaskedWindow)


def test_define_window_invalid_type():
    with pytest.raises(TypeError):
        define_window('invalid')


def test_validate_window_basic(basic_shape):
    w = define_window((3, 3))
    w.validate(reduce=False, shape=basic_shape)


def test_validate_window_reduction_valid(basic_shape):
    w = define_window((5, 5))
    w.validate(reduce=True, shape=basic_shape)


def test_validate_window_window_too_large():
    w = define_window((10, 10))
    with pytest.raises(ValueError):
        w.validate(reduce=False, a=np.empty(shape=(5, 5)))


def test_validate_window_even_not_allowed():
    w = define_window((4, 4))
    with pytest.raises(ValueError):
        w.validate(reduce=False, allow_even=False, a=np.empty(shape=(10, 10)))


def test_validate_window_mask_empty():
    mask = np.zeros((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError):
        define_window(mask)


def test_define_fringes_reduce():
    window = define_window((3, 3))
    fringes = window.get_fringes(reduce=True)
    assert np.array_equal(fringes, (0, 0))

    window = define_window((5, 5))
    fringes = window.get_fringes(reduce=False)
    assert np.array_equal(fringes, (2, 2))


def test_define_ind_inner():
    window = define_window((3, 3))
    s = window.get_ind_inner(reduce=True)
    assert isinstance(s, tuple)

    window = define_window((5, 5))
    s = window.get_ind_inner(reduce=False)
    assert isinstance(s, tuple)

import numpy as np
import pytest
from scipy.stats import pearsonr

from pyspatialstats.focal_stats import focal_correlation
from pyspatialstats.windows import define_window


def overlapping_arrays(m, preserve_input=True):
    """
    Taking two maps of the same raster_shape and returns them with  all the cells that don't exist in the other set to np.nan

    Parameters
    ----------
    m: iterable of :obj:`~numpy.ndarray`
        list of a minimum of 2 identically shaped numpy arrays.
    preserve_input : bool, optional
        if set to True the data is copied before applying the mask to preserve the input arrays. If set to False the
        memory space of the input arrays will be used.

    Returns
    -------
    m : list of :obj:`~numpy.ndarray`

    Raises
    ------
    TypeError
        if arrays are unreadable
    """
    if len(m) < 2:
        raise IndexError("list needs to contain a minimum of two arrays")

    for a in m:
        if not isinstance(a, np.ndarray):
            raise TypeError("all entries in the list need to be ndarrays")

    if not np.all([a.shape == m[0].shape for a in m]):
        raise ValueError("arrays are not of the same size")

    m_new = []
    for a in m:
        if a.dtype != np.float64:
            m_new.append(a.astype(np.float64))
        elif preserve_input:
            m_new.append(a.copy())
        else:
            m_new.append(a)
    m = m_new

    valid_cells = np.logical_and.reduce([~np.isnan(a) for a in m])

    for a in m:
        a[~valid_cells] = np.nan

    return m


def focal_correlation_simple(map1, map2, window=5, fraction_accepted=0.7):
    """
    Takes two maps and returning the local correlation between them with the same dimensions as the input maps.
    Correlation calculated in a rolling window, with a Window object. If either of the input maps contains
    a NaN value on a location, the output map will also have a NaN on that location. This is a simplified version of
    correlate_maps() in raster_functions with the purpose of testing. It is super slow, so don't throw large maps
    at it.

    Parameters
    ----------
    map1, map2 : array-like
        Input arrays that will be correlated. If not present in dtype `np.float64` it will be converted internally.
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.

    Returns
    -------
    corr : :obj:`~numpy.ndarray`
        numpy array of the same raster_shape as a and b with the local correlation

    """
    map1, map2 = overlapping_arrays([map1, map2])
    window = define_window(window)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    fringes = window_shape // 2
    corr = np.full(map1.shape, np.nan)

    for i in range(fringes[0], map1.shape[0] - fringes[0]):
        for j in range(fringes[1], map1.shape[1] - fringes[1]):
            ind = np.s_[
                i - fringes[0] : i + fringes[0] + 1, j - fringes[1] : j + fringes[1] + 1
            ]

            if np.isnan(map1[i, j]) or np.isnan(map2[i, j]):
                continue

            d1 = map1[ind][mask]
            d2 = map2[ind][mask]

            d1 = d1[~np.isnan(d1)]
            d2 = d2[~np.isnan(d2)]

            if d1.size < fraction_accepted * mask.sum():
                continue

            if np.all(d1 == d1[0]) or np.all(d2 == d2[0]):
                corr[i, j] = 0
                continue

            corr[i, j] = pearsonr(d1, d2)[0]

    return corr


def test_correlation_values():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    # Cython implementation
    assert np.allclose(
        pearsonr(a.flatten(), b.flatten())[0],
        focal_correlation(a, b, window=5, reduce=True),
    )

    # Local implementation
    assert np.allclose(
        pearsonr(a.flatten(), b.flatten())[0],
        focal_correlation_simple(a, b, window=5)[2, 2],
    )


def test_correlation_values_large():
    # compare for larger raster_shape
    a = np.random.rand(25, 25)
    b = np.random.rand(25, 25)

    assert np.allclose(
        focal_correlation(a, b), focal_correlation_simple(a, b), equal_nan=True
    )


def test_correlation_values_mask():
    a = np.random.rand(15, 15)
    b = np.random.rand(15, 15)
    window = define_window(np.random.RandomState().rand(5, 5) > 0.5)

    assert np.allclose(
        focal_correlation(a, b, window=window, fraction_accepted=0),
        focal_correlation_simple(a, b, window=window),
        equal_nan=True,
    )


def test_correlation_values_mask_reduce():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)
    window = define_window(np.random.RandomState().rand(5, 5) > 0.5)

    assert np.allclose(
        focal_correlation(a, b, window=window, fraction_accepted=0, reduce=True),
        focal_correlation_simple(a, b, window=window)[2, 2],
        equal_nan=True,
    )


def test_correlation_shape():
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)

    assert focal_correlation(a, b, window=3).shape == a.shape
    assert focal_correlation(a, b, window=10, reduce=True).shape == (1, 1)


def test_correlation_errors():
    with pytest.raises(ValueError):
        focal_correlation(
            np.random.rand(10, 10), np.random.rand(10, 10), window=5, verbose=2
        )

    with pytest.raises(ValueError):
        focal_correlation(
            np.random.rand(10, 10), np.random.rand(10, 10), window=5, reduce=2
        )

    # not 2D
    with pytest.raises(IndexError):
        a = np.random.rand(10, 10, 10)
        b = np.random.rand(10, 10, 10)
        focal_correlation(a, b)

    # different shapes
    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 15)
        focal_correlation(a, b)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window="x")

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window=1)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window=11)

    # uneven window_shape is not supported
    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window=4)

    # Not exactly divided in reduce mode
    with pytest.raises((NotImplementedError, ValueError)):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window=4, reduce=True)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, fraction_accepted=-0.1)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, fraction_accepted=1.1)


def test_nan_behaviour():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)
    a[2, 2] = np.nan
    assert np.allclose(
        focal_correlation(a, b), focal_correlation_simple(a, b), equal_nan=True
    )
    assert np.isnan(focal_correlation(a, b)[2, 2])

    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)
    a[1, 1] = np.nan
    assert np.allclose(
        focal_correlation(a, b), focal_correlation_simple(a, b), equal_nan=True
    )
    assert not np.isnan(focal_correlation(a, b)[2, 2])
    assert np.isnan(focal_correlation(a, b, fraction_accepted=1)[2, 2])
    assert not np.isnan(focal_correlation(a, b, fraction_accepted=0)[2, 2])


def test_correlation_dtype():
    a = np.random.rand(5, 5).astype(np.int32)
    b = np.random.rand(5, 5).astype(np.int32)
    assert focal_correlation(a, b).dtype == np.float64

    a = np.random.rand(5, 5).astype(np.float64)
    b = np.random.rand(5, 5).astype(np.float64)
    assert focal_correlation(a, b).dtype == np.float64

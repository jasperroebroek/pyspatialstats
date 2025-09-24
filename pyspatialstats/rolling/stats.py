import numpy as np
from numpy.typing import ArrayLike

from pyspatialstats.rolling.window import rolling_window
from pyspatialstats.types.windows import WindowT
from pyspatialstats.windows import define_window


def rolling_sum(
    a: ArrayLike,
    *,
    window: WindowT,
    reduce: bool = False,
) -> np.ndarray:
    """
    Takes an array and returns the rolling sum. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : array-like
        Array to create the sliding window view from.
    window : int, array-like, Window
        Window that is applied over `a`. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a boolean array or a :class:`pyspatialstats.windows.Window` object.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the input array. If set to True, every entry is used exactly once, meaning that the sliding windows
        do not overlap each other. This creating much smaller output array.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling sum over array `a`. Resulting shape depends on reduce parameter. See :stat_func:`rolling_window` for
        documentation.
    """
    a = np.asarray(a)
    shape = np.asarray(a.shape)

    window = define_window(window)
    window.validate(reduce, allow_even=True, shape=shape)
    window_shape = window.get_shape(a.ndim)

    if window.masked or reduce:
        axis = -1 if window.masked else tuple(range(a.ndim, 2 * a.ndim))
        return rolling_window(a, window=window, reduce=reduce).sum(axis=axis)

    if np.issubdtype(a.dtype, np.bool_):
        dtype = np.intp
    else:
        dtype = a.dtype

    r = np.zeros(shape + 1, dtype=dtype)
    r[(slice(1, None),) * a.ndim] = a

    for i in range(a.ndim):
        if window_shape[i] == 1:
            continue
        else:
            ind1 = [slice(None)] * a.ndim
            ind1[i] = slice(window_shape[i], None)
            ind1 = tuple(ind1)
            ind2 = [slice(None)] * a.ndim
            ind2[i] = slice(None, -window_shape[i])
            ind2 = tuple(ind2)

            np.cumsum(r, axis=i, out=r)
            r[ind1] = r[ind1] - r[ind2]

    s = ()
    for i in range(a.ndim):
        s = s + (slice(window_shape[i], None),)

    return r[s]


def rolling_mean(
    a: ArrayLike,
    *,
    window: WindowT,
    reduce: bool = False,
) -> np.ndarray:
    """
    Takes an array and returns the rolling mean. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : array-like
        Array to create the sliding window view from.
    window : int, array-like, Window
        Window that is applied over `a`. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a boolean array or a :class:`pyspatialstats.windows.Window` object.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the input array. If set to True, every entry is used exactly once, meaning that the sliding windows
        do not overlap each other. This creating much smaller output array.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling mean over array `a`. Resulting shape depends on reduce parameter. See :stat_func:`rolling_window` for
        documentation.
    """
    a = np.asarray(a)
    shape = np.asarray(a.shape)

    window = define_window(window)
    window.validate(reduce, allow_even=True, shape=shape)

    window_shape = window.get_shape(a.ndim)

    div = window.get_mask(a.ndim).sum() if window.masked else np.prod(window_shape)

    return rolling_sum(a, window=window, reduce=reduce) / div

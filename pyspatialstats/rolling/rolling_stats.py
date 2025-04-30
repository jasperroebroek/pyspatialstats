from typing import Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import validate_call

from pyspatialstats.rolling.rolling_window import rolling_window
from pyspatialstats.types import Mask, PositiveInt
from pyspatialstats.windows import Window, define_window, validate_window


@validate_call(config={"arbitrary_types_allowed": True})
def rolling_sum(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    reduce: bool = False,
) -> NDArray:
    """
    Takes an array and returns the rolling sum. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : array_like
        Input array
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a :class:`pyspatialstats.window.Window` object.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see ``flatten`` in :func:`pyspatialstats.rolling.rolling_window`. If set to true,
        every entry is used exactly once, meaning that the sliding views do not overlap. This Creates much smaller
        output dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling sum over array ``a``. Resulting shape depends on reduce parameter. See :func:`rolling_window` for
        documentation.
    """
    a = np.asarray(a)
    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce)

    window_shape = np.asarray(window.get_shape(a.ndim))

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


@validate_call(config={"arbitrary_types_allowed": True})
def rolling_mean(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    reduce: bool = False,
) -> NDArray:
    """
    Takes an array and returns the rolling mean. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : array_like
        Input array
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a :class:`pyspatialstats.window.Window` object.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see ``flatten`` in :func:`pyspatialstats.rolling.rolling_window`. If set to true,
        every entry is used exactly once, meaning that the sliding views do not overlap. This Creates much smaller
        output dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling mean over array ``a``. Resulting shape depends on reduce parameter. See :func:`rolling_window` for
        documentation.
    """
    a = np.asarray(a)
    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce)

    window_shape = np.asarray(window.get_shape(a.ndim))

    div = window.get_mask(a.ndim).sum() if window.masked else np.prod(window_shape)

    return rolling_sum(a, window=window, reduce=reduce) / div

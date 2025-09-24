import numpy as np
from numpy.lib.stride_tricks import as_strided

from pyspatialstats.types.windows import WindowT
from pyspatialstats.windows import define_window


def rolling_window(
    a: np.ndarray,
    *,
    window: WindowT,
    flatten: bool = False,
    reduce: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Takes an array and returns a windowed version, similar to :stat_func:`numpy.lib.stride_tricks.as_strided`. If flatten is
    True, or a masked window is provided, the windowed view will be flattened, resulting in an array that has only one
    dimension more than the input array. This will require a copy of the data, increasing the memory usage. This can be
    problematic for large arrays and large window sizes.

    Parameters
    ----------
    a : array-like
        Array to create the sliding window view from.
    window : int, array-like, Window
        Window that is applied over `a`. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a boolean array or a :class:`pyspatialstats.windows.Window` object. The output will be of
        dimension ``a.ndim + a.ndim``. If a mask is provided (or a Window that is masked), it will be used to flatten
        `a_view`, resulting in dimensionality ``a.ndim + 1`` as the final cy_result, just as in the case of `flatten` is
        True.
    flatten : bool, optional
        Flag to flatten the windowed view to 1 dimension. If set to True, the dimensionality of the output will be
        ``a.ndim + 1``, otherwise it will be ``a.ndim + a.ndim``, which is the default.
        reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the input array. If set to True, every entry is used exactly once, meaning that the sliding windows
        do not overlap each other. This creating much smaller output array.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the input array. If set to True, every entry is used exactly once, meaning that the sliding windows
        do not overlap each other. This creating much smaller output array.
    kwargs : dict, optional
        Arguments for :stat_func:`~numpy.lib.stride_tricks.as_strided`, notably ``subok`` and ``writeable`` (see numpy
        documentation).

    Returns
    -------
    view :obj:`~numpy.ndarray`
        Sliding window view of the array. The sliding window dimensions are inserted at the end, and the original
        dimensions are trimmed as required by the size of the sliding window. That is, ``view.shape = x_shape_trimmed +
        window_shape``, where x_shape_trimmed is x.shape with every entry reduced by one less than the corresponding
        window size. If `flatten` is True or a masked window is provided the view will have shape ``x_shape_trimmed +
        np.prod(window_shape)``.
    """
    a = np.asarray(a)

    window = define_window(window)
    window.validate(reduce, allow_even=True, shape=a.shape)

    window_shape = window.get_shape(a.ndim)

    if reduce:
        output_shape = np.r_[np.floor_divide(a.shape, window_shape), window_shape]
        output_strides = np.r_[np.multiply(a.strides, window_shape), a.strides]

    else:
        output_shape = np.r_[np.subtract(a.shape, window_shape) + 1, window_shape]
        output_strides = np.r_[a.strides, a.strides]

    # create view on the data with new data_shape and strides
    strided_a = as_strided(a, shape=output_shape, strides=output_strides, **kwargs)

    if window.masked or flatten:
        return strided_a[..., window.get_mask(a.ndim)]

    return strided_a

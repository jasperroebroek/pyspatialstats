import numpy as np
from numpy.typing import ArrayLike

from pyspatialstats.enums import MajorityMode
from pyspatialstats.focal.core.majority import _focal_majority
from pyspatialstats.rolling import rolling_window
from pyspatialstats.types.arrays import RasterFloat64
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import create_output_array, parse_raster, timeit
from pyspatialstats.windows import define_window, validate_window


@timeit
def focal_majority(
    a: ArrayLike,
    *,
    window: WindowT,
    fraction_accepted: float = 0.7,
    verbose: bool = False,
    reduce: bool = False,
    majority_mode: MajorityMode = MajorityMode.NAN,
) -> RasterFloat64:
    """
    Focal majority

    Parameters
    ----------
    a : array-like
        Input array
    window : int, array-like, Window
        Window that is applied over `a`. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a boolean array or a :class:`pyspatialstats.window.Window` object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable

        * ``0``: all views are calculated if at least 1 value is present
        * ``1``: only views completely filled with values are calculated
        * ``0-1``: fraction of acceptability

    reduce : bool, optional
        Use all pixels exactly once, without windows overlapping. The resulting array will have the shape:
        ``a_shape / window_shape``
    verbose : bool, optional
        Verbosity with timing. False by default
    majority_mode : MajorityMode, optional
        Differt modes of dealing with more than one value occurring equally often:

            - ``NAN``: when more than one class has the same score NaN will be assigned
            - ``ASCENDING``: the first occurrence of the maximum count will be assigned
            - ``DESCENDING``: the last occurrence of the maximum count will be assigned

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the focal statistic. If `reduce` is set to False, the output has the same shape as the input,
        while if `reduce` is True, the output is reduced by the window size: ``raster_shape // window_shape``.
    """
    a = parse_raster(a)

    window = define_window(window)
    validate_window(window, a.shape, reduce, allow_even=False)
    mask = window.get_mask()

    fringe = window.get_fringes(reduce)
    ind_inner = window.get_ind_inner(reduce)
    threshold = window.get_threshold(fraction_accepted=fraction_accepted)

    r = create_output_array(a, window.get_raster_shape(), reduce)
    a_windowed = rolling_window(a, window=window, reduce=reduce)

    _focal_majority(
        a=a_windowed,
        mask=mask,
        values=np.empty(mask.sum(), dtype=a_windowed.dtype),
        r=r[ind_inner],
        fringe=np.asarray(fringe, dtype=np.int32),
        threshold=threshold,
        reduce=reduce,
        mode=majority_mode.value,
    )

    return r

from typing import Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import validate_call

from pyspatialstats.focal_stats.core.correlation import _focal_correlation
from pyspatialstats.types import Fraction, Mask, PositiveInt, RasterFloat64
from pyspatialstats.utils import parse_raster, timeit
from pyspatialstats.windows import Window, define_window, validate_window


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_correlation(
    a: NDArray,
    b: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window = 5,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
) -> RasterFloat64:
    """
    Focal correlation

    Parameters
    ----------
    a, b : array-like
        Input arrays that will be correlated. If not present in dtype :obj:`~numpy.float64` it will be converted
        internally. They need to have the same shape and have two dimensions.
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a custom ``Window`` object.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable

        * ``0``: all views are calculated if at least 1 value is present
        * ``1``: only views completely filled with values are calculated
        * ``0-1``: fraction of acceptability

    reduce : bool, optional
        Reuse all cells exactly once by setting a stepsize of the same size as window_shape. The resulting raster will
        have the raster_shape: ``raster_shape/window_shape``
    verbose : bool, optional
        Verbosity with timing. False by default

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the local correlation. If ``reduce`` is set to False, the output has the same raster_shape as the
        input raster, while if ``reduce`` is True, the output is reduced by the window size:
        ``raster_shape // window_shape``.
    """
    a = parse_raster(a)
    b = parse_raster(b)

    raster_shape = np.asarray(a.shape)

    if a.shape != b.shape:
        raise ValueError(f"Input arrays have different shapes: {a.shape=}, {b.shape=}")

    window = define_window(window)
    validate_window(window, raster_shape, reduce, allow_even=False)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    corr = _focal_correlation(
        a,
        b,
        window_shape=window_shape,
        mask=mask,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
    )

    return np.asarray(corr)

from functools import partial
from typing import Literal, Optional

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.enums import ErrorType
from pyspatialstats.focal.core.mean import _focal_mean, _focal_mean_bootstrap, _focal_mean_std
from pyspatialstats.focal._core import focal_stats, focal_stats_base
from pyspatialstats.focal.result_config import FocalMeanResultConfig
from pyspatialstats.results.stats import MeanResult
from pyspatialstats.types.arrays import Array
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import timeit


@timeit
def focal_mean(
    a: Array,
    *,
    window: WindowT,
    fraction_accepted: float = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    chunks: Optional[int | tuple[int, int]] = None,
    error: Optional[ErrorType] = None,
    bootstrap_config: Optional[BootstrapConfig] = None,
    out: Optional[MeanResult] = None,
) -> MeanResult:
    """
    Focal mean

    Parameters
    ----------
    a: Array
        Input array to compute the focal mean on. Must be two-dimensional.
    window : int, array-like, or Window
        Window applied over the input array. It can be:

        - An integer (interpreted as a square window),
        - A sequence of integers (interpreted as a rectangular window),
        - A boolean array,
        - Or a :class:`pyspatialstats.windows.Window` object.
    fraction_accepted : float, optional
        Fraction of valid (non-NaN) cells per window required for computation.

        - ``0``: all views are used if at least 1 value is present
        - ``1``: only fully valid views are used
        - Between ``0`` and ``1``: minimum fraction of valid cells required

        Default is 0.7.
    verbose : bool, optional
        If True, print progress message with timing. Default is False.
    reduce : bool, optional
        If True, uses each pixel exactly once without overlapping windows. The resulting array shape is
        ``a_shape / window_shape``. Default is False.
    chunks : int or tuple of int, optional
        Shape of chunks to split the array into. If None, the array is not split into chunks, which is the default.
    error : {'parametric', 'bootstrap'}, optional
        Error type to compute. If None, no error is computed, which is the default.
    bootstrap_config : BootstrapConfig, optional
        Bootstrap configuration object.
    out : MeanResult, optional
        MeanResult object to update in-place

    Returns
    -------
    MeanResult
        Dataclass containing the focal mean array and (optionally) uncertainty measures.
    """
    func_kwargs = {}

    match error:
        case 'parametric':
            stat_func = _focal_mean_std
        case 'bootstrap':
            stat_func = _focal_mean_bootstrap
            if bootstrap_config is None:
                bootstrap_config = BootstrapConfig()
            func_kwargs.update(**bootstrap_config.__dict__)
        case None:
            stat_func = _focal_mean
        case _:
            raise ValueError(f'Error not understood: {error}')

    return focal_stats(
        data={'a': a},
        func=partial(focal_stats_base, stat_func=stat_func, **func_kwargs),
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        chunks=chunks,
        result_config=FocalMeanResultConfig(error=error),
        out=out,
    )

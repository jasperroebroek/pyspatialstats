from functools import partial
from typing import Optional

import numpy as np

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.enums import Uncertainty
from pyspatialstats.focal.core.mean import _focal_mean, _focal_mean_bootstrap
from pyspatialstats.focal.core.std import _focal_std_means_precomputed
from pyspatialstats.focal.focal_core import focal_stats, focal_stats_base
from pyspatialstats.focal.result_config import FocalMeanResultConfig
from pyspatialstats.types.results import MeanResult
from pyspatialstats.types.arrays import Array
from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import timeit
from pyspatialstats.windows import Window, define_window


def _focal_mean_base(
    a: Array,
    *,
    window: Window,
    fraction_accepted: float,
    reduce: bool,
    bootstrap_config: Optional[BootstrapConfig],
    out: Optional[MeanResult],
    result_config: FocalMeanResultConfig,
) -> MeanResult:
    if result_config.uncertainty == Uncertainty.SE:
        if bootstrap_config is None:
            bootstrap_config = BootstrapConfig()

        return focal_stats_base(
            a,
            cy_func=_focal_mean_bootstrap,
            window=window,
            fraction_accepted=fraction_accepted,
            reduce=reduce,
            result_config=result_config,
            out=out,
            **bootstrap_config.__dict__,
        )

    mean = focal_stats_base(
        a,
        cy_func=_focal_mean,
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        out=out.mean if out is not None else None,
    )

    if result_config.uncertainty is None:
        return MeanResult(mean=mean)

    window = define_window(window)
    ind_inner = window.get_ind_inner(ndim=2, reduce=reduce)

    a_mean = mean[ind_inner]
    if not isinstance(a_mean, np.ndarray):
        a_mean = np.asarray(a_mean)

    std = focal_stats_base(
        a,
        cy_func=_focal_std_means_precomputed,
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        a_mean=a_mean,
        dof=0,
        out=out.std if out is not None else None,
    )

    return MeanResult(mean=mean, std=std)


@timeit
def focal_mean(
    a: Array,
    *,
    window: WindowT,
    fraction_accepted: float = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    chunks: Optional[int | tuple[int, int]] = None,
    uncertainty: Optional[Uncertainty] = None,
    bootstrap_config: Optional[BootstrapConfig] = None,
    out: Optional[MeanResult] = None,
) -> MeanResult:
    """
    Focal mean.

    Parameters
    ----------
    a: Array
        Input array to compute the focal mean on. Must be two-dimensional.
    window : int, array-like, or Window
        Window applied over the input array. It can be:

        - An integer (interpreted as a square window),
        - A sequence of integers (interpreted as a rectangular window),
        - A boolean array,
        - Or a :class:`pyspatialstats.window.Window` object.
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
    uncertainty : Uncertainty, optional
        Type of uncertainty to calculate. If None, no uncertainty is computed, which is the default.
    bootstrap_config : BootstrapConfig, optional
        Bootstrap configuration object. Required if uncertainty is set to use bootstrapping. Default is None.
    out : MeanResult, optional
        MeanResult object to update in-place

    Returns
    -------
    MeanResult
        Dataclass containing the focal mean array and (optionally) uncertainty measures.
    """
    return focal_stats(
        a,
        func=partial(_focal_mean_base, bootstrap_config=bootstrap_config),
        window=window,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
        chunks=chunks,
        result_config=FocalMeanResultConfig(uncertainty=uncertainty),
        out=out,
    )

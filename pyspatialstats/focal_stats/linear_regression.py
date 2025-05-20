from typing import Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import validate_call

from pyspatialstats.focal_stats.core.linear_regression import (
    CyFocalLinearRegressionResult,
    _focal_linear_regression,
)
from pyspatialstats.results import LinearRegressionResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.stat_utils import calculate_p_value
from pyspatialstats.types import Fraction, Mask, PositiveInt
from pyspatialstats.utils import (
    create_output_array,
    parse_raster,
    timeit,
)
from pyspatialstats.windows import (
    Window,
    define_window,
    validate_window,
)


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_linear_regression(
    a1: NDArray,
    a2: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,  # noqa
    reduce: bool = False,
    p_values: bool = False,
) -> LinearRegressionResult:
    a1 = parse_raster(a1)
    a2 = parse_raster(a2)

    if a1.shape != a2.shape:
        raise ValueError(
            f"Input arrays have different shapes: {a1.shape=}, {a2.shape=}"
        )

    window = define_window(window)
    validate_window(window, a1.shape, reduce, allow_even=False)
    mask = window.get_mask(2)

    fringe = window.get_fringes(reduce)
    ind_inner = window.get_ind_inner(reduce)
    threshold = fraction_accepted * mask.sum()

    df = create_output_array(a1, window.get_shape(), reduce, dtype=np.uintp)
    a = create_output_array(a1, window.get_shape(), reduce)
    b = create_output_array(a1, window.get_shape(), reduce)
    se_a = create_output_array(a1, window.get_shape(), reduce)
    se_b = create_output_array(a1, window.get_shape(), reduce)
    t_a = create_output_array(a1, window.get_shape(), reduce)
    t_b = create_output_array(a1, window.get_shape(), reduce)

    r = CyFocalLinearRegressionResult(
        df=df[ind_inner],
        a=a[ind_inner],
        b=b[ind_inner],
        se_a=se_a[ind_inner],
        se_b=se_b[ind_inner],
        t_a=t_a[ind_inner],
        t_b=t_b[ind_inner],
    )

    a1_windowed = rolling_window(a1, window=window, reduce=reduce)
    a2_windowed = rolling_window(a2, window=window, reduce=reduce)

    _focal_linear_regression(
        a1=a1_windowed,
        a2=a2_windowed,
        mask=window.get_mask(),
        r=r,
        fringe=fringe,
        threshold=threshold,
        reduce=reduce,
    )

    p_a = calculate_p_value(t_a, df) if p_values else None
    p_b = calculate_p_value(t_b, df) if p_values else None

    return LinearRegressionResult(
        a=a,
        b=b,
        se_a=se_a,
        se_b=se_b,
        t_a=t_a,
        t_b=t_b,
        p_a=p_a,
        p_b=p_b,
    )

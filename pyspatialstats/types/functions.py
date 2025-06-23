from typing import Optional, Protocol

from numpy._typing import ArrayLike

from pyspatialstats.types.results import StatResult
from pyspatialstats.types.arrays import Array, RasterFloat64, RasterSizeT
from pyspatialstats.types.windows import WindowT


class FocalStatsFunctionSingle(Protocol):
    def __call__(
        self,
        a: ArrayLike,
        *,
        window: WindowT,
        fraction_accepted: float,
        verbose: bool,
        reduce: bool,
        out: Optional[StatResult | Array] = None,
    ) -> RasterFloat64 | RasterSizeT | StatResult: ...


class FocalStatsFunctionDouble(Protocol):
    def __call__(
        self,
        a1: ArrayLike,
        a2: ArrayLike,
        *,
        window: WindowT,
        fraction_accepted: float,
        verbose: bool,
        reduce: bool,
        out: Optional[StatResult | Array] = None,
    ) -> RasterFloat64 | RasterSizeT | StatResult: ...


FocalStatsFunction = FocalStatsFunctionSingle | FocalStatsFunctionDouble

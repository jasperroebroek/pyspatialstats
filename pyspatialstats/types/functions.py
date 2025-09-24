from typing import Optional, Protocol

from numpy.typing import ArrayLike

from pyspatialstats.results.stats import StatResult
from pyspatialstats.types.arrays import Array, RasterFloat64, RasterSizeT
from pyspatialstats.types.windows import WindowT


class FocalStatsFunction(Protocol):
    def __call__(
        self,
        *args: ArrayLike,
        window: WindowT,
        fraction_accepted: float,
        reduce: bool,
        out: Optional[StatResult | Array] = None,
        **kwargs,
    ) -> RasterFloat64 | RasterSizeT | StatResult: ...

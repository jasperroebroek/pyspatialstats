from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pyspatialstats.types.results import FocalStatResults, StatResult
from pyspatialstats.types.arrays import Array, RasterT
from pyspatialstats.views import RasterViewPair
from pyspatialstats.windows import Window


class FocalResultConfig(ABC):
    @property
    @abstractmethod
    def return_type(self) -> type[FocalStatResults]:
        pass

    @property
    @abstractmethod
    def fields(self) -> tuple[str, ...]:
        pass

    @property
    def active_fields(self) -> tuple[str, ...]:
        return self.fields

    @property
    def cy_fields(self) -> tuple[str, ...]:
        return self.fields

    @abstractmethod
    def create_output(self, a: Array, window: Window, reduce: bool) -> FocalStatResults:
        pass

    @abstractmethod
    def create_tile_output(
        self, out: Array, tile_view: RasterViewPair, window: Window, reduce: bool
    ) -> Optional[FocalStatResults]:
        pass

    @abstractmethod
    def validate_output(self, a: Array, window: Window, reduce: bool, out: FocalStatResults) -> None:
        pass

    @abstractmethod
    def get_cython_input(self, a: Array, window: Window, reduce: bool, out: FocalStatResults) -> dict[str, RasterT]:
        pass

    @abstractmethod
    def write_output(
        self, window: Window, reduce: bool, out: FocalStatResults, cy_result: dict[str, np.ndarray]
    ) -> FocalStatResults:
        pass

    @abstractmethod
    def write_tile_output(
        self,
        window: Window,
        reduce: bool,
        out: FocalStatResults,
        result: np.ndarray | StatResult,
        tile_view: RasterViewPair,
    ) -> None:
        pass

    def parse_output(self, a: Array, out: Optional[FocalStatResults], window: Window, reduce: bool) -> FocalStatResults:
        if out is not None:
            self.validate_output(a=a, out=out, window=window, reduce=reduce)
            return out
        return self.create_output(a=a, window=window, reduce=reduce)

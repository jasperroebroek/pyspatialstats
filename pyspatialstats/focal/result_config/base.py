from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pyspatialstats.results.stats import FocalStatResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.types.arrays import Array, RasterShape, RasterT
from pyspatialstats.utils import get_dtype, parse_array
from pyspatialstats.views import ArrayViewPair
from pyspatialstats.windows import Window


class FocalResultConfig(ABC):
    @property
    @abstractmethod
    def return_type(self) -> type[FocalStatResult]:
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
    def create_output(self, raster_shape: RasterShape, window: Window, reduce: bool) -> FocalStatResult:
        """Method to create the output array/dataclass for the focal statistics. This method should return an instance
        of the return_type. The output is of the shape of the raster_shape (or smaller with reduce=True). Meant for
        creating the combined output for all tiles if tiling is enabled."""
        pass

    @abstractmethod
    def create_tile_output(
        self, out: FocalStatResult, tile_view: ArrayViewPair, window: Window, reduce: bool
    ) -> Optional[FocalStatResult]:
        """Method to create the output array/dataclass for the focal statistics. This method should return an instance
        of the return_type. The output is of the shape of the tile_view (or smaller with reduce=True). Meant for
        creating the output for a single tile"""
        pass

    @abstractmethod
    def validate_output(self, raster_shape: RasterShape, window: Window, reduce: bool, out: FocalStatResult) -> None:
        """Method to validate the output array/dataclass for the focal statistics. This method should raise an exception
        if the output is not valid, either shape, dimensions, or dtype."""
        pass

    @abstractmethod
    def get_cython_input(
        self, raster_shape: RasterShape, window: Window, reduce: bool, out: FocalStatResult
    ) -> dict[str, RasterT]:
        """Method to get the cython input for the focal statistics. This method should return a dictionary with the
        input arrays/dataclasses for the cython function, meaning windowed."""
        pass

    @abstractmethod
    def write_output(
        self, window: Window, reduce: bool, out: FocalStatResult, cy_result: dict[str, np.ndarray]
    ) -> FocalStatResult:
        """Method to write the cython output to the output array/dataclass."""
        pass

    def parse_output(
        self, raster_shape: RasterShape, out: Optional[FocalStatResult], window: Window, reduce: bool
    ) -> FocalStatResult:
        """Create the output array/dataclass for the focal statistics if it is not already created (out is None),
        otherwise validate the output."""
        if out is not None:
            self.validate_output(raster_shape=raster_shape, out=out, window=window, reduce=reduce)
            return out
        return self.create_output(raster_shape=raster_shape, window=window, reduce=reduce)

    def get_dtype(self, name: str) -> np.dtype:
        """Get the dtype for a field."""
        return get_dtype(name)

    def get_ndim(self, name: str) -> int:
        """Get the number of dimensions for a field."""
        return 2

    def parse_array(self, name: str, array: Array) -> Array:
        """Parse an array to the correct dtype and ndim."""
        return parse_array(array, ndim=self.get_ndim(name), dtype=self.get_dtype(name))

    def window_data(self, data: dict[str, Array], window: Window, reduce: bool) -> dict[str, Array]:
        """Window the data arrays in the data dictionary."""
        window_shape = window.get_raster_shape()
        return {k: rolling_window(r, window=window_shape, reduce=reduce) for k, r in data.items()}

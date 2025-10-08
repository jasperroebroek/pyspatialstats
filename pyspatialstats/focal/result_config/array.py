from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike

from pyspatialstats.focal.result_config import FocalResultConfig
from pyspatialstats.focal.utils import create_output_array
from pyspatialstats.types.arrays import Array, RasterShape, RasterT
from pyspatialstats.utils import validate_array
from pyspatialstats.views import ArrayViewPair
from pyspatialstats.windows import Window


@dataclass
class FocalArrayResultConfig(FocalResultConfig):
    dtype: DTypeLike = np.float64

    @property
    def return_type(self) -> type[Array]:
        return Array

    @property
    def fields(self) -> tuple[str, ...]:
        return ('r',)

    def create_output(self, raster_shape: RasterShape, window: Window, reduce: bool) -> Array:
        shape = window.define_windowed_shape(reduce, shape=raster_shape)
        return create_output_array(shape, self.dtype)

    def create_tile_output(self, out: Array, tile_view: ArrayViewPair, window: Window, reduce: bool) -> Optional[Array]:
        if not isinstance(out, Array):
            raise TypeError(f'Expected Array but got {type(out).__name__}')
        return out[tuple(tile_view.output.get_external_slices(window, reduce))]

    def validate_output(self, raster_shape: RasterShape, window: Window, reduce: bool, out: Array) -> None:
        if not isinstance(out, self.return_type):
            raise TypeError(f'Expected Array but got {type(out).__name__}')
        expected_shape = window.define_windowed_shape(reduce=reduce, shape=raster_shape)
        validate_array('r', out, expected_shape)

    def get_cython_input(
        self, raster_shape: RasterShape, window: Window, reduce: bool, out: Array
    ) -> dict[str, RasterT]:
        ind_inner = window.get_ind_inner(ndim=2, reduce=reduce)
        if not isinstance(out, np.ndarray):
            out = self.create_output(raster_shape, window, reduce)
        return {self.fields[0]: out[ind_inner]}

    def write_output(self, window: Window, reduce: bool, out: Array, cy_result: dict[str, np.ndarray]) -> Array:
        ind_inner = window.get_ind_inner(reduce=reduce, ndim=2)
        if np.shares_memory(out, cy_result):
            return out
        out[ind_inner] = cy_result[self.fields[0]]
        return out

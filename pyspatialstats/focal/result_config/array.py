from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy._typing import DTypeLike

from pyspatialstats.focal.result_config import FocalResultConfig
from pyspatialstats.focal.utils import create_output_raster
from pyspatialstats.types.results import StatResult
from pyspatialstats.types.arrays import Array, RasterT
from pyspatialstats.utils import validate_raster
from pyspatialstats.views import RasterViewPair
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

    def create_output(self, a: Array, window: Window, reduce: bool) -> Array:
        shape = window.define_windowed_shape(reduce, a=a)
        return create_output_raster(shape, self.dtype)

    def create_tile_output(
        self, out: Array, tile_view: RasterViewPair, window: Window, reduce: bool
    ) -> Optional[Array]:
        if not isinstance(out, Array):
            raise TypeError(f'Expected Array but got {type(out).__name__}')
        if isinstance(out, np.ndarray):
            return out[*tile_view.output.get_external_slices(window, reduce)]
        return None

    def validate_output(self, a: Array, window: Window, reduce: bool, out: Array) -> None:
        if not isinstance(out, self.return_type):
            raise TypeError(f'Expected Array but got {type(out).__name__}')
        expected_shape = window.define_windowed_shape(reduce=reduce, a=a)[:2]
        validate_raster('r', out, expected_shape)

    def get_cython_input(self, a: Array, window: Window, reduce: bool, out: Array) -> dict[str, RasterT]:
        ind_inner = window.get_ind_inner(ndim=2, reduce=reduce)
        if isinstance(out, np.ndarray):
            r = out[ind_inner]
        else:
            r = self.create_output(a, window, reduce)[ind_inner]
        return {self.fields[0]: r}

    def write_output(self, window: Window, reduce: bool, out: Array, cy_result: dict[str, np.ndarray]) -> Array:
        ind_inner = window.get_ind_inner(reduce=reduce, ndim=2)
        if np.shares_memory(out, cy_result):
            return out
        out[ind_inner] = cy_result[self.fields[0]]
        return out

    def write_tile_output(
        self, window: Window, reduce: bool, out: Array, result: np.ndarray | StatResult, tile_view: RasterViewPair
    ) -> None:
        ind_inner = window.get_ind_inner(ndim=2, reduce=reduce)
        out[*tile_view.output.slices] = result[ind_inner]

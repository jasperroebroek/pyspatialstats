from abc import ABC
from dataclasses import fields

import numpy as np

from pyspatialstats.focal.result_config.base import FocalResultConfig
from pyspatialstats.focal.utils import create_output_array
from pyspatialstats.results.stats import FocalStatResult, StatResult
from pyspatialstats.types.arrays import RasterShape, RasterT
from pyspatialstats.utils import validate_array
from pyspatialstats.views import ArrayViewPair
from pyspatialstats.windows import Window


class FocalDataClassResultConfig(FocalResultConfig, ABC):
    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self.return_type))

    def create_output(self, raster_shape: RasterShape, window: Window, reduce: bool) -> StatResult:
        return self.return_type(
            **{
                field: create_output_array(
                    shape=self.get_output_shape(field, raster_shape, window, reduce),
                    dtype=self.get_dtype(field),
                )
                for field in self.active_fields
            }
        )

    def create_tile_output(self, out: StatResult, tile_view: ArrayViewPair, window: Window, reduce: bool) -> StatResult:
        if not isinstance(out, StatResult):
            raise TypeError(f'Expected StatResult but got {type(out).__name__}')

        return self.return_type(
            **{
                field: getattr(out, field)[tuple(tile_view.output.get_external_slices(window, reduce))]
                for field in self.active_fields
            }
        )

    def validate_output(self, raster_shape: RasterShape, window: Window, reduce: bool, out: StatResult) -> None:
        if not isinstance(out, self.return_type):
            raise TypeError(f'Expected StatResult but got {type(out).__name__}')
        for field in self.fields:
            validate_array(
                name=field,
                r=getattr(out, field),
                expected_shape=self.get_output_shape(field, raster_shape, window, reduce),
                dtype=self.get_dtype(field),
            )

    def get_output_shape(self, name: str, raster_shape: RasterShape, window: Window, reduce: bool) -> RasterShape:
        return window.define_windowed_shape(reduce=reduce, shape=raster_shape)

    def get_cython_input(
        self, raster_shape: RasterShape, window: Window, reduce: bool, out: FocalStatResult
    ) -> dict[str, RasterT]:
        ind_inner = window.get_ind_inner(ndim=2, reduce=reduce)
        cython_input = {}
        for field in self.cy_fields:
            if isinstance(getattr(out, field), np.ndarray):
                cython_input[field] = getattr(out, field)[ind_inner]
            else:
                shape = self.get_output_shape(field, raster_shape, window, reduce)
                cython_input[field] = create_output_array(shape, self.get_dtype(field))[ind_inner]
        return cython_input

    def write_output(
        self, window: Window, reduce: bool, out: StatResult, cy_result: dict[str, np.ndarray]
    ) -> StatResult:
        ind_inner = window.get_ind_inner(reduce=reduce, ndim=2)
        for field in self.cy_fields:
            if np.shares_memory(getattr(out, field), cy_result[field]):
                continue
            getattr(out, field)[ind_inner] = cy_result[field]
        return out

from abc import ABC
from dataclasses import fields

import numpy as np

from pyspatialstats.focal.result_config.base import FocalResultConfig
from pyspatialstats.focal.utils import create_output_raster
from pyspatialstats.types.results import FocalStatResults, StatResult
from pyspatialstats.types.arrays import Array, RasterT
from pyspatialstats.utils import get_dtype, validate_raster
from pyspatialstats.views import RasterViewPair
from pyspatialstats.windows import Window


class FocalDataClassResultConfig(FocalResultConfig, ABC):
    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self.return_type))

    def create_output(self, a: Array, window: Window, reduce: bool) -> StatResult:
        shape = window.define_windowed_shape(reduce, a=a)
        return self.return_type(
            **{field: create_output_raster(shape=shape, dtype=get_dtype(field)) for field in self.active_fields}
        )

    def create_tile_output(
        self, out: StatResult, tile_view: RasterViewPair, window: Window, reduce: bool
    ) -> StatResult:
        if not isinstance(out, StatResult):
            raise TypeError(f'Expected StatResult but got {type(out).__name__}')

        fields_subset = {}
        for field in self.active_fields:
            if isinstance(getattr(out, field), np.ndarray):
                fields_subset[field] = getattr(out, field)[*tile_view.output.get_external_slices(window, reduce)]
            else:
                fields_subset[field] = create_output_raster(
                    tile_view.output.get_external_shape(window, reduce), dtype=get_dtype(field)
                )

        return self.return_type(**fields_subset)

    def validate_output(
        self,
        a: Array,
        window: Window,
        reduce: bool,
        out: StatResult,
    ) -> None:
        if not isinstance(out, self.return_type):
            raise TypeError(f'Expected StatResult but got {type(out).__name__}')
        expected_shape = window.define_windowed_shape(reduce=reduce, a=a)
        expected_raster_shape = (expected_shape[0], expected_shape[1])
        for field in self.fields:
            validate_raster(field, getattr(out, field), expected_raster_shape)

    def get_cython_input(self, a: Array, window: Window, reduce: bool, out: FocalStatResults) -> dict[str, RasterT]:
        ind_inner = window.get_ind_inner(ndim=2, reduce=reduce)
        cython_input = {}
        for field in self.cy_fields:
            if isinstance(getattr(out, field), np.ndarray):
                cython_input[field] = getattr(out, field)[ind_inner]
            else:
                shape = window.define_windowed_shape(reduce, a=a)
                cython_input[field] = create_output_raster(shape, get_dtype(field))[ind_inner]
        return cython_input

    def write_output(self, window: Window, reduce: bool, out: StatResult, cy_result: dict[str, np.ndarray]) -> StatResult:
        ind_inner = window.get_ind_inner(reduce=reduce, ndim=2)
        for field in self.cy_fields:
            if np.shares_memory(getattr(out, field), cy_result[field]):
                continue
            getattr(out, field)[ind_inner] = cy_result[field]
        return out

    def write_tile_output(
        self, window: Window, reduce: bool, out: StatResult, result: StatResult, tile_view: RasterViewPair
    ) -> None:
        ind_inner = window.get_ind_inner(ndim=2, reduce=reduce)
        for field in self.active_fields:
            if np.shares_memory(getattr(out, field), getattr(result, field)):
                continue
            getattr(out, field)[*tile_view.output.slices] = getattr(result, field)[ind_inner]

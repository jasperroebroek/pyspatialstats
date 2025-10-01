from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import DTypeLike

from pyspatialstats.enums import ErrorType
from pyspatialstats.focal.result_config.dataclass import FocalDataClassResultConfig
from pyspatialstats.results.stats import CorrelationResult, MeanResult, RegressionResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.types.arrays import Array, RasterShape, Shape
from pyspatialstats.utils import parse_array
from pyspatialstats.windows import Window


@dataclass
class FocalMeanResultConfig(FocalDataClassResultConfig):
    error: Optional[Literal['bootstrap', 'parametric']] = None

    def __post_init__(self):
        if self.error is not None and self.error not in ('bootstrap', 'parametric'):
            raise ValueError(f'Error not understood: {self.error}')

    @property
    def return_type(self) -> type[MeanResult]:
        return MeanResult

    @property
    def active_fields(self) -> tuple[str, ...]:
        if self.error == 'bootstrap':
            return ('mean', 'se')
        if self.error == 'parametric':
            return ('mean', 'std')
        return ('mean',)

    @property
    def cy_fields(self) -> tuple[str, ...]:
        return self.active_fields


@dataclass
class FocalCorrelationResultConfig(FocalDataClassResultConfig):
    p_values: bool = False

    @property
    def return_type(self) -> type[CorrelationResult]:
        return CorrelationResult

    @property
    def active_fields(self) -> tuple[str, ...]:
        fields = tuple()
        for field in self.fields:
            if field == 'p' and not self.p_values:
                continue
            fields += (field,)
        return fields

    @property
    def cy_fields(self) -> tuple[str, ...]:
        return tuple(field for field in self.fields if field != 'p')


@dataclass
class FocalLinearRegressionResultConfig(FocalDataClassResultConfig):
    nf: int = 2
    x_ndim: int = 2
    error: Optional[ErrorType] = None

    def __post_init__(self):
        if self.error is not None and self.error not in ('bootstrap', 'parametric'):
            raise ValueError(f'Error not understood: {self.error}')

    @property
    def return_type(self) -> type[RegressionResult]:
        return RegressionResult

    @property
    def active_fields(self) -> tuple[str, ...]:
        match self.error:
            case 'parametric':
                return tuple(field for field in self.fields if field != 'r_squared_se')
            case 'bootstrap':
                return self.fields
            case None:
                return ('df', 'beta')

    @property
    def cy_fields(self) -> tuple[str, ...]:
        return self.active_fields

    def get_dtype(self, name: str) -> DTypeLike:
        return np.float64

    def get_ndim(self, name: str) -> int:
        if name == 'x':
            return self.x_ndim
        elif name in ('beta', 'beta_se', 't', 'p'):
            return 3
        elif name in ('y', 'df', 'r_squared', 'r_squared_se'):
            return 2
        else:
            raise ValueError(f'Unknown field: {name}')

    def get_output_shape(self, name: str, raster_shape: RasterShape, window: Window, reduce: bool) -> Shape:
        shape = window.define_windowed_shape(reduce=reduce, shape=raster_shape) + (self.nf - (name == 'x'),)
        return shape[: self.get_ndim(name)]

    def parse_array(self, name: str, array: Array) -> Array:
        r = parse_array(array, ndim=self.get_ndim(name), dtype=self.get_dtype(name))
        if name == 'x' and self.x_ndim == 2:
            r = r[..., np.newaxis]
        return r

    def window_data(self, data: dict[str, Array], window: Window, reduce: bool) -> dict[str, Array]:
        window_shape = {'y': window.get_raster_shape(), 'x': window.get_raster_shape() + (self.nf - 1,)}
        return {k: rolling_window(r, window=window_shape[k], reduce=reduce) for k, r in data.items()}

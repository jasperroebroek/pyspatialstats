import numpy as np
from numpy.typing import DTypeLike

from pyspatialstats.types.arrays import RasterT, Shape


def create_output_array(shape: Shape, dtype: DTypeLike = np.float64) -> RasterT:
    if len(shape) not in (2, 3):
        raise ValueError(f'Invalid raster shape {shape}')
    fill_value = np.nan if np.issubdtype(dtype, np.floating) else 0
    return np.full(shape, dtype=dtype, fill_value=fill_value)

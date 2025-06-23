import numpy as np
from numpy._typing import _DTypeLike, _ShapeLike

from pyspatialstats.types.arrays import RasterT


def create_output_raster(
    shape: _ShapeLike,
    dtype: _DTypeLike = np.float64,
) -> RasterT:
    if len(shape) != 2:
        raise ValueError(f'Invalid raster shape {shape}')
    fill_value = np.nan if np.issubdtype(dtype, np.floating) else 0
    return np.full((shape[0], shape[1]), dtype=dtype, fill_value=fill_value)

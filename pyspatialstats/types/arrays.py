import numpy as np
from numpy._typing._dtype_like import _SCT

# ARRAYS
# generic ndarray types
Mask = np.ndarray[tuple[int, ...], np.bool_]
VectorInt32 = np.ndarray[tuple[int], np.int32]
# specific raster types
RasterFloat32 = np.ndarray[tuple[int, int], np.float32]
RasterFloat64 = np.ndarray[tuple[int, int], np.float64]
RasterInt32 = np.ndarray[tuple[int, int], np.int32]
RasterInt64 = np.ndarray[tuple[int, int], np.int64]
RasterSizeT = np.ndarray[tuple[int, int], np.uintp]
RasterBool = np.ndarray[tuple[int, int], np.bool_]
# generic raster types
RasterT = np.ndarray[tuple[int, int], np.dtype[_SCT]]
RasterNumeric = RasterInt32 | RasterInt64 | RasterFloat32 | RasterFloat64

from typing import Protocol, runtime_checkable

import numpy as np
from numpy._typing._dtype_like import _SCT, _DTypeLike

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


@runtime_checkable
class Array(Protocol):
    @property
    def dtype(self) -> _DTypeLike: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    def __getitem__(self, index: tuple[slice, ...]) -> np.ndarray[tuple[int, ...], np.dtype[_SCT]]: ...
    def __setitem__(self, index: tuple[slice, ...], value: np.ndarray[tuple[int, ...], np.dtype[_SCT]]) -> None: ...

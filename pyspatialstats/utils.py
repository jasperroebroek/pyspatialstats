import time
from functools import wraps
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from pyspatialstats.results.stats import StatResult
from pyspatialstats.types.arrays import Array, RasterNumeric, Shape


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        verbose = kwargs.get('verbose', False)

        if verbose:
            print_args = []

            for arg in args:
                if isinstance(arg, np.ndarray):
                    print_args.append(f'ndarray({arg.shape})')

            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    print_args.append(f'{key}=ndarray({value.shape})')
                elif isinstance(value, StatResult):
                    print_args.append(f'{key}=StatResult({value.shape})')
                else:
                    print_args.append(f'{key}={value}')

            print(f'{func.__name__}({", ".join(print_args)}) took {total_time:.4f} seconds')

        return result

    return timeit_wrapper


def parse_array(a: ArrayLike, ndim: int, dtype: Optional[DTypeLike] = None) -> np.ndarray:
    a_parsed = np.asarray(a, dtype=dtype)

    if a_parsed.ndim != ndim:
        raise IndexError(f'Only {ndim}D data is supported, found {a_parsed.ndim}')
    if dtype is None and a_parsed.dtype not in (np.float32, np.float64, np.int32, np.int64):
        raise TypeError(f'Unsupported data type {a.dtype=}')

    return a_parsed


def parse_raster(a: ArrayLike, dtype: Optional[DTypeLike] = None) -> RasterNumeric:
    """Convert to 2D array"""
    return parse_array(a, ndim=2, dtype=dtype)


def get_dtype(name: str) -> DTypeLike:
    if name in ('ind', 'count', 'df'):
        return np.uintp
    else:
        return np.float64


def validate_array(name: str, r: Optional[Array], expected_shape: Shape, dtype: Optional[DTypeLike] = None) -> None:
    if r is None:
        return
    if not isinstance(r, Array):
        raise TypeError(f'Expected array-like but got {type(r).__name__}')
    if not np.allclose(r.shape, expected_shape):
        raise ValueError(f'Shape {r.shape} does not match expected shape {expected_shape} for {name}')
    if dtype is None:
        dtype = get_dtype(name)
    if not np.isdtype(r.dtype, dtype):
        raise ValueError(f'Wrong dtype, got {r.dtype} and expected {dtype}, for array {name}')

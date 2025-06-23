import time
from functools import wraps
from typing import Optional

import numpy as np
from numpy._typing import ArrayLike, DTypeLike
from numpy._typing._shape import _ShapeLike

from pyspatialstats.types.results import StatResult
from pyspatialstats.types.arrays import Array, RasterNumeric


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
                    print_args.append(f'{key}=StatResult({value.get_shape()})')
                else:
                    print_args.append(f'{key}={value}')

            print(f'{func.__name__}({", ".join(print_args)}) Took {total_time:.4f} seconds')

        return result

    return timeit_wrapper


def parse_raster(a: ArrayLike) -> RasterNumeric:
    """Convert to 2D array"""
    a_parsed = np.asarray(a)

    if a_parsed.ndim != 2:
        raise IndexError('Only 2D data is supported')
    if a_parsed.dtype not in (np.float32, np.float64, np.int32, np.int64):
        raise TypeError(f'Unsupported data type {a.dtype=}')

    return a_parsed


def get_dtype(name: str) -> DTypeLike:
    if name in ('ind', 'count', 'df'):
        return np.uintp
    else:
        return np.float64


def validate_raster(name: str, r: Optional[Array], expected_shape: _ShapeLike) -> None:
    if r is None:
        return
    if not isinstance(r, Array):
        raise TypeError(f'Expected array-like but got {type(r).__name__}')
    if not np.allclose(r.shape, expected_shape):
        raise ValueError(f'Shape {r.shape} does not match expected shape {expected_shape} for {name}')
    if not np.isdtype(r.dtype, get_dtype(name)):
        raise ValueError(f'Wrong dtype, got {r.dtype} and expected {get_dtype(name)}')

import time
from functools import wraps

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from pyspatialstats.types.arrays import (
    RasterNumeric,
    RasterT,
)


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


def define_output_shape(a: np.ndarray, window_shape: tuple[int, ...], reduce: bool) -> tuple[int, ...]:
    a_shape = np.asarray(a.shape)
    window_shape = np.asarray(window_shape)

    if a_shape.size != window_shape.size:
        raise ValueError('a and window_shape must have the same number of dimensions')

    return tuple(a_shape // window_shape) if reduce else a.shape


def create_output_array(
    a: RasterT,
    window_shape: tuple[int, int],
    reduce: bool,
    dtype: DTypeLike = np.float64,
) -> RasterT:
    shape = define_output_shape(a, window_shape, reduce)
    fill_value = np.nan if np.issubdtype(dtype, np.floating) else 0
    return np.full(shape, dtype=dtype, fill_value=fill_value)

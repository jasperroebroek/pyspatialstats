"""
This module defines the definitions of the views in the sliding window methods and focal statistics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyspatialstats.types.arrays import Array, Mask, Shape


class Window(ABC):
    """Abstract base class for windows"""

    @abstractmethod
    def get_shape(self, ndim: int = 2) -> tuple[int, ...]:
        pass

    def get_raster_shape(self) -> tuple[int, int]:
        return self.get_shape()[0], self.get_shape()[1]

    @abstractmethod
    def get_mask(self, ndim: int = 2) -> Mask:
        pass

    @property
    @abstractmethod
    def masked(self) -> bool:
        pass

    def get_fringes(self, reduce: bool, ndim: int = 2) -> tuple[int, ...]:
        """Get the fringes of the window, i.e. the number of pixels between the center and the edge of the window"""
        if reduce:
            return tuple(0 for _ in range(ndim))
        return tuple(x // 2 for x in self.get_shape(ndim))

    def get_ind_inner(self, reduce: bool, ndim: int = 2) -> tuple[slice, ...]:
        """"Numpy compatible slices to remove the fringes from an array, where the values are NaN"""
        if reduce:
            return (slice(None),) * ndim

        return tuple(slice(fringe, -fringe) for fringe in self.get_fringes(reduce, ndim))

    def get_threshold(self, fraction_accepted: float = 0.7, ndim: int = 2) -> float:
        """Minimum amount of data points necessary to calculate the statistic in the window"""
        if fraction_accepted < 0 or fraction_accepted > 1:
            raise ValueError('fraction_accepted must between 0 and 1')
        return max(fraction_accepted * self.get_mask(ndim).sum(), 1)

    def validate(
        self,
        reduce: bool,
        allow_even: bool = False,
        a: Optional[Array] = None,
        shape: Optional[Shape] = None,
    ) -> None:
        """"Validate the window for a given array (`a`) or shape"""

        if a is None and shape is None:
            raise ValueError('Neither `a` nor shape are given')
        if a is not None and shape is not None:
            raise ValueError('Both `a` and shape are given')

        shape = a.shape if a is not None else shape
        window_shape = self.get_shape(len(shape))

        if np.any(np.less(shape, window_shape)):
            raise ValueError(f'Window bigger than input array: {shape=}, {self=}')

        if reduce:
            if not np.all(np.remainder(shape, window_shape) == 0):
                raise ValueError('not all dimensions are divisible by window_shape')

        if not allow_even and not reduce:
            if np.any(np.remainder(window_shape, 2) == 0):
                raise ValueError('Uneven window size is not allowed when not reducing')

        if all((ws == 1 for ws in window_shape)):
            raise ValueError(f'Window size cannot only contain 1s {window_shape=}')

    def define_windowed_shape(self, reduce: bool, a: Optional[Array] = None, shape: Optional[Shape] = None) -> Shape:
        """Define the shape of the windowed array"""

        if a is None and shape is None:
            raise ValueError('Neither a nor shape are given')

        ndim = a.ndim if a is not None else len(shape)
        shape = a.shape if a is not None else shape
        window_shape = self.get_shape(ndim=ndim)

        if len(shape) != len(window_shape):
            raise ValueError('a and window_shape must have the same number of dimensions')

        return tuple(np.floor_divide(shape, window_shape)) if reduce else shape


@dataclass
class RectangularWindow(Window):
    window_size: int | tuple[int, ...]

    def get_shape(self, ndim: int = 2) -> tuple[int, ...]:
        if isinstance(self.window_size, int):
            return (self.window_size,) * ndim

        if len(self.window_size) != ndim:
            raise IndexError(f'dimensions do not match the size of the window: {ndim=} window_size={self.window_size}')

        return self.window_size

    def get_mask(self, ndim: int = 2) -> Mask:
        return np.ones(self.get_shape(ndim), dtype=np.bool_)

    @property
    def masked(self) -> bool:
        return False


@dataclass
class MaskedWindow(Window):
    mask: Mask

    def __post_init__(self):
        if self.mask.sum() == 0:
            raise ValueError('Mask cannot be empty')

    def match_shape(self, ndim: int) -> None:
        if self.mask.ndim != ndim:
            raise IndexError(f'dimensions do not match the size of the mask: {ndim=} {self.mask.ndim=}')

    def get_shape(self, ndim: int = 2) -> tuple[int, ...]:
        self.match_shape(ndim)
        return self.mask.shape

    def get_mask(self, ndim: int = 2) -> Mask:
        self.match_shape(ndim)
        return self.mask

    @property
    def masked(self) -> bool:
        return True


def define_window(window: int | tuple[int, ...] | list[int] | Mask | Window) -> Window:
    if isinstance(window, Window):
        return window
    if isinstance(window, np.ndarray) and np.issubdtype(window.dtype, np.bool_):
        return MaskedWindow(mask=window)
    if isinstance(window, (int, tuple, list)):
        return RectangularWindow(window_size=window)

    raise TypeError(f"Window can't be parsed from {window}. Must be int, tuple of int or binary array")

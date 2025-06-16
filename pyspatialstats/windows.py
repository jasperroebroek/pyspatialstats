"""
This module defines the definitions of the views in the sliding window methods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy._typing._shape import _ShapeLike

from pyspatialstats.types.arrays import Mask


class Window(ABC):
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
        if reduce:
            return tuple(0 for _ in range(ndim))
        return tuple(x // 2 for x in self.get_shape(ndim))

    def get_ind_inner(self, reduce: bool, ndim: int = 2) -> tuple[slice, ...]:
        if reduce:
            return (slice(None),) * ndim

        return tuple(slice(fringe, -fringe) for fringe in self.get_fringes(reduce, ndim))

    def get_threshold(self, fraction_accepted: float = 0.7, ndim: int = 2) -> float:
        """Minimum amount of data points necessary to calculate the statistic in the window"""
        if fraction_accepted < 0 or fraction_accepted > 1:
            raise ValueError('fraction_accepted must between 0 and 1')
        return max(fraction_accepted * self.get_mask(ndim).sum(), 1)


@dataclass
class RectangularWindow(Window):
    window_size: int | tuple[int, ...]

    def get_shape(self, ndim: int = 2) -> tuple[int, ...]:
        if isinstance(self.window_size, int):
            return (self.window_size,) * ndim

        if len(self.window_size) != ndim:
            raise IndexError(f'dimensions do not match the size of the window: {ndim=} {self.window_size=}')

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


def validate_window(window: Window, shape: _ShapeLike, reduce: bool, allow_even: bool = False) -> None:
    shape = np.asarray(shape)
    window_shape = np.asarray(window.get_shape(shape.size))

    if np.any(shape < window_shape):
        raise ValueError(f'Window bigger than input array: {shape=}, {window=}')

    if reduce:
        if not np.all(shape % window_shape == 0):
            raise ValueError('not all dimensions are divisible by window_shape')

    if not allow_even and not reduce:
        if np.any(window_shape % 2 == 0):
            raise ValueError('Uneven window size is not allowed when not reducing')

    if np.all(window_shape == 1):
        raise ValueError(f'Window size cannot only contain 1s {window_shape=}')

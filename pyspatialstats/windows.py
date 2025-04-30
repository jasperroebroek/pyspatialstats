"""
This module defines the definitions of the views in the sliding window methods
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel, validate_call

from pyspatialstats.types import Mask, PositiveInt, Shape


class Window(ABC):
    @abstractmethod
    def get_shape(self, ndim: PositiveInt) -> Shape:
        pass

    @abstractmethod
    def get_mask(self, ndim: PositiveInt) -> Mask:
        pass

    @property
    @abstractmethod
    def masked(self) -> bool:
        pass


class RectangularWindow(Window, BaseModel):
    window_size: PositiveInt | Shape

    def get_shape(self, ndim: PositiveInt) -> Shape:
        if isinstance(self.window_size, int):
            return (self.window_size,) * ndim

        if len(self.window_size) != ndim:
            raise IndexError(
                f"dimensions do not match the size of the window: {ndim=} {self.window_size=}"
            )

        return self.window_size

    def get_mask(self, ndim: PositiveInt) -> Mask:
        return np.ones(self.get_shape(ndim), dtype=np.bool_)

    @property
    def masked(self) -> bool:
        return False


class MaskedWindow(Window, BaseModel):
    mask: Mask

    def match_shape(self, ndim: PositiveInt) -> None:
        if self.mask.ndim != ndim:
            raise IndexError(
                f"dimensions do not match the size of the mask: {ndim=} {self.mask.ndim=}"
            )

    def get_shape(self, ndim: PositiveInt) -> Shape:
        self.match_shape(ndim)
        return self.mask.shape

    def get_mask(self, ndim: PositiveInt) -> Mask:
        self.match_shape(ndim)
        return self.mask

    @property
    def masked(self) -> bool:
        return True


def define_window(window: PositiveInt | Shape | Mask | Window) -> Window:
    if isinstance(window, Window):
        return window
    if isinstance(window, Mask):
        return MaskedWindow(mask=window)
    if isinstance(window, (int, Sequence)):
        return RectangularWindow(window_size=window)

    raise TypeError(
        f"Window can't be parsed from {window}. Must be int, sequence of int or binary array"
    )


@validate_call(config={"arbitrary_types_allowed": True})
def validate_window(
    window: Window, shape: NDArray[Any, int], reduce: bool, allow_even: bool = True
) -> None:
    shape = np.asarray(shape)
    window_shape = np.asarray(window.get_shape(shape.size))

    if np.any(shape < window_shape):
        raise ValueError(f"Window bigger than input array: {shape=}, {window=}")

    if reduce:
        if not np.array_equal(shape // window_shape, shape / window_shape):
            raise ValueError("not all dimensions are divisible by window_shape")

    if not allow_even and not reduce:
        if np.all(window_shape % 2 == 0):
            raise ValueError("Uneven window size is not allowed when not reducing")

    if np.all(window_shape == 1):
        raise ValueError(f"Window size cannot only contain 1s {window_shape=}")


def define_fringes(window: Window, reduce: bool) -> NDArray[Any, int]:
    """Only available for 2D"""
    if reduce:
        return np.array((0, 0))

    window_shape = np.asarray(window.get_shape(2))
    return window_shape // 2


def define_ind_inner(window: Window, reduce: bool) -> tuple[slice, slice]:
    """Only available for 2D"""
    if reduce:
        return np.s_[:, :]

    fringes = define_fringes(window, reduce)
    return np.s_[fringes[0] : -fringes[0], fringes[1] : -fringes[1]]

import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import DTypeLike, NDArray

from pyspatialstats.types.windows import WindowT
from pyspatialstats.utils import timeit
from pyspatialstats.views import ArrayViewPair, construct_window_views
from pyspatialstats.windows import Window, define_window


@dataclass
class MemmapContext:
    raster_shape: tuple[int, int]
    window: Window
    reduce: bool
    dtype: DTypeLike = np.float64

    def __post_init__(self):
        self.window.validate(self.reduce, shape=self.raster_shape)
        self.memmap_shape = self.window.define_windowed_shape(
            window=self.window, reduce=self.reduce, shape=self.raster_shape
        )

        self.open: bool = False
        self.memmap: Optional[np.memmap] = None

    def create(self) -> np.memmap:
        if not self.open:
            self.open = True
            self.temp_file = tempfile.NamedTemporaryFile(mode='w+')
            self.memmap = np.memmap(
                filename=self.temp_file.name,
                dtype=self.dtype,
                mode='w+',
                shape=self.memmap_shape,
            )

        return self.memmap

    def close(self):
        if not self.open:
            raise FileNotFoundError('File is not open')
        else:
            self.open = False
            self.temp_file.close()

    def __enter__(self) -> np.memmap:
        return self.create()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class OutputDict:
    def __init__(self, keys: list[str], **kwargs):
        self.keys = keys
        self.kw = kwargs
        self.contexts = {}
        self.memmaps = {}

    def __enter__(self):
        for key in self.keys:
            self.contexts[key] = MemmapContext(**self.kw)
            self.memmaps[key] = self.contexts[key].create()
        return self.memmaps

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key in self.keys:
            self.contexts[key].close()


def process_window(
    fn: Callable,
    inputs: Dict[str, NDArray],
    outputs: Dict[str, NDArray],
    views: ArrayViewPair,
    **kwargs,
) -> None:
    input_slices = views.input.slices
    output_slices = views.output.slices

    result = fn(
        **{key: inputs[key][..., input_slices[0], input_slices[1]] for key in inputs},
        **kwargs,
    )

    for key in outputs:
        outputs[key][..., output_slices[0], output_slices[1]] = result[key]


@timeit
def focal_function(
    fn: Callable,
    inputs: Dict[str, NDArray],
    outputs: Dict[str, NDArray],
    window: WindowT,
    reduce: bool = False,
    **kwargs,
) -> None:
    """Focal statistics with an arbitrary function. prefer 'threads' always works, 'processes' only works with memmaps,
    but provides potentially large speed-ups"""
    raster_shapes = []
    for key in inputs:
        s = inputs[key].shape[-2:]
        if len(s) != 2:
            raise IndexError('All inputs need to be at least 2D')
        raster_shapes.append(s)

    for s in raster_shapes:
        if not s == raster_shapes[0]:
            raise IndexError(f'Not all input rasters have the same shape: {raster_shapes}')

    window = define_window(window)
    window.validate(reduce, allow_even=False, shape=raster_shapes[0])
    window_shape = window.get_shape(2)

    for key in outputs:
        shape = outputs[key].shape[-2:]
        if reduce:
            if (
                raster_shapes[0][0] // window_shape[0],
                raster_shapes[0][1] // window_shape[1],
            ) != shape:
                raise IndexError(f'Output shapes not matching input shapes: {raster_shapes[0]} {shape}')

        elif shape != raster_shapes[0]:
            raise IndexError(f'Output shapes not matching input shapes: {raster_shapes[0]} {shape}')

    view_pairs = construct_window_views(raster_shapes[0], window_shape, reduce)

    Parallel(prefer='threads', mmap_mode='r+')(
        delayed(process_window)(fn, inputs, outputs, vp, **kwargs) for vp in view_pairs
    )

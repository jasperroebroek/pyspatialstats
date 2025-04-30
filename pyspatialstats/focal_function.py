import tempfile
from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from numpy._typing import DTypeLike
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict, validate_call

from pyspatialstats.views import RasterViewPair, construct_views
from pyspatialstats.types import Mask, PositiveInt, Shape2D
from pyspatialstats.utils import timeit
from pyspatialstats.windows import Window, define_window, validate_window


class MemmapContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    raster_shape: Shape2D
    window_shape: Shape2D
    reduce: bool
    dtype: DTypeLike = np.float64

    def __post_init__(self):
        if self.reduce:
            self.memmap_shape = (
                self.raster_shape[0] // self.window_shape[0],
                self.raster_shape[1] // self.window_shape[1],
            )
        else:
            self.memmap_shape = self.raster_shape

        self.open: bool = False
        self.memmap: Optional[np.memmap] = None

    def create(self) -> np.memmap:
        if not self.open:
            self.open = True
            self.temp_file = tempfile.NamedTemporaryFile(mode="w+")
            self.memmap = np.memmap(
                filename=self.temp_file.name,
                dtype=self.dtype,
                mode="w+",
                shape=self.memmap_shape,
            )

        return self.memmap

    def close(self):
        if not self.open:
            raise FileNotFoundError("File is not open")
        else:
            self.open = False
            self.temp_file.close()

    def __enter__(self) -> np.memmap:
        return self.create()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class OutputDict:
    def __init__(self, keys: List[str], **kwargs):
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
    views: RasterViewPair,
    **kwargs,
) -> None:
    input_slices = views.input.slices

    result = fn(
        **{key: inputs[key][..., input_slices[0], input_slices[1]] for key in inputs},
        **kwargs,
    )

    for key in outputs:
        output_slices = views.output.slices
        outputs[key][..., output_slices[0], output_slices[1]] = result[key]


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_function(
    fn: Callable,
    inputs: Dict[str, NDArray],
    outputs: Dict[str, NDArray],
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    reduce: bool = False,
    # joblib Parallel arg
    n_jobs: PositiveInt = 1,
    verbose: bool = False,
    prefer: Literal["threads", "processes"] = "threads",
    # kwargs go to fn
    **kwargs,
) -> None:
    """Focal statistics with an arbitrary function. prefer 'threads' always works, 'processes' only works with memmaps,
    but provides potentially large speed-ups"""
    raster_shapes = []
    for key in inputs:
        s = inputs[key].shape[-2:]
        if len(s) != 2:
            raise IndexError("All inputs need to be at least 2D")
        raster_shapes.append(s)

    for s in raster_shapes:
        if not s == raster_shapes[0]:
            raise IndexError(
                f"Not all input rasters have the same shape: {raster_shapes}"
            )

    window = define_window(window)
    print(str(window))
    validate_window(window, raster_shapes[0], reduce, allow_even=False)
    view_shape = window.get_shape(2)

    for key in outputs:
        shape = outputs[key].shape[-2:]
        if (
            reduce
            and (
                raster_shapes[0][0] // view_shape[0],
                raster_shapes[0][1] // view_shape[1],
            )
            != shape
            or not reduce
            and shape != raster_shapes[0]
        ):
            raise IndexError(
                f"Output shapes not matching input shapes: {raster_shapes[0]} {shape}"
            )

    view_pairs = construct_views(raster_shapes[0], view_shape, reduce)

    Parallel(n_jobs=n_jobs, verbose=verbose, prefer=prefer, mmap_mode="r+")(
        delayed(process_window)(fn, inputs, outputs, wp, **kwargs)
        for wp in view_pairs
    )

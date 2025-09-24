from typing import Callable

import numpy as np

from pyspatialstats.types.arrays import Mask
from pyspatialstats.windows import Window

WindowT = int | tuple[int, ...] | list[int] | Mask | Window

# todo; verify
# ViewFunction arguments correspond to keys in the `input` dictionary, and should return another dict with the keys
# corresponding to the `output` dictionary of the `focal_function` function.
ViewFunction = Callable[[list[np.ndarray]], list[float]]

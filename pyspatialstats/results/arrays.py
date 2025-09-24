import numpy as np

FloatResult = float | np.ndarray[tuple[int], np.float64] | np.ndarray[tuple[int, int], np.float64]
SizeTResult = np.uintp | np.ndarray[tuple[int], np.uintp] | np.ndarray[tuple[int, int], np.uintp]

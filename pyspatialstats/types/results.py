from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from pyspatialstats.types.arrays import Array

FloatResult = float | np.ndarray[tuple[int], np.float64] | np.ndarray[tuple[int, int], np.float64]
SizeTResult = np.uintp | np.ndarray[tuple[int], np.uintp] | np.ndarray[tuple[int, int], np.uintp]


@dataclass
class StatResult:
    def get_shape(self) -> Optional[tuple[int, int]]:
        for field in fields(self):
            if getattr(self, field.name) is not None:
                return getattr(self, field.name).shape
        return None


@dataclass
class CorrelationResult(StatResult):
    c: FloatResult
    df: Optional[SizeTResult] = None
    p: Optional[FloatResult] = None


@dataclass
class LinearRegressionResult(StatResult):
    df: SizeTResult
    a: FloatResult
    b: FloatResult
    se_a: FloatResult
    se_b: FloatResult
    t_a: FloatResult
    t_b: FloatResult
    p_a: Optional[FloatResult] = None
    p_b: Optional[FloatResult] = None


@dataclass
class MeanResult(StatResult):
    mean: FloatResult
    se: Optional[FloatResult] = None
    std: Optional[FloatResult] = None


FocalStatResults = Array | StatResult
GroupedStatResults = Array | StatResult

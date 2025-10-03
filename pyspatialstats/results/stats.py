from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import pandas as pd

from pyspatialstats.results.arrays import FloatResult, SizeTResult
from pyspatialstats.types.arrays import Array, VectorSizeT


@dataclass
class StatResult:
    @property
    def shape(self) -> Optional[tuple[int, int]]:
        for field in fields(self):
            if getattr(self, field.name) is not None:
                return getattr(self, field.name).shape
        return None

    @property
    def size(self) -> Optional[int]:
        for field in fields(self):
            if getattr(self, field.name) is not None:
                return getattr(self, field.name).size
        return None


@dataclass
class CorrelationResult(StatResult):
    c: FloatResult
    df: Optional[SizeTResult] = None
    p: Optional[FloatResult] = None


@dataclass
class RegressionResult(StatResult):
    """df is used to represent error values, in which case they are negative numbers. The results are not valid in this
    case."""

    df: SizeTResult
    beta: FloatResult
    beta_se: Optional[FloatResult] = None
    t: Optional[FloatResult] = None
    p: Optional[FloatResult] = None
    r_squared: Optional[FloatResult] = None
    r_squared_se: Optional[FloatResult] = None


@dataclass
class MeanResult(StatResult):
    mean: FloatResult
    se: Optional[FloatResult] = None
    std: Optional[FloatResult] = None


GroupedStatResult = Array | StatResult
FocalStatResult = Array | StatResult


@dataclass
class IndexedGroupedStatResult:
    index: VectorSizeT
    result: GroupedStatResult

    def to_dataframe(self, name: str = 'result') -> pd.DataFrame:
        if isinstance(self.result, np.ndarray):
            return pd.DataFrame(index=self.index, data={name: self.result})

        d = {}
        for field in fields(self.result):
            if getattr(self.result, field.name) is None:
                continue
            current_array = getattr(self.result, field.name)
            if current_array.ndim == 1:
                d[field.name] = current_array
            elif current_array.ndim == 2:
                for i in range(current_array.shape[1]):
                    d[f'{field.name}_{i}'] = current_array[:, i]
            else:
                raise IndexError(f"Can't parse array to dataframe with shape: {current_array.shape}")
        return pd.DataFrame(index=self.index, data=d)

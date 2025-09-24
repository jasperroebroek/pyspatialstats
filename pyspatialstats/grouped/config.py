from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

from pyspatialstats.grouped.utils import parse_data


class GroupedStatAccumulator(Protocol):
    def post_init(self, **kwargs): ...
    def add_data(
        self,
        ind: np.ndarray[tuple[int], np.uintp],
        **kwargs: np.ndarray[tuple[int], np.generic],
    ) -> None: ...
    def __add__(self, other: 'GroupedStatAccumulator') -> 'GroupedStatAccumulator': ...


@dataclass
class GroupedResultConfig:
    accumulator_type: type[GroupedStatAccumulator]
    parse_data_fun: Callable = parse_data
    name: str = 'r'
    to_result_func: str = 'to_result'
    to_filtered_result_func: str = 'to_filtered_result'
    kwargs: Optional[dict] = None

    def get_accumulator(self) -> GroupedStatAccumulator:
        acc = self.accumulator_type()
        if self.kwargs is not None:
            acc.post_init(**self.kwargs)
        return acc

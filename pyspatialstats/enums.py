from enum import IntEnum
from typing import Literal


class MajorityMode(IntEnum):
    ASCENDING = 0
    DESCENDING = 1
    NAN = 2


ErrorType = Literal['parametric', 'bootstrap']

from enum import IntEnum, StrEnum


class MajorityMode(IntEnum):
    ASCENDING = 0
    DESCENDING = 1
    NAN = 2


class Uncertainty(StrEnum):
    STD = 'std'
    SE = 'se'

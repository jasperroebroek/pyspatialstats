from .base import FocalResultConfig
from .array import FocalArrayResultConfig
from .focal_stat_results import FocalCorrelationResultConfig, FocalLinearRegressionResultConfig, FocalMeanResultConfig

__all__ = [
    'FocalArrayResultConfig',
    'FocalCorrelationResultConfig',
    'FocalLinearRegressionResultConfig',
    'FocalMeanResultConfig',
    'FocalResultConfig',
]

"""
Module providing function that calculate statistics on groups.
"""

from pyspatialstats.grouped.stats import (
    grouped_correlation,
    grouped_count,
    grouped_linear_regression,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_std,
    grouped_sum,
)
from pyspatialstats.grouped.utils import define_max_ind

__all__ = [
    'define_max_ind',
    'grouped_count',
    'grouped_min',
    'grouped_max',
    'grouped_mean',
    'grouped_std',
    'grouped_sum',
    'grouped_correlation',
    'grouped_linear_regression',
]

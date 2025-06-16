"""
Module providing function that calculate statistics on groups. All function allow for the parameters ``ind`` and ``v``
/``v1`` and ``v2`` in case of multiple variables needed to calculate the statistics such as linear regression and
correlation.

* ind
    Contains group index. Will be parsed to unsigned integers.
* v*
    Variable(s). If one, 'v', otherwise v1, v2 etc. They will be parsed to floats.

"""
from pyspatialstats.grouped.stats import (
    grouped_correlation,
    grouped_correlation_pd,
    grouped_count,
    grouped_count_pd,
    grouped_linear_regression,
    grouped_linear_regression_pd,
    grouped_max,
    grouped_max_pd,
    grouped_mean,
    grouped_mean_pd,
    grouped_mean_std_pd,
    grouped_min,
    grouped_min_pd,
    grouped_std,
    grouped_std_pd,
)
from pyspatialstats.grouped.utils import define_max_ind, generate_index

__all__ = [
    "define_max_ind",
    "generate_index",
    "grouped_count",
    "grouped_count_pd",
    "grouped_min",
    "grouped_min_pd",
    "grouped_max",
    "grouped_max_pd",
    "grouped_mean",
    "grouped_mean_pd",
    "grouped_std",
    "grouped_std_pd",
    "grouped_mean_std_pd",
    "grouped_correlation",
    "grouped_correlation_pd",
    "grouped_linear_regression",
    "grouped_linear_regression_pd",
]

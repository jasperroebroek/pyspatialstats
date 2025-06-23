.. currentmodule:: focal_stats_base

#############
API reference
#############

Focal statistics
================

This module provides focal statistics functionality, similar to the methods in the ArcGIS software. The various functions in this module accept a 2D array as input data. The sliding window characteristics (dimensions and masking) are provided through the ``window`` keyword. This accepts either an integer, a boolean mask, or a :class:`focal_stats_base.window.Window` object. The functions return an array, either of the same dimensions as the input data, or an array of smaller dimensions if the ``reduce`` parameter is used. This allows for a non-overlapping sliding window. :func:`focal_correlation` calculates the correlation between two arrays in contrast to the other functions that operate on a single array.

.. autosummary::
    :toctree: generated/focal_statistics

    focal_sum
    focal_min
    focal_max
    focal_mean
    focal_std
    focal_majority
    focal_correlation


Grouped statistics
==================

.. currentmodule:: grouped_stats

This module provides functions that calculate statistics based on strata, allowing for data of any dimensionality. To use these functions, you must provide an array ``ind`` with the same shape as the data, which represent the strata index.

For each statistic, two functions are available:

#. Array-based output: Returns a NumPy array where each index corresponds to the stratum.
#. DataFrame-based output: Returns a Pandas DataFrame where the index only includes the actual strata, and the calculated statistic values are stored in the columns.

.. autosummary::
    :toctree: generated/grouped_stats

    grouped_count
    grouped_count_pd
    grouped_min
    grouped_min_pd
    grouped_max
    grouped_max_pd
    grouped_mean
    grouped_mean_pd
    grouped_std
    grouped_std_pd
    grouped_mean_std_pd
    grouped_correlation
    grouped_correlation_pd
    grouped_linear_regression
    grouped_linear_regression_pd


Strata statistics
=================

.. currentmodule:: strata_stats

This module implements functions that calculates statistics for each stratum and reapplies it to the input raster. This depends on the grouped statistics module. It is only available on 2D data.

.. autosummary::
    :toctree: generated/strata_stats

    strata_count
    strata_min
    strata_max
    strata_mean
    strata_std
    strata_mean_std
    strata_correlation
    strata_linear_regression


Rolling functions
=================

.. currentmodule:: rolling

This module provides rolling functions that can process ND arrays. These functions are using the same sliding window approach as the focal statistics through the ``window`` parameter. However, they do not specifically account for NaN values, matching the default behavior of NumPy. Designed for flexibility, these functions are meant to construct custom focal statistics methods in any dimensionality. These methods are similar to the :func:`numpy.lib.stride_tricks.sliding_window_view` function.

.. autosummary::
    :toctree: generated/rolling

    rolling_window
    rolling_sum
    rolling_mean


Windows
=======

.. currentmodule:: window

The sliding window methods as described above are implemented in this module, through the :class:`focal_stats_base.window.Window` class. Two concrete implementations are provided: RectangularWindow and MaskedWindow. Custom implementations can be provided by subclassing the :class:`focal_stats_base.window.Window` class, implementing the ``get_shape`` and ``get_mask`` methods and the ``masked`` property.


.. autosummary::
    :toctree: generated/window

    RectangularWindow
    MaskedWindow

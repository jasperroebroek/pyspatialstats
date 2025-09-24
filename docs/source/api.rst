.. currentmodule:: focal

#############
API reference
#############

Focal statistics
================

This module provides focal statistics functionality, similar to the methods in the ArcGIS software. The various functions in this module accept a 2D array as input data. The sliding window characteristics (dimensions and masking) are provided through the ``window`` keyword. This accepts either an integer, a boolean mask, or a :class:`pyspatialstats.windows.Window` object. The functions return an array, either of the same dimensions as the input data, or an array of smaller dimensions if the ``reduce`` parameter is used. This allows for a non-overlapping sliding window. :func:`focal_correlation` calculates the correlation between two arrays in contrast to the other functions that operate on a single array.

.. autosummary::
    :toctree: generated/focal

    focal_sum
    focal_min
    focal_max
    focal_mean
    focal_std
    focal_majority
    focal_correlation


Grouped statistics
==================

.. currentmodule:: grouped

This module provides functions that calculate statistics based on group indices, allowing for data of any dimensionality. To use these functions, you must provide an array ``ind`` with the same shape as the data, which represent the index.

.. autosummary::
    :toctree: generated/grouped_stats

    grouped_count
    grouped_min
    grouped_max
    grouped_mean
    grouped_std
    grouped_correlation
    grouped_linear_regression


Zonal statistics
=================

.. currentmodule:: zonal

This module implements functions that calculates statistics for each group index and reapplies it to the input raster. This depends on the grouped statistics module. It is only available on 2D data.

.. autosummary::
    :toctree: generated/zonal

    zonal_count
    zonal_min
    zonal_max
    zonal_mean
    zonal_std
    zonal_correlation
    zonal_linear_regression


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

.. currentmodule:: windows

The sliding window methods as described above are implemented in this module, through the :class:`pyspatialstats.windows.Window` class. Two concrete implementations are provided: `RectangularWindow` and `MaskedWindow`. Custom implementations can be provided by subclassing the :class:`pyspatialstats.windows.Window` class, implementing the ``get_shape`` and ``get_mask`` methods and the ``masked`` property.


.. autosummary::
    :toctree: generated/windows

    RectangularWindow
    MaskedWindow

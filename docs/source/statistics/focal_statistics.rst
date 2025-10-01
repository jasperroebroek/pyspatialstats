.. _statistics/focal_statistics:

.. currentmodule:: focal

Focal statistics
================

Focal statistics are statistical operations applied to neighborhoods of data points, often referred to as sliding or rolling window operations. The operation is applied to each pixel of a raster, iterating over all four directions to calculate the focal statistic.

The implementations in this module share many parameters, defining how the windows are defined, how the output is structured and how the calculations are performed:

* ``window``: The window characteristics (dimensions and masking) are provided through the ``window`` keyword. This accepts either an integer, a boolean mask, or a :class:`Window` object.
* ``reduce``: If set to True, the output array will have smaller dimensions than the input array, with a non-overlapping sliding window. False will yield an output array with the same dimensions as the input array, with an overlapping sliding window.
* ``out``: An array (numpy/xarray/zarr etc) or StatResult object, depending on the return type of the function, to store the results in. If not provided, new arrays will be created.
* ``chunks``: Chunk size to split the calculation over. If provided, parallel may be used through a joblib context manager. See :ref:`here  <methods/parallel_processing>`, for more details.

Currently implemented methods are:

* :func:`focal_sum`
* :func:`focal_min`
* :func:`focal_max`
* :func:`focal_mean`
* :func:`focal_std`
* :func:`focal_majority`
* :func:`focal_correlation`
* :func:`focal_linear_regression`

Examples
--------

An example from the ArcGIS documentation, considering the focal sum of a raster with a specified neighborhood of 3x3 pixels. The values in this window are summed and placed in the output array at the location of the most central pixel in the window:

.. image:: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/GUID-8FD3FAA9-99E0-41E9-A4F3-0B410168F442-web.png
    :alt: focal sum example

And for the complete raster:

.. image:: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/GUID-CB626440-C076-4B04-B8A9-D589B0648E7D-web.png
    :alt: focal sum example full

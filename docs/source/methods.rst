.. currentmodule:: focal

Focal statistics
================

Focal statistics are functions calculated on the neighborhood of the data, often referred to as sliding, or rolling window operations on 2D data. Implemented functions range from mean and standard deviation to quantiles and mode (majority).

Conceptually, the algorithm iterates over each pixel of a raster, and looks in all four directions to calculate the focal statistic. The neighborhoods can overlap, but might also be stacked next to each other, so a sparse output is generated.

An example from the ArcGIS documentation, considering the focal sum of a raster with a specified neighborhood of 3x3 pixels. The values in this window are summed and placed in the output array at the location of the most central pixel in the window:

.. image:: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/GUID-8FD3FAA9-99E0-41E9-A4F3-0B410168F442-web.png
    :alt: focal sum example

The second example shows what the same example looks like when a complete raster is considered:

.. image:: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/GUID-CB626440-C076-4B04-B8A9-D589B0648E7D-web.png
    :alt: focal sum example

Grouped/zonal statistics
=========================

Grouped statistics are statistics calculated on groups of pixels defined by an index (:mod:`pyspatialstats.grouped`). In this package they are defined in any dimension. The calculated values can be reapplied on an output with the same dimensions as the input. For 2D raster data this is available in the :mod:`pyspatialstats.zonal` module.

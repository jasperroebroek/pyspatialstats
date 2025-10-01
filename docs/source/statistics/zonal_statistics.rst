.. _statistics/zonal_statistics:

.. currentmodule:: zonal

Zonal statistics
================

Zonal statistics are statistics calculated on groups of pixels defined by an index (:mod:`pyspatialstats.zonal`), and reapplied to the structure of the index data. The implementation of this module depends on the :mod:`pyspatialstats.grouped` module, sharing the same parameters. These methods can be used to for example to create a map of the mean of a variable per country, with each pixel of the country displaying the same value.

Currently implemented methods are:

* :func:`zonal_min`
* :func:`zonal_max`
* :func:`zonal_mean`
* :func:`zonal_std`
* :func:`zonal_correlation`
* :func:`zonal_linear_regression`

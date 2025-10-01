.. focal_stats_base documentation master file, created by
   sphinx-quickstart on Wed Dec  1 14:19:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*****************************
Spatial statistics for python
*****************************

This module aims to provide spatial statistics for python, that runs without the installation of extensive GIS packages. For more details see the documentation.

The package implements three different categories of spatial statistics:

* :ref:`Focal statistics <statistics/focal_statistics>` (:mod:`pyspatialstats.focal`), which are calculated as a moving window over input rasters (2D).
* :ref:`Grouped statistics <statistics/grouped_statistics>` (:mod:`pyspatialstats.grouped`), which calculates the statistics based on group indices (xD)
* :ref:`Zonal statistics <statistics/zonal_statistics>` (:mod:`pyspatialstats.zonal`), which calculates the statistics for each group index and reapplies it to the index. This depends on the grouped statistics module.


*************
Documentation
*************

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Statistics

   statistics/focal_statistics
   statistics/grouped_statistics
   statistics/zonal_statistics

.. toctree::
   :maxdepth: 2
   :caption: Methods

   methods/windows
   methods/rolling_window
   methods/parallel_processing

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/focal_stats
   notebooks/custom_focal_stats

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   whats_new
   api


License
=======

pyspatialstats is published under a MIT license.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

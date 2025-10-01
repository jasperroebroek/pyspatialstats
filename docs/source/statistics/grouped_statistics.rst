.. _statistics/grouped_statistics:

.. currentmodule:: grouped

Grouped statistics
=========================

Grouped statistics are statistics calculated on groups of pixels defined by a common index (:mod:`pyspatialstats.grouped`). The functions in this module share many parameters, defining how the calculations are performed:

* ``ind``: The index array, with the same shape as the data, representing the group index.
* ``filtered``: If to filter out groups without observations. If True, a pandas DataFrame is returned, otherwise a numpy array, where the place of the value corresponds to the index of the group.
* ``chunks``: Chunk size to split the calculation over. If provided, parallel may be used through a joblib context manager. See :ref:`here  <methods/parallel_processing>`, for more details.

Currently implemented methods are:

* :func:`grouped_min`
* :func:`grouped_max`
* :func:`grouped_mean`
* :func:`grouped_std`
* :func:`grouped_correlation`
* :func:`grouped_linear_regression`

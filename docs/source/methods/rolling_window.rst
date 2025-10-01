.. _methods/rolling_window:

Rolling window
==============

.. currentmodule:: rolling

This module provides rolling functions that can process ND arrays. These functions are using the same sliding window approach as the focal statistics through the ``window`` parameter. However, they do not specifically account for NaN values, matching the default behavior of NumPy. Designed for flexibility, these functions are meant to construct custom focal statistics methods in any dimensionality. These methods are similar to the :func:`numpy.lib.stride_tricks.sliding_window_view` function.

Currently implemented methods are:

* :func:`rolling_window`
* :func:`rolling_sum`
* :func:`rolling_mean`

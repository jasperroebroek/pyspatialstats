Windows
=======

.. currentmodule:: windows

The :mod:`pyspatialstats.windows` module implements different window types
for the sliding window approach used in the :mod:`pyspatialstats.focal` and
:mod:`pyspatialstats.rolling` modules.

Two concrete implementations are provided: `RectangularWindow` and
`MaskedWindow`. Custom implementations can be provided by subclassing the
:class:`Window` class, implementing the
``get_shape`` and ``get_mask`` methods and the ``masked`` properties.

.. code:: python

    from pyspatialstats import RectangularWindow

    window = RectangularWindow((3, 3))
    window.get_shape()

.. parsed-literal::

    (3, 3)


RectangularWindow
-----------------

The :class:`RectangularWindow` class is a concrete implementation of the
:class:`Window` class. It's shape is defined by either an integer,
which yields a square window in any dimension, or a tuple of integers, representing
the size of the window in each dimension.


MaskedWindow
------------

The :class:`MaskedWindow` class defines a window based on a boolean mask, allowing
any shape of the window to be defined.

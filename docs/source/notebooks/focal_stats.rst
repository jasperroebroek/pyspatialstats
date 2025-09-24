None

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../../source/notebooks/focal_stats.ipynb>`_.

.. _focal_stats:

Usage example
=============

.. code:: python

    import pyspatialstats.focal as fs
    import rasterio as rio
    import matplotlib.pyplot as plt
    import numpy as np
    import os

.. code:: python

    os.chdir('../../../')

Loading raster (containing water table depth (Fan et al., 2017)).

.. code:: python

    with rio.open('data/wtd.tif') as f:
        a = f.read(1).astype(np.float64)
        a[a == -999.9] = np.nan

Inspecting the data

.. code:: python

    plt.imshow(a, cmap='Blues', vmax=100)
    plt.title('Water table depth')
    plt.colorbar()




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x153e75010>




.. image:: focal_stats_files/focal_stats_6_1.png


Focal statistics
----------------

Calculation of the focal mean:

.. code:: python

    plt.imshow(fs.focal_mean(a, window=15).mean, vmax=100, cmap='Blues')




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x153fee850>




.. image:: focal_stats_files/focal_stats_9_1.png


This looks quite similar to the input raster, but with smoothing
applied. Let’s try a higher window, which should increase the smoothing

.. code:: python

    plt.imshow(fs.focal_mean(a, window=25).mean, vmax=100, cmap='Blues')




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x15407a210>




.. image:: focal_stats_files/focal_stats_11_1.png


This same functionality can be used to reduce the shape of the raster
based on this window.

.. code:: python

    x = fs.focal_mean(a, window=108, reduce=True).mean
    plt.imshow(x, vmax=100, cmap='Blues')




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x1541056d0>




.. image:: focal_stats_files/focal_stats_13_1.png


The shape of this new raster is exactly 108 times smaller than the input
raster. Note that for this to work both x and y-axes need to be
divisible by the window size.



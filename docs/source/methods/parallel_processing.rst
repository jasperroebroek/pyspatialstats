.. _methods/parallel_processing:

Parallel processing
===================

Parallel processing can be used to speed up the calculation of statistics by splitting the calculation over multiple chunks. Chunking is possible through the ``chunks`` parameter in the focal, grouped and zonal statistics functions. The parallel processing is implemented using joblib, and can be used by providing a joblib context manager to the function.

For example, to calculate the focal mean of a raster with 1000x1000 pixels, using 5 threads and chunking the calculation over chunks of 100 pixels, the following code can be used:

.. code:: python

    a = np.random.rand(1000, 1000)
    with parallel_config(backend='threading', n_jobs=5):
        focal_mean(a, window=7, chunks=100)

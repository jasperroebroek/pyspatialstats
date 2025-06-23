# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport isnan
from libc.stdlib cimport malloc, free
from pyspatialstats.bootstrap.mean cimport _bootstrap_mean, CyBootstrapMeanResult
from pyspatialstats.random.random cimport RandomInts
from pyspatialstats.types.cy_types cimport numeric


cpdef void _focal_mean(
    numeric[:, :, :, :] a,
    np.npy_uint8[:, ::1] mask,
    double[:, :] r,
    int[:] fringe,
    double threshold,
    bint reduce
):
    cdef:
        size_t i, j, p, q, count_values
        numeric[:, :] window
        double a_sum

    with nogil:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                window = a[i, j]

                if not reduce and isnan(window[fringe[0], fringe[1]]):
                    continue

                a_sum = 0
                count_values = 0

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if not isnan(window[p, q]) and mask[p, q]:
                            a_sum += window[p, q]
                            count_values += 1

                if count_values < threshold:
                    continue

                r[i, j] = a_sum / count_values


cpdef void _focal_mean_bootstrap(
    numeric[:, :, :, :] a,
    np.npy_uint8[:, ::1] mask,
    double[:, :] mean,
    double[:, :] se,
    int[:] fringe,
    double threshold,
    bint reduce,
    size_t n_bootstraps,
    int seed
):
    cdef:
        size_t i, j, p, q, count_values
        numeric[:, :] window
        double a_sum
        double* window_values
        double* means
        CyBootstrapMeanResult r
        RandomInts rng

    window_values = <double *> malloc(mask.shape[0] * mask.shape[1] * sizeof(double))
    if window_values == NULL:
        raise MemoryError("'values' memory allocation failed")

    means = <double *> malloc(n_bootstraps * sizeof(double))
    if means == NULL:
        free(window_values)
        raise MemoryError("Memory allocation failed")

    rng = RandomInts(seed if seed != 0 else None)

    with nogil:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                window = a[i, j]

                if not reduce and isnan(window[fringe[0], fringe[1]]):
                    continue

                count_values = 0

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if not isnan(window[p, q]) and mask[p, q]:
                            window_values[count_values] = window[p, q]
                            count_values += 1

                if count_values == 0 or count_values < threshold:
                    continue

                r = _bootstrap_mean(
                    v=window_values,
                    n_samples=count_values,
                    n_bootstraps=n_bootstraps,
                    rng=rng,
                    means=means
                )

                mean[i, j] = r.mean
                se[i, j] = r.se

    free(window_values)
    free(means)

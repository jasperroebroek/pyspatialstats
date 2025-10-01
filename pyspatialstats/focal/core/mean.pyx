# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport isnan
from libc.stdlib cimport malloc, free
from pyspatialstats.stats.welford cimport WelfordState, ws_mean, ws_std, ws_add, ws_reset
from pyspatialstats.bootstrap.mean cimport bootstrap_mean
from pyspatialstats.random.random cimport Random


cpdef void _focal_mean(
    double[:, :, :, :] a,
    np.npy_uint8[:, ::1] mask,
    double[:, :] mean,
    int[:] fringe,
    double threshold,
    bint reduce
):
    cdef:
        size_t i, j, p, q, count_values
        double[:, :] window
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

                mean[i, j] = a_sum / count_values


cpdef void _focal_mean_std(
    double[:, :, :, :] a,
    np.npy_uint8[:, ::1] mask,
    double[:, :] mean,
    double[: ,:] std,
    int[:] fringe,
    double threshold,
    bint reduce
):
    cdef:
        size_t i, j, p, q, count_values
        double[:, :] window
        WelfordState ws

    with nogil:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                window = a[i, j]

                if not reduce and isnan(window[fringe[0], fringe[1]]):
                    continue

                ws_reset(&ws)

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if not isnan(window[p, q]) and mask[p, q]:
                            ws_add(&ws, window[p, q])

                if ws.count < threshold:
                    continue

                mean[i, j] = ws_mean(&ws)
                std[i, j] = ws_std(&ws, 0)


cpdef void _focal_mean_bootstrap(
    double[:, :, :, :] a,
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
        double[:, :] window
        double a_sum
        double* window_values
        WelfordState result
        Random rng

    window_values = <double *> malloc(mask.shape[0] * mask.shape[1] * sizeof(double))
    if window_values == NULL:
        raise MemoryError("'window_values' memory allocation failed")

    rng = Random(seed)

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

                bootstrap_mean(
                    v=window_values,
                    n_samples=count_values,
                    n_boot=n_bootstraps,
                    rng=rng,
                    result=&result
                )

                mean[i, j] = ws_mean(&result)
                se[i, j] = ws_std(&result)

    free(window_values)

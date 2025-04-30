# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

from libc.math cimport isnan
from libc.stdlib cimport calloc, free, malloc
from numpy.math cimport NAN
cimport numpy as cnp
from pyspatialstats.grouped_stats.core.count cimport _define_max_ind, _grouped_count


cdef float* _grouped_mean(size_t[:] ind, float[:] v, size_t max_ind) nogil:
    cdef:
        size_t i, k, n = ind.shape[0]
        long *count_v = <long *> calloc(max_ind + 1, sizeof(long))
        float *sum_v = <float *> calloc(max_ind + 1, sizeof(float))
        float *mean_v = sum_v

    if count_v == NULL or sum_v == NULL:
        free(count_v)
        free(sum_v)
        with gil:
            raise MemoryError("count_v or sum_v memory error")

    for i in range(n):
        if ind[i] == 0:
            continue
        if isnan(v[i]):
            continue
        sum_v[ind[i]] += v[i]
        count_v[ind[i]] += 1

    for k in range(max_ind + 1):
        if count_v[k] == 0:
            mean_v[k] = NAN
        else:
            mean_v[k] /= count_v[k]

    free(count_v)

    return mean_v


def grouped_mean_npy(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        size_t max_ind
        float *r

    with nogil:
        max_ind = _define_max_ind(ind)
        r = _grouped_mean(ind, v, max_ind)

    result_array = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT,  r)
    cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    return result_array


def grouped_mean_npy_filtered(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        size_t i, max_ind, c = 0, num_inds = 0
        long *count_v
        float *r_v
        float *rf_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r_v = _grouped_mean(ind, v, max_ind)
            count_v = _grouped_count(ind, v, max_ind)

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    num_inds += 1

            rf_v = <float *> malloc(num_inds * sizeof(float))

            if rf_v == NULL:
                with gil:
                    raise MemoryError("rf_v memory error")

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    rf_v[c] = r_v[i]
                    c += 1

        result_array = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_FLOAT, rf_v)
        cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    finally:
        free(r_v)
        free(count_v)

    return result_array

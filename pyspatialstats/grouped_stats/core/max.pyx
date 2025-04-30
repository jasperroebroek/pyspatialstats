# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

from libc.math cimport isnan
from libc.stdlib cimport malloc, free
from numpy.math cimport INFINITY, NAN
cimport numpy as cnp
from pyspatialstats.grouped_stats.core.count cimport _define_max_ind, _grouped_count


cdef float* _grouped_max(size_t[:] ind, float[:] v, size_t max_ind) nogil:
    cdef:
        size_t i, k, n = ind.shape[0]
        float *max_v = <float *> malloc((max_ind + 1) * sizeof(float))

    if max_v == NULL:
        with gil:
            raise MemoryError("max_v memory error")

    for k in range(max_ind + 1):
        max_v[k] = -INFINITY

    for i in range(n):
        if ind[i] == 0:
            continue
        if isnan(v[i]):
            continue
        if v[i] > max_v[ind[i]]:
            max_v[ind[i]] = v[i]

    for k in range(max_ind + 1):
        if max_v[k] == -INFINITY:
            max_v[k] = NAN

    return max_v


def grouped_max_npy(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        size_t max_ind
        float *r

    with nogil:
        max_ind = _define_max_ind(ind)
        r = _grouped_max(ind, v, max_ind)

    result_array = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT, r)
    cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    return result_array


def grouped_max_npy_filtered(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        size_t i, max_ind, c = 0, num_inds = 0
        long *count_v
        float *r_v
        float *rf_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r_v = _grouped_max(ind, v, max_ind)
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

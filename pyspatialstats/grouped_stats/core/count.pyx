# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport isnan
from libc.stdlib cimport calloc, free
cimport numpy as cnp
import numpy as np


cdef size_t _define_max_ind(size_t[:] ind) noexcept nogil:
    cdef:
        size_t i, max_ind = 0, n = ind.shape[0]

    for i in range(n):
        if ind[i] > max_ind:
            max_ind = ind[i]

    return max_ind


cdef long* _grouped_count(size_t[:] ind, float[:] v, size_t max_ind) nogil:
    cdef:
        size_t i, n = ind.shape[0]
        long *count_v = <long *> calloc(max_ind + 1, sizeof(long))

    if count_v == NULL:
        with gil:
            raise MemoryError("count_v memory error")

    for i in range(n):
        if ind[i] == 0:
            continue
        if isnan(v[i]):
            continue
        count_v[ind[i]] += 1

    return count_v


def define_max_ind(size_t[:] ind):
    cdef size_t max_ind

    with nogil:
        max_ind = _define_max_ind(ind)

    return max_ind


def grouped_count_npy(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        size_t max_ind
        long* r

    with nogil:
        max_ind = _define_max_ind(ind)
        r = _grouped_count(ind, v, max_ind)

    result_array = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_LONG, <void *> r)
    cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    return result_array


def grouped_count_npy_filtered(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        size_t i, max_ind, c = 0, num_inds = 0
        long *r, *rf

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r = _grouped_count(ind, v, max_ind)

            rf = <long *> calloc(max_ind + 1, sizeof(long))

            for i in range(max_ind + 1):
                if r[i] > 0:
                    num_inds += 1
                    rf[c] = r[i]
                    c += 1

        result_array = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_LONG, rf)
        cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    finally:
        free(r)

    return result_array

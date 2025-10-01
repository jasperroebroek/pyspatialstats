# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

cimport numpy as np
from libc.math cimport isnan, NAN
from libc.stdlib cimport malloc, free


cpdef void _focal_majority(
    double[:, :, :, :] a,
    np.npy_uint8[:, ::1] mask,
    double[:, :] r,
    int[:] fringe,
    double threshold,
    bint reduce,
    int mode
):
    cdef:
        size_t i, j, p, q, c, v, count_values, curr_max_count, num_values
        double[:, :] window
        double curr_value
        bint in_store, is_double
        size_t[:] counts
        double* values

    num_values = mask.shape[0] * mask.shape[1]

    values = <double*> malloc(num_values * sizeof(double))
    if values == NULL:
        raise MemoryError

    counts = np.zeros(num_values, dtype=np.uintp)

    with nogil:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                window = a[i, j]

                if not reduce and isnan(window[fringe[0], fringe[1]]):
                    continue

                values[0] = 0
                counts[0] = 0
                count_values = 0
                c = 1

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if not isnan(window[p, q]) and mask[p, q]:
                            if count_values == 0:
                                values[0] = window[p, q]

                            in_store = False
                            for v in range(c):
                                if window[p, q] == values[v]:
                                    counts[v] += 1
                                    in_store = True
                                    break

                            if not in_store:
                                values[c] = window[p, q]
                                counts[c] = 1
                                c += 1

                            count_values = count_values + 1

                if count_values < threshold:
                    continue

                if mode == 0: # ascending
                    curr_max_count = 0
                    curr_value = NAN
                    for v in range(c):
                        if counts[v] > curr_max_count:
                            curr_max_count = counts[v]
                            curr_value = values[v]

                if mode == 1: # descending
                    curr_max_count = 0
                    curr_value = NAN
                    for v in range(c):
                        if counts[v] >= curr_max_count:
                            curr_max_count = counts[v]
                            curr_value = values[v]

                if mode == 2: # nan
                    curr_max_count = 0
                    curr_value = 0
                    is_double = False
                    for v in range(c):
                        if counts[v] == curr_max_count:
                            is_double = True
                        elif counts[v] > curr_max_count:
                            curr_max_count = counts[v]
                            curr_value = values[v]
                            is_double = False

                    if is_double:
                        curr_value = NAN

                r[i, j] = curr_value

    free(values)
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport isnan


cpdef void _focal_sum(
    double[:, :, :, :] a,
    np.npy_uint8[:, ::1] mask,
    double[:, :] r,
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

                r[i, j] = a_sum

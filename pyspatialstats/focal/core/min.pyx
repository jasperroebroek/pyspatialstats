# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
from libc.math cimport isnan
from pyspatialstats.types.cy_types cimport numeric


cpdef void _focal_min(
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
        double curr_min

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
                            if window[p, q] < curr_min or count_values == 0:
                                curr_min = window[p, q]
                            count_values = count_values + 1

                if count_values < threshold:
                    continue

                r[i, j] = curr_min

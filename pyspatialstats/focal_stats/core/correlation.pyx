# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Algorithm to correlate two arrays (2D) with each other
"""
import numpy as np

from pyspatialstats.focal_stats.core.iteration_params cimport define_iter_params, IterParams

cimport numpy as np
from libc.stdlib cimport free
from libc.math cimport isnan, sqrt


cpdef double[:, ::1] _focal_correlation(double[:, ::1] a,
                                        double[:, ::1] b,
                                        int[:] window_shape,
                                        np.npy_uint8[:, ::1] mask,
                                        double fraction_accepted,
                                        bint reduce,
                                        ):
    cdef:
        size_t p, q, i, j, x, y
        double[:, ::1] corr
        double r_num, d1_mean, d2_mean, d1_sum, d2_sum, c1_dist, c2_dist, r_den_d1, r_den_d2
        double num_values, threshold, count_values, first_value1, first_value2
        bint all_equal_d1, all_equal_d2
        size_t shape[2]
        size_t ws[2]
        IterParams *ip

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_shape[0]
    ws[1] = window_shape[1]

    ip = define_iter_params(shape, ws, fraction_accepted, reduce)
    corr = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]) or isnan(b[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                d1_sum = 0
                d2_sum = 0
                count_values = 0
                all_equal_d1 = True
                all_equal_d2 = True

                for p in range(window_shape[0]):
                    for q in range(window_shape[1]):
                        if not isnan(a[i + p, j + q]) and not isnan(b[i + p, j + q]) and mask[p, q]:
                            if count_values == 0:
                                first_value1 = a[i + p, j + q]
                                first_value2 = b[i + p, j + q]
                            d1_sum = d1_sum + a[i + p, j + q]
                            d2_sum = d2_sum + b[i + p, j + q]

                            if a[i + p, j + q] != first_value1:
                                all_equal_d1 = False
                            if b[i + p, j + q] != first_value2:
                                all_equal_d2 = False

                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                elif all_equal_d1 or all_equal_d2:
                    corr[y + ip.fringe[0], x + ip.fringe[1]] = 0

                else:
                    d1_mean = d1_sum / count_values
                    d2_mean = d2_sum / count_values

                    r_num = 0
                    r_den_d1 = 0
                    r_den_d2 = 0

                    for p in range(window_shape[0]):
                        for q in range(window_shape[1]):
                            if not isnan(a[i + p, j + q]) and not isnan(b[i + p, j + q]) and mask[p, q]:
                                c1_dist = a[i + p, j + q] - d1_mean
                                c2_dist = b[i + p, j + q] - d2_mean

                                r_num = r_num + (c1_dist * c2_dist)
                                r_den_d1 = r_den_d1 + c1_dist ** 2
                                r_den_d2 = r_den_d2 + c2_dist ** 2

                    corr[y + ip.fringe[0], x + ip.fringe[1]] = r_num / sqrt(r_den_d1 * r_den_d2)

    free(ip)
    return corr

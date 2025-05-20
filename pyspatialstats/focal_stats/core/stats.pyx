# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

from pyspatialstats.focal_stats.core.iteration cimport define_iter_params, IterParams

cimport numpy as np
from libc.math cimport isnan, sqrt, NAN
from libc.stdlib cimport free


cpdef double[:, ::1] _focal_std(double[:, ::1] a,
                                int[:] window_shape,
                                np.npy_uint8[:, ::1] mask,
                                double fraction_accepted,
                                bint reduce,
                                int dof
                                ):
    cdef:
        size_t p, q, i, j, x, y, count_values
        double[:, ::1] r
        double a_sum, a_mean, x_sum
        double first_value1, first_value2
        bint all_equal_d1, all_equal_d2
        size_t shape[2]
        size_t ws[2]
        IterParams *ip

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_shape[0]
    ws[1] = window_shape[1]

    ip = define_iter_params(shape, ws, fraction_accepted, reduce)
    r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                a_sum = 0
                count_values = 0

                for p in range(window_shape[0]):
                    for q in range(window_shape[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            a_sum = a_sum + a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    a_mean = a_sum / count_values
                    x_sum = 0

                    for p in range(window_shape[0]):
                        for q in range(window_shape[1]):
                            if not isnan(a[i + p, j + q]) and mask[p, q]:
                                x_sum = x_sum + (a[i + p, j + q] - a_mean) ** 2

                    r[y + ip.fringe[0], x + ip.fringe[1]] = sqrt(x_sum / (count_values - dof))

    free(ip)
    return r


cpdef double[:, ::1] _focal_sum(double[:, ::1] a,
                                int[:] window_shape,
                                np.npy_uint8[:, ::1] mask,
                                double fraction_accepted,
                                bint reduce,
                                ):
    cdef:
        size_t p, q, i, j, x, y, count_values
        double[:, ::1] r
        double a_sum
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_shape[0]
    ws[1] = window_shape[1]

    ip = define_iter_params(shape, ws, fraction_accepted, reduce)
    r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                a_sum = 0
                count_values = 0

                for p in range(window_shape[0]):
                    for q in range(window_shape[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            a_sum = a_sum + a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    r[y + ip.fringe[0], x + ip.fringe[1]] = a_sum

    free(ip)
    return r


cpdef double[:, ::1] _focal_min(double[:, ::1] a,
                                int[:] window_shape,
                                np.npy_uint8[:, ::1] mask,
                                double fraction_accepted,
                                bint reduce,
                                ):
    cdef:
        size_t p, q, i, j, x, y, count_values
        double[:, ::1] r
        double curr_min
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_shape[0]
    ws[1] = window_shape[1]

    ip = define_iter_params(shape, ws, fraction_accepted, reduce)
    r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                curr_min = 0
                count_values = 0

                for p in range(window_shape[0]):
                    for q in range(window_shape[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            if a[i + p, j + q] < curr_min or count_values == 0:
                                curr_min = a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    r[y + ip.fringe[0], x + ip.fringe[1]] = curr_min

    free(ip)
    return r


cpdef double[:, ::1] _focal_max(double[:, ::1] a,
                                int[:] window_shape,
                                np.npy_uint8[:, ::1] mask,
                                double fraction_accepted,
                                bint reduce,
                                ):
    cdef:
        size_t p, q, i, j, x, y, count_values
        double[:, ::1] r
        double curr_max
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_shape[0]
    ws[1] = window_shape[1]

    ip = define_iter_params(shape, ws, fraction_accepted, reduce)
    r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                curr_max = 0
                count_values = 0

                for p in range(window_shape[0]):
                    for q in range(window_shape[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            if a[i + p, j + q] > curr_max or count_values == 0:
                                curr_max = a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    r[y + ip.fringe[0], x + ip.fringe[1]] = curr_max

    free(ip)
    return r


cpdef double[:, ::1] _focal_majority(double[:, ::1] a,
                                     int[:] window_shape,
                                     np.npy_uint8[:, ::1] mask,
                                     double fraction_accepted,
                                     bint reduce,
                                     size_t mode
                                     ):
    cdef:
        size_t p, q, i, j, x, y, v, c, count_values
        double[:, ::1] r
        double curr_value
        size_t curr_max_count
        size_t shape[2]
        size_t ws[2]
        bint in_store, is_double
        double[:] values
        int[:] counts

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_shape[0]
    ws[1] = window_shape[1]

    ip = define_iter_params(shape, ws, fraction_accepted, reduce)

    values = np.full(ip.num_values, dtype=np.float64, fill_value=np.nan)
    counts = np.zeros(ip.num_values, dtype=np.int32)
    r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                values[0] = 0
                counts[0] = 0
                count_values = 0
                c = 1

                for p in range(window_shape[0]):
                    for q in range(window_shape[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            in_store = False
                            if count_values == 0:
                                values[0] = a[i + p, j + q]
                            for v in range(c):
                                if a[i + p, j + q] == values[v]:
                                    counts[v] = counts[v] + 1
                                    in_store = True
                            if not in_store:
                                values[c] = a[i + p, j + q]
                                counts[c] = 1
                                c = c + 1

                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
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
                        curr_value = NAN
                        is_double = False
                        for v in range(c):
                            if counts[v] == curr_max_count:
                                is_double = True
                            if counts[v] > curr_max_count:
                                curr_max_count = counts[v]
                                curr_value = values[v]
                                is_double = False

                        if is_double:
                            curr_value = NAN

                    r[y + ip.fringe[0], x + ip.fringe[1]] = curr_value

    free(ip)
    return r

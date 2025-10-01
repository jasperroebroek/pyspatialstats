# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
from libc.math cimport isnan, sqrt


cpdef void _focal_correlation(
    double[:, :, :, :] a1,
    double[:, :, :, :] a2,
    np.npy_uint8[:, ::1] mask,
    # return rasters
    size_t[:, :] df,
    double[:, :] c,
    # parameters
    int[:] fringe,
    double threshold,
    bint reduce,
):
    cdef:
        size_t i, j, p, q, count
        double r_num, d1_mean, d2_mean, d1_sum, d2_sum, c1_dist, c2_dist, r_den_d1, r_den_d2
        double[:, :] a1_window, a2_window

    threshold = threshold if threshold > 2 else 2

    with nogil:
        for i in range(a1.shape[0]):
            for j in range(a1.shape[1]):
                a1_window = a1[i, j]
                a2_window = a2[i, j]

                if not reduce and (isnan(a1_window[fringe[0], fringe[1]]) or isnan(a2_window[fringe[0], fringe[1]])):
                    continue

                d1_sum = 0
                d2_sum = 0
                count = 0

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if isnan(a1_window[p, q]) or isnan(a2_window[p, q]) or not mask[p, q]:
                            continue
                        d1_sum = d1_sum + a1_window[p, q]
                        d2_sum = d2_sum + a2_window[p, q]
                        count += 1

                if count < threshold:
                    continue

                d1_mean = d1_sum / count
                d2_mean = d2_sum / count

                r_num = 0
                r_den_d1 = 0
                r_den_d2 = 0

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if isnan(a1_window[p, q]) or isnan(a2_window[p, q]) or not mask[p, q]:
                            continue

                        c1_dist = a1_window[p, q] - d1_mean
                        c2_dist = a2_window[p, q] - d2_mean

                        r_num = r_num + (c1_dist * c2_dist)
                        r_den_d1 = r_den_d1 + c1_dist ** 2
                        r_den_d2 = r_den_d2 + c2_dist ** 2

                if r_den_d1 == 0 or r_den_d2 == 0:
                    c[i, j] = 0
                    df[i, j] = 0
                    continue

                c[i, j] = r_num / sqrt(r_den_d1 * r_den_d2)
                df[i, j] = count - 2

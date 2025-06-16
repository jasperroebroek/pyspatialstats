# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport isnan, sqrt
from pyspatialstats.types.cy_types cimport CyFocalLinearRegressionResult, numeric_v1, numeric_v2


cpdef void _focal_linear_regression(
    numeric_v1[:, :, :, :] a1,
    numeric_v2[:, :, :, :] a2,
    np.npy_uint8[:, ::1] mask,
    CyFocalLinearRegressionResult r,
    int[:] fringe,
    double threshold,
    bint reduce
):
    cdef:
        size_t i, j, p, q, count
        double sum_a1, sum_a2, sum_a1_a2, sum_a1_squared, sum_residuals_squared, se, ss_a1_residuals
        numeric_v1[:, :] a1_window
        numeric_v2[:, :] a2_window

    threshold = threshold if threshold > 2 else 2

    with nogil:
        for i in range(a1.shape[0]):
            for j in range(a1.shape[1]):
                a1_window = a1[i, j]
                a2_window = a2[i, j]

                if not reduce and (isnan(a1_window[fringe[0], fringe[1]]) or isnan(a2_window[fringe[0], fringe[1]])):
                    continue

                count = 0
                sum_a1 = 0
                sum_a2 = 0
                sum_a1_a2 = 0
                sum_a1_squared = 0
                sum_a2_squared = 0
                sum_residuals_squared = 0
                ss_a1_residuals = 0

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if isnan(a1_window[p, q]) or isnan(a2_window[p, q]) or not mask[p, q]:
                            continue
                        count += 1
                        sum_a1 += a1_window[p, q]
                        sum_a2 += a2_window[p, q]
                        sum_a1_a2 += a1_window[p, q] * a2_window[p, q]
                        sum_a1_squared += a1_window[p, q] * a1_window[p, q]

                if count < threshold:
                    continue

                r.df[i, j] = count - 2

                r.a[i, j] = (count * sum_a1_a2 - sum_a1 * sum_a2) / (count * sum_a1_squared - (sum_a1 * sum_a1))
                r.b[i, j] = (sum_a2 - r.a[i, j] * sum_a1) / count

                for p in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if not isnan(a1_window[p, q]) and not isnan(a2_window[p, q]) and mask[p, q]:
                            residual = a2_window[p, q] - (r.a[i, j] * a1_window[p, q] + r.b[i, j])
                            sum_residuals_squared += residual * residual

                se = sqrt(sum_residuals_squared / (count - 2))
                ss_a1_residuals = sum_a1_squared - (sum_a1 ** 2) / count

                r.se_a[i, j] = se / sqrt(ss_a1_residuals)
                r.se_b[i, j] = se * sqrt((1.0 / count) + ((sum_a1 / count) ** 2) / ss_a1_residuals)

                r.t_a[i, j] = r.a[i, j] / r.se_a[i, j]
                r.t_b[i, j] = r.b[i, j] / r.se_b[i, j]  

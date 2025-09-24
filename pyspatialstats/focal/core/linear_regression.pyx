# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free
import numpy as np
cimport numpy as np
from libc.math cimport isnan
from pyspatialstats.stats.linear_regression cimport (
    LinearRegressionState, LinearRegressionResult, lrs_reset, lrs_add, lrs_to_result, lrs_new, lrr_new
)


cpdef void _focal_linear_regression(
    double[:, :, :, :, :, :] x,
    double[:, :, :, :] y,
    np.npy_uint8[:, ::1] mask,
    double[:, :, :] beta,
    double[:, :, :] beta_se,
    double[:, :] r_squared,
    double[:, :] df,
    int[:] fringe,
    double threshold,
    bint reduce
):
    cdef:
        size_t i, j, k, r, q, nf = x.shape[5] + 1
        bint valid
        double[:, :, :] x_window
        double[:, :] y_window
        LinearRegressionState* lrs = lrs_new(nf)
        LinearRegressionResult* lrr = lrr_new(nf)

    if lrs is NULL or lrr is NULL:
        free(lrs)
        free(lrr)
        raise MemoryError("Failed to allocate memory for linear regression state and/or result")

    threshold = threshold if threshold > nf else nf

    with nogil:
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                df[i, j] = 0

                x_window = x[i, j, 0]
                y_window = y[i, j]

                if not reduce:
                    if isnan(y_window[fringe[0], fringe[1]]):
                        continue
                    valid = True
                    for k in range(nf - 1):
                        if isnan(x_window[fringe[0], fringe[1], k]):
                            valid = False
                            break
                    if not valid:
                        continue

                lrs_reset(lrs)

                for r in range(mask.shape[0]):
                    for q in range(mask.shape[1]):
                        if not mask[r, q]:
                            continue

                        if isnan(y_window[r, q]):
                            continue

                        valid = True
                        for k in range(nf - 1):
                            if isnan(x_window[r, q, k]):
                                valid = False
                                break
                        if not valid:
                            continue

                        lrs_add(lrs, y_window[r, q], x_window[r, q])

                if lrs.count < threshold:
                    continue

                lrs_to_result(lrs, lrr)

                df[i, j] = lrr.df
                r_squared[i, j] = lrr.r_squared

                for k in range(nf):
                    beta[i, j, k] = lrr.beta[k]
                    beta_se[i, j, k] = lrr.beta_se[k]

    free(lrs)
    free(lrr)

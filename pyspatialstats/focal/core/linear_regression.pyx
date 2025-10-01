# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free
import numpy as np
cimport numpy as np
from libc.math cimport isnan
from pyspatialstats.bootstrap.linear_regression cimport bootstrap_linear_regression
from pyspatialstats.random.random cimport Random
from pyspatialstats.stats.linear_regression cimport (
    LinearRegressionState, LinearRegressionResult, lrs_reset, lrs_add, lrs_to_result, lrs_new, lrr_new
)
from pyspatialstats.stats.welford cimport WelfordState, ws_array_new, ws_new


cpdef void _focal_linear_regression_simple(
    double[:, :, :, :, :, :] x,
    double[:, :, :, :] y,
    np.npy_uint8[:, ::1] mask,
    double[:, :, :] beta,
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

                lrs_to_result(lrs, lrr, False, False)

                if lrr.status > 0:
                    df[i, j] = -lrr.status
                    continue

                df[i, j] = lrr.df
                for k in range(nf):
                    beta[i, j, k] = lrr.beta[k]

    free(lrs)
    free(lrr)


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

                lrs_to_result(lrs, lrr, True, True)

                if lrr.status > 0:
                    df[i, j] = -lrr.status
                    continue

                df[i, j] = lrr.df
                r_squared[i, j] = lrr.r_squared

                for k in range(nf):
                    beta[i, j, k] = lrr.beta[k]
                    beta_se[i, j, k] = lrr.beta_se[k]

    free(lrs)
    free(lrr)


cpdef _focal_linear_regression_bootstrap(
    double[:, :, :, :, :, :] x,
    double[:, :, :, :] y,
    np.npy_uint8[:, ::1] mask,
    double[:, :, :] beta,
    double[:, :, :] beta_se,
    double[:, :] r_squared,
    double[:, :] r_squared_se,
    double[:, :] df,
    int[:] fringe,
    double threshold,
    bint reduce,
    size_t n_boot,
    int seed
):
    cdef:
        size_t count_values
        size_t i, j, k, r, q, nf = x.shape[5] + 1
        bint valid
        double[:, :, :] x_window
        double[:, :] y_window
        double[:, :] x_window_values = np.empty((mask.shape[0] * mask.shape[1], nf - 1), dtype=np.float64)
        double[:] y_window_values = np.empty((mask.shape[0] * mask.shape[1]), dtype=np.float64)

        LinearRegressionState* lrs_tmp = lrs_new(nf)
        LinearRegressionResult* lrr_tmp = lrr_new(nf)
        LinearRegressionResult* lrr = lrr_new(nf)

        WelfordState* ws_r2 = ws_new()
        WelfordState* ws_beta = ws_array_new(nf)

    if lrs_tmp is NULL or lrr_tmp is NULL or lrr is NULL:
        free(lrs_tmp)
        free(lrr_tmp)
        free(lrr)
        raise MemoryError("Failed to allocate memory for bootstrap linear regression")

    rng = Random(seed)

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

                count_values = 0

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

                        y_window_values[count_values] = y_window[r, q]
                        for k in range(nf - 1):
                            x_window_values[count_values, k] = x_window[r, q, k]
                        count_values += 1

                if count_values < threshold:
                    continue

                bootstrap_linear_regression(
                    x_window_values,
                    y_window_values,
                    count_values,
                    n_boot,
                    rng,
                    lrs_tmp,
                    lrr_tmp,
                    lrr,
                    ws_r2,
                    ws_beta
                )

                if lrr.status > 0:
                    df[i, j] = -lrr.status
                    continue

                df[i, j] = lrr.df
                r_squared[i, j] = lrr.r_squared
                r_squared_se[i, j] = lrr.r_squared_se

                for k in range(nf):
                    beta[i, j, k] = lrr.beta[k]
                    beta_se[i, j, k] = lrr.beta_se[k]

    free(lrs_tmp)
    free(lrr_tmp)
    free(lrr)

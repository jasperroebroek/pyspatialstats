# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

from libc.stdlib cimport free
from pyspatialstats.random.random cimport Random
from pyspatialstats.stats.linear_regression cimport (
    CyRegressionResult, LinearRegressionResult, lrr_new, lrr_reset, lrr_free
)
from pyspatialstats.stats.linear_regression cimport (
    LinearRegressionState, lrs_reset, lrs_add, lrs_to_result, lrs_new, lrs_free
)
from pyspatialstats.stats.welford cimport WelfordState, ws_reset, ws_add, ws_mean, ws_std, ws_array_new, ws_new, ws_free


cdef void bootstrap_linear_regression(
    double[:, :] x,                     # n_samples × n_features
    double[:] y,                        # vector length n_samples
    size_t n_samples,
    size_t n_boot,
    Random rng,
    LinearRegressionState* lrs_tmp,     # scratch state
    LinearRegressionResult* lrr_tmp,    # scratch result
    LinearRegressionResult* lrr,        # output bootstrap result
    WelfordState* ws_r2,                # external accumulator for R²
    WelfordState* ws_beta               # external accumulators for β (array of size nf)
) noexcept nogil:
    """
    Bootstrapped linear regression. Fits models on resampled data,
    aggregates β and R² across bootstraps, and stores results in out_lrr.
    """
    cdef:
        size_t i, j, k, idx, nf = lrr.nf

    # Reset accumulators
    ws_reset(ws_r2)
    for k in range(nf):
        ws_reset(&ws_beta[k])

    # Bootstrap loop
    for i in range(n_boot):
        lrs_reset(lrs_tmp)

        for j in range(n_samples):
            idx = rng.integer(bound=n_samples)
            lrs_add(lrs_tmp, y[idx], x[idx, :])

        lrs_to_result(lrs_tmp, lrr_tmp, True, True)

        if lrr_tmp.status == 2:
            lrr_reset(lrr)
            lrr.status = lrr_tmp.status
            return

        # Model failed to converge or too few observations
        if lrr_tmp.status > 0 and lrr_tmp.status != 2:
            continue

        for k in range(nf):
            ws_add(&ws_beta[k], lrr_tmp.beta[k])
        ws_add(ws_r2, lrr_tmp.r_squared)

    lrr.df = lrr_tmp.df  # They should all be the same

    if ws_r2.count < 2:
        lrr.status = 1
        return

    # Write aggregated results into lrr
    lrr.status = 0
    lrr.r_squared = ws_mean(ws_r2)
    lrr.r_squared_se = ws_std(ws_r2)

    for k in range(nf):
        lrr.beta[k] = ws_mean(&ws_beta[k])
        lrr.beta_se[k] = ws_std(&ws_beta[k])


def py_bootstrap_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 0
) -> CyRegressionResult:
    """
    Python wrapper for bootstrap_linear_regression. Debugging only
    """
    cdef:
        size_t k
        size_t n_samples = x.shape[0]
        size_t nf = x.shape[1] + 1
        Random rng = Random(seed)
        LinearRegressionState* lrs_tmp = NULL
        LinearRegressionResult* lrr_tmp = NULL
        LinearRegressionResult* lrr = NULL
        WelfordState* ws_r2 = NULL
        WelfordState* ws_beta = NULL

    print(f"{nf=}")

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same number of rows")
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    if n_bootstraps < 2:
        raise ValueError("Bootstrap sample size must be at least 2")
    if n_samples < 2:
        raise ValueError("At least 2 samples need to be present")

    lrr = lrr_new(nf)

    if lrr == NULL:
        raise MemoryError()

    try:
        lrs_tmp = lrs_new(nf)
        lrr_tmp = lrr_new(nf)
        ws_r2 = ws_new()
        ws_beta = ws_array_new(nf)

        if lrs_tmp == NULL or lrr_tmp == NULL or ws_r2 == NULL or ws_beta == NULL:
            raise MemoryError()

        bootstrap_linear_regression(
            x, y, n_samples, n_bootstraps,
            rng, lrs_tmp, lrr_tmp, lrr,
            ws_r2, ws_beta
        )

    finally:
        free(ws_beta)
        ws_free(ws_r2)
        lrr_free(lrr_tmp)
        lrs_free(lrs_tmp)

    r = CyRegressionResult._from_c_struct(lrr, n_params=nf)
    lrr_free(lrr)
    return r

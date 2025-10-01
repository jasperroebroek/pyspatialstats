from pyspatialstats.random.random cimport Random
from pyspatialstats.stats.linear_regression cimport LinearRegressionState, LinearRegressionResult
from pyspatialstats.stats.welford cimport WelfordState


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
) noexcept nogil

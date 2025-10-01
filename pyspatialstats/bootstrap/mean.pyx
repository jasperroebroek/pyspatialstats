# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from pyspatialstats.results.stats import MeanResult

from pyspatialstats.stats.welford cimport WelfordState, ws_reset, ws_add, ws_mean, ws_std
from pyspatialstats.random.random cimport Random


cdef void bootstrap_mean(double* v, size_t n_samples, size_t n_boot, Random rng, WelfordState *result) noexcept nogil:
    """This function does not have to look for NaNs, they are filtered out before"""
    cdef:
        size_t i, j
        double mean

    ws_reset(result)

    for i in range(n_boot):
        mean = 0
        for j in range(n_samples):
            mean += v[rng.integer(bound=n_samples)]
        mean /= n_samples
        ws_add(result, mean)


def py_bootstrap_mean(v: np.ndarray, n_bootstraps: int, seed: int = 0) -> MeanResult:
    cdef:
        double[:] v_parsed = np.asarray(v, dtype=np.float64)
        Random rng = Random(seed)
        WelfordState result = WelfordState()

    if n_bootstraps < 2:
        raise ValueError("Bootstrap sample size must be at least 2")

    n_samples = v_parsed.size

    if n_samples < 2:
        raise ValueError("At least 2 samples need to be present")

    bootstrap_mean(&v_parsed[0], n_samples, n_bootstraps, rng, &result)

    return MeanResult(mean=ws_mean(&result), se=ws_std(&result))

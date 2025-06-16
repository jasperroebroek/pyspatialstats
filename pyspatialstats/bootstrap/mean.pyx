import numpy as np

from pyspatialstats.random.random cimport RandomInts
from libc.stdlib cimport calloc, free
from libc.math cimport sqrt


cdef CyBootstrapMeanResult _bootstrap_mean(double* v, size_t n_samples, size_t n_bootstraps, RandomInts rng, double *means) noexcept nogil:
    """This function does not have to look for NaNs, they are filtered out before"""
    cdef:
        size_t i, j
        double mean, se, mean_sum = 0, sum_squared_diff = 0

    for i in range(n_bootstraps):
        means[i] = 0
        for j in range(n_samples):
            means[i] += v[rng.next_value(bound=n_samples)]
        means[i] /= n_samples
        mean_sum += means[i]

    mean = mean_sum / n_bootstraps

    for i in range(n_bootstraps):
        sum_squared_diff += (means[i] - mean) ** 2

    se = sqrt(sum_squared_diff / n_bootstraps)

    # with gil:
    #     for i in range(n_bootstraps):
    #         print(f"means[{i}]={means[i]}")
    #     print(f"{mean=}")

    return CyBootstrapMeanResult(mean=mean, se=se)


def py_bootstrap_mean(v: np.ndarray, n_bootstraps: int, seed: int = 0) -> tuple[double, double]:
    cdef:
        double *means = <double *> calloc(n_bootstraps, sizeof(double))
        double[:] v_parsed = np.asarray(v, dtype=np.float64)
        RandomInts rng = RandomInts(seed)
        
    if means == NULL:
        raise MemoryError("Memory allocation failed for means array")

    if n_bootstraps < 2:
        raise ValueError("Bootstrap sample size must be at least 2")

    n_samples = v_parsed.size

    if n_samples < 2:
        raise ValueError("At least 2 samples need to be present")

    result = _bootstrap_mean(&v_parsed[0], n_samples, n_bootstraps, rng, means)

    free(means)
    return result.mean, result.se

from pyspatialstats.random.random cimport RandomInts

cdef struct CyBootstrapMeanResult:
    double mean
    double se

cdef CyBootstrapMeanResult _bootstrap_mean(
    double* v,
    size_t n_samples,
    size_t n_bootstraps,
    RandomInts rng,
    double *means
) noexcept nogil

from pyspatialstats.random.random cimport RandomInts

cdef struct CyGroupedBootstrapMeanResult:
    double *mean_v
    double *se_v

cdef double* _grouped_mean(size_t[:] ind, double[:] v, size_t max_ind) except NULL nogil
cdef CyGroupedBootstrapMeanResult _grouped_mean_bootstrap(
    size_t[:] ind,
    double[:] v,
    size_t max_ind,
    size_t n_bootstraps,
    RandomInts rng
) nogil

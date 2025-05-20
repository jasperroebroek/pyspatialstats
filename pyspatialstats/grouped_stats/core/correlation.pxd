cdef struct CyGroupedCorrelationResult:
    double *c, *t
    size_t *df

cdef CyGroupedCorrelationResult _grouped_correlation(size_t[:] ind, double[:] v1, double[:] v2, size_t max_ind) except * nogil

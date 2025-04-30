cdef struct CyGroupedCorrelationResult:
    float *c
    float *t
    long *df

cdef CyGroupedCorrelationResult _grouped_correlation(size_t[:] ind, float[:] v1, float[:] v2, size_t max_ind) except * nogil

from pyspatialstats.types.cy_types cimport CyGroupedCorrelationResult

cdef CyGroupedCorrelationResult _grouped_correlation(size_t[:] ind, double[:] v1, double[:] v2, size_t max_ind) except * nogil

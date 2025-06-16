from pyspatialstats.types.cy_types cimport CyGroupedLinearRegressionResult

cdef CyGroupedLinearRegressionResult _grouped_linear_regression(
    size_t[:] ind,
    double[:] v1,
    double[:] v2,
    size_t max_ind
) except * nogil

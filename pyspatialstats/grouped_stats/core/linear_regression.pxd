cdef struct CyGroupedLinearRegressionResult:
    size_t *df
    double *a, *b, *se_a, *se_b, *t_a, *t_b


cdef CyGroupedLinearRegressionResult _grouped_linear_regression(size_t[:] ind,
                                                                double[:] v1,
                                                                double[:] v2,
                                                                size_t max_ind) except * nogil

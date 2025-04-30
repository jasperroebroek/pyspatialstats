cdef struct CyGroupedLinearRegressionResult:
    long *df
    float *a
    float *b
    float *se_a
    float *se_b
    float *t_a
    float *t_b


cdef CyGroupedLinearRegressionResult _grouped_linear_regression(size_t[:] ind,
                                                                float[:] v1,
                                                                float[:] v2,
                                                                size_t max_ind) except * nogil

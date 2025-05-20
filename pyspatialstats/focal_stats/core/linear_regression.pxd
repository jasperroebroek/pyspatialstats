cdef class CyFocalLinearRegressionResult:
    cdef:
        size_t[:, :] df
        double[:, :] a, b, se_a, se_b, t_a, t_b

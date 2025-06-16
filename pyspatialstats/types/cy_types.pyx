cdef class CyFocalCorrelationResult:
    def __cinit__(
        self,
        size_t[:, :] df,
        double[:, :] c,
    ):
        self.df = df
        self.c = c


cdef class CyFocalLinearRegressionResult:
    def __cinit__(
        self,
        size_t[:, :] df,
        double[:, :] a,
        double[:, :] b,
        double[:, :] se_a,
        double[:, :] se_b,
        double[:, :] t_a,
        double[:, :] t_b,
    ):
        self.df = df
        self.a = a
        self.b = b
        self.se_a = se_a
        self.se_b = se_b
        self.t_a = t_a
        self.t_b = t_b
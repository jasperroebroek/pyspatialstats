ctypedef fused numeric:
    int
    long
    long long
    float
    double

ctypedef numeric numeric_v1
ctypedef numeric numeric_v2

ctypedef fused numeric_return:
    double
    size_t


cdef struct CyGroupedCorrelationResult:
    double *c
    double *t
    size_t *df

cdef struct CyGroupedLinearRegressionResult:
    size_t *df
    double *a
    double *b
    double *se_a
    double *se_b
    double *t_a
    double *t_b

cdef class CyFocalCorrelationResult:
    cdef:
        size_t[:, :] df
        double[:, :] c

cdef class CyFocalLinearRegressionResult:
    cdef:
        size_t[:, :] df
        double[:, :] a, b, se_a, se_b, t_a, t_b
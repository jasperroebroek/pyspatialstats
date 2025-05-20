cdef class CyFocalCorrelationResult:
    cdef:
        size_t[:, :] df
        double[:, :] c

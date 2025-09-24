from numpy.random cimport bitgen_t
from libc.stdint cimport uint64_t, int64_t

cdef class Random:
    cdef:
        object py_gen
        bitgen_t *rng
    cdef inline uint64_t integer(self, uint64_t bound) noexcept nogil
    cdef uint64_t[:] randints(self, uint64_t bound, int n)
    cdef inline int64_t poisson(self, double lam) noexcept nogil
    cdef int64_t[:] randpoisson(self, double lam, int n)

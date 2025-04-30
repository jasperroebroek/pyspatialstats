from numpy.random cimport bitgen_t
from libc.stdint cimport uint64_t

cdef class RandomInts:
    cdef:
        object py_gen
        bitgen_t *rng
    cdef inline uint64_t next_value(self, uint64_t bound) noexcept nogil
    cdef uint64_t[:] randints(self, uint64_t bound, int n)

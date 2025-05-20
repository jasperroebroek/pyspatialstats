# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from numpy.random import PCG64

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from libc.stdint cimport uint64_t
from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport random_bounded_uint64


cdef const char *capsule_name = "BitGenerator"


cdef class RandomInts:
    """
    Code adapted from a mix of the numpy documentation and
    https://gist.github.com/ev-br/3d3e5b9682ef8c147e457e9cd2b190a0

    Not safe for multi-threading, race conditions occur

    Drawn values are exclusive of bound
    """
    def __init__(self, seed=0):
        self.py_gen= PCG64(seed)
        capsule = self.py_gen.capsule
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        self.rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    cdef inline uint64_t next_value(self, uint64_t bound) noexcept nogil:
        """random_bounded_uint64 returns a value including rng"""
        return random_bounded_uint64(self.rng, off=0, rng=bound - 1, mask=0, use_masked=0)

    cdef uint64_t[:] randints(self, uint64_t bound, int n):
        cdef:
            int i
            uint64_t[:] r = np.empty(n, dtype=np.uint64)

        for i in range(n):
            r[i] = self.next_value(bound)

        return r

    def np_randints(self, bound: int, n: int) -> np.ndarray:
        return np.asarray(self.randints(bound, n))

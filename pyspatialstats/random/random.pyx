# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from numpy.random import PCG64

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from libc.stdint cimport uint64_t, int64_t
from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport random_bounded_uint64, random_poisson

cdef const char *capsule_name = "BitGenerator"


cdef class Random:
    """
    This Cython class uses NumPy’s PCG64 bit generator for fast random number generation.

    Note:
    - `np_randints(bound, n)`: generates `n` random integers in `[0, bound)`
    - `np_randpoisson(lam, n)`: generates `n` samples from Poisson distribution with mean `lam`
    """
    def __init__(self, seed=0):
        self.py_gen= PCG64(seed)
        capsule = self.py_gen.capsule
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        self.rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    cdef inline uint64_t integer(self, uint64_t bound) noexcept nogil:
        """random_bounded_uint64 returns a value including rng"""
        return random_bounded_uint64(self.rng, off=0, rng=bound - 1, mask=0, use_masked=0)

    cdef uint64_t[:] randints(self, uint64_t bound, int n):
        cdef:
            int i
            uint64_t[:] r = np.empty(n, dtype=np.uint64)

        for i in range(n):
            r[i] = self.integer(bound)

        return r

    cdef inline int64_t poisson(self, double lam) noexcept nogil:
        return random_poisson(self.rng, lam)

    cdef int64_t[:] randpoisson(self, double lam, int n):
        cdef:
            int i
            int64_t[:] r = np.empty(n, dtype=np.int64)

        for i in range(n):
            r[i] = self.poisson(lam)

        return r

    def np_randints(self, bound: int, n: int) -> np.ndarray:
        return np.asarray(self.randints(bound, n))

    def np_randpoisson(self, lam: float, n: int) -> np.ndarray:
        return np.asarray(self.randpoisson(lam, n))

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt
from numpy.math cimport NAN
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memset

# =============================================================================
# CorrelationState Functions
# =============================================================================

cdef inline CorrelationState* crs_new() noexcept nogil:
    return <CorrelationState*> calloc(1, sizeof(CorrelationState))


cdef inline void crs_init(CorrelationState* crs) noexcept nogil:
    if crs == NULL:
        return
    memset(crs, 0, sizeof(CorrelationState))


cdef inline void crs_free(CorrelationState* crs) noexcept nogil:
    if crs != NULL:
        free(crs)


cdef inline void crs_reset(CorrelationState* crs) noexcept nogil:
    crs_init(crs)


cdef inline void crs_add(CorrelationState* crs, double v1, double v2) noexcept nogil:
    cdef double delta1 = v1 - crs.mean_v1
    cdef double delta2 = v2 - crs.mean_v2

    crs.count += 1
    crs.mean_v1 += delta1 / crs.count
    crs.mean_v2 += delta2 / crs.count
    crs.m2_v1 += delta1 * (v1 - crs.mean_v1)
    crs.m2_v2 += delta2 * (v2 - crs.mean_v2)
    crs.cov_v1_v2 += delta1 * (v2 - crs.mean_v2)


cdef inline void crs_merge(CorrelationState* crs_into, CorrelationState* crs_from) noexcept nogil:
    """Merge one CorrelationState into another."""
    cdef size_t n1 = crs_into.count
    cdef size_t n2 = crs_from.count
    cdef double n = n1 + n2

    if n == 0:
        crs_reset(crs_into)
        return

    cdef double delta1 = crs_from.mean_v1 - crs_into.mean_v1
    cdef double delta2 = crs_from.mean_v2 - crs_into.mean_v2

    crs_into.count = n1 + n2
    crs_into.mean_v1 += delta1 * n2 / n
    crs_into.mean_v2 += delta2 * n2 / n
    crs_into.m2_v1 += crs_from.m2_v1 + delta1 * delta1 * (n1 * n2 / n)
    crs_into.m2_v2 += crs_from.m2_v2 + delta2 * delta2 * (n1 * n2 / n)
    crs_into.cov_v1_v2 += crs_from.cov_v1_v2 + delta1 * delta2 * (n1 * n2 / n)


cdef inline double crs_corr(CorrelationState* crs) noexcept nogil:
    if crs.count < 2:
        return NAN
    cdef double den = sqrt(crs.m2_v1 * crs.m2_v2)
    if den == 0.0:
        return NAN
    return crs.cov_v1_v2 / den


cdef inline double crs_cov(CorrelationState* crs) noexcept nogil:
    if crs.count < 2:
        return NAN
    return crs.cov_v1_v2 / (crs.count - 1)


cdef inline double crs_var1(CorrelationState* crs) noexcept nogil:
    if crs.count < 2:
        return NAN
    return crs.m2_v1 / (crs.count - 1)


cdef inline double crs_var2(CorrelationState* crs) noexcept nogil:
    if crs.count < 2:
        return NAN
    return crs.m2_v2 / (crs.count - 1)


cdef inline CorrelationState* crs_array_new(size_t count) noexcept nogil:
    """
    Create an array of CorrelationState structures.

    Parameters
    ----------
    count : size_t
        Number of states to allocate

    Returns
    -------
    CorrelationState* : Pointer to array, or NULL on failure
    """
    cdef CorrelationState* crs_array = <CorrelationState*> calloc(count, sizeof(CorrelationState))
    return crs_array

# =============================================================================
# Extension type (for testing)
# =============================================================================

cdef class Correlation:
    """
    Fast correlation using Cython.

    Example
    -------
    >>> c = Correlation()
    >>> c.add(1.0, 2.0)
    >>> c.add(2.0, 3.0)
    >>> c.corr
    1.0
    """

    cdef CorrelationState* _state

    def __cinit__(self):
        self._state = crs_new()
        if self._state == NULL:
            raise MemoryError("Failed to allocate CorrelationState")

    def __dealloc__(self):
        if self._state != NULL:
            crs_free(self._state)

    @property
    def count(self):
        return self._state.count if self._state != NULL else 0

    def reset(self):
        if self._state != NULL:
            crs_reset(self._state)

    def add(self, double v1, double v2):
        if self._state == NULL:
            raise RuntimeError("CorrelationState not initialized")
        with nogil:
            crs_add(self._state, v1, v2)

    def merge(self, Correlation other):
        if self._state == NULL or other._state == NULL:
            raise RuntimeError("CorrelationState not initialized")
        with nogil:
            crs_merge(self._state, other._state)

    @property
    def corr(self):
        return crs_corr(self._state)

    @property
    def cov(self):
        return crs_cov(self._state)

    @property
    def var1(self):
        return crs_var1(self._state)

    @property
    def var2(self):
        return crs_var2(self._state)

    def summary(self):
        return {
            "count": self.count,
            "corr": self.corr,
            "cov": self.cov,
            "var1": self.var1,
            "var2": self.var2,
        }

    def repr__(self):
        p = f"count={self.count}, corr={self.corr}"
        return f"Correlation({p})"

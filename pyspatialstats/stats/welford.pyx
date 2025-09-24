# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport sqrt
from numpy.math cimport NAN
from libc.stdlib cimport calloc, free
from libc.string cimport memset

# =============================================================================
# WelfordState Functions
# =============================================================================

cdef inline WelfordState* ws_new() noexcept nogil:
    return <WelfordState*> calloc(1, sizeof(WelfordState))


cdef inline void ws_init(WelfordState* ws) noexcept nogil:
    if ws == NULL:
        return
    memset(ws, 0, sizeof(WelfordState))


cdef inline void ws_free(WelfordState* ws) noexcept nogil:
    if ws != NULL:
        free(ws)


cdef inline void ws_reset(WelfordState* ws) noexcept nogil:
    ws_init(ws)


cdef inline void ws_add(WelfordState* ws, double v, double weight = 1.0) noexcept nogil:
    """
    Add a value `v` with weight `weight` to the Welford accumulator.
    """
    if weight <= 0:
        return

    cdef double delta
    cdef double total_count = ws.count + weight

    delta = v - ws.mean
    ws.mean += (weight / total_count) * delta
    ws.m2 += weight * delta * (v - ws.mean)
    ws.count = total_count


cdef inline void ws_merge(WelfordState* ws_into, WelfordState* ws_from) noexcept nogil:
    cdef:
        double n1, n2, n, delta

    n1 = ws_into.count
    n2 = ws_from.count
    n = n1 + n2

    if n == 0:
        ws_reset(ws_into)
        return

    delta = ws_from.mean - ws_into.mean
    ws_into.count = n1 + n2
    ws_into.mean += delta * n2 / n
    ws_into.m2 += ws_from.m2 + delta * delta * n1 * n2 / n


cdef inline double ws_mean(WelfordState* ws) noexcept nogil:
    return NAN if ws.count == 0 else ws.mean


cdef inline double ws_std(WelfordState* ws, size_t ddof = 1) noexcept nogil:
    if ws.count < 2:
        return NAN
    return sqrt(ws.m2 / (ws.count - ddof))


cdef inline WelfordState* ws_array_new(size_t count) noexcept nogil:
    return <WelfordState*> calloc(count, sizeof(WelfordState))

# =============================================================================
# Python Extension Type
# =============================================================================

cdef class Welford:
    """
    Incremental Welford accumulator for mean and standard deviation.
    """

    cdef WelfordState* _state

    def __cinit__(self):
        self._state = ws_new()
        if self._state == NULL:
            raise MemoryError("Failed to allocate WelfordState")

    def __dealloc__(self):
        if self._state != NULL:
            ws_free(self._state)

    @property
    def count(self):
        return self._state.count if self._state != NULL else 0

    def reset(self):
        if self._state != NULL:
            ws_reset(self._state)

    def add(self, double v):
        if self._state == NULL:
            raise RuntimeError("WelfordState not initialized")
        with nogil:
            ws_add(self._state, v)

    def merge(self, Welford other):
        if self._state == NULL or other._state == NULL:
            raise RuntimeError("WelfordState not initialized")
        with nogil:
            ws_merge(self._state, other._state)

    @property
    def mean(self):
        return ws_mean(self._state)

    @property
    def std(self):
        return ws_std(self._state)

    def summary(self):
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
        }

    def __repr__(self):
        return f"Welford(count={self.count}, mean={self.mean:.6f}, std={self.std:.6f})"

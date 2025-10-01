# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from typing import Optional

cimport numpy as cnp
import numpy as np
from libc.math cimport NAN
from libc.stdlib cimport free, malloc, calloc
from libc.string cimport memset

from pyspatialstats.random.random cimport Random
from pyspatialstats.results.stats import IndexedGroupedStatResult


cdef class BaseGroupedStatAccumulator:
    def __cinit__(self, capacity: int = 0):
        self.eltsize = 0
        self.capacity = 0
        self.count_v = NULL
        self.stat_v = NULL

        if capacity < 0:
            raise ValueError("capacity needs to be positive integer")

    def post_init(self, **kwargs):
        return

    def __dealloc__(self):
        self._deallocate()

    cdef void _deallocate(self) noexcept nogil:
        if self.count_v != NULL:
            free(self.count_v)
            self.count_v = NULL
        if self.stat_v != NULL:
            free(self.stat_v)
            self.stat_v = NULL

    cdef int _reset_stat_v(self) except -1 nogil:
        """
        0: success
        -1: failure
        """
        with gil:
            print("BaseGrouped reset_stat_v")
        memset(self.stat_v, 0, self.capacity * self.eltsize)
        return 0

    cdef void reset(self) noexcept nogil:
        """Reset to initial state"""
        self._deallocate()
        self.capacity = 0

    cdef int resize(self, size_t capacity) except -1 nogil:
        """
        0: success
        -1: failure
        """
        if self.eltsize == 0:
            with gil:
                raise ValueError("Element size is unknown")

        if capacity == 0:
            return 0

        self.reset()
        self.capacity = capacity

        self.count_v = <size_t *> calloc(capacity, sizeof(size_t))
        self.stat_v = malloc(capacity * self.eltsize)

        if self.stat_v == NULL or self.count_v == NULL:
            return -1

        return self._reset_stat_v()

    def py_resize(self, capacity: int):
        return self.resize(capacity)

    cdef inline size_t get_count(self, size_t ind) noexcept nogil:
        if self.count_v == NULL or ind >= self.capacity:
            return 0
        return self.count_v[ind]

    cdef size_t get_num_inds(self) noexcept nogil:
        if self.count_v == NULL:
            return 0

        cdef size_t i, c = 0
        for i in range(self.capacity):
            if self.count_v[i] > 0:
                c += 1
        return c

    cpdef size_t[:] build_indices(self):
        cdef:
            size_t i, c = 0, num_inds = self.get_num_inds()
            size_t[:] indices = np.empty(num_inds, dtype=np.uintp)

        for i in range(self.capacity):
            if self.count_v[i] > 0:
                indices[c] = i
                c += 1

        return indices

    def get_stat_v(self) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def to_dict(self):
        return {
            'capacity': self.capacity,
            'eltsize': self.eltsize,
            'num_inds': self.get_num_inds(),
            'indices': self.build_indices().base if self.count_v != NULL else None,
            'count_v': np.asarray([self.count_v[i] for i in range(self.capacity)]) if self.count_v != NULL else None,
            'stat_v': self.get_stat_v_py(),
        }


cdef class BaseGroupedBootstrapAccumulator(BaseGroupedStatAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.n_boot = 1
        self.rng = Random()

    def post_init(self, n_boot: int, seed: Optional[int] = None):
        self.n_boot = n_boot
        self.rng = Random(seed)

    cdef int bootstrap_resize(self, size_t capacity) except -1 nogil:
        return self.resize(capacity * self.n_boot)

    cpdef size_t[:] build_indices(self):
        cdef:
            size_t i, j, c = 0, n = self.capacity // self.n_boot
            size_t[:] group_counts = np.zeros(n, dtype=np.uintp)
            size_t[:] indices

        for i in range(n):
            for j in range(self.n_boot):
                group_counts[i] += self.count_v[i * self.n_boot + j]
            if group_counts[i] > 0:
                c += 1

        indices = np.empty(c, dtype=np.uintp)

        c = 0
        for i in range(n):
            if group_counts[i] > 0:
                indices[c] = i
                c += 1

        return indices


cdef class GroupedFloatStatAccumulator(BaseGroupedStatAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.eltsize = sizeof(double)
        self.resize(capacity)

    cdef int _reset_stat_v(self) except -1 nogil:
        with gil:
            print("GroupedFloat reset_stat_v")
        cdef:
            size_t i
            double *stat_v = self.get_stat_v()
        if stat_v == NULL:
            return -1
        for i in range(self.capacity):
            stat_v[i] = NAN
        return 0

    cdef double* get_stat_v(self) noexcept nogil:
        return <double *> self.stat_v

    def get_stat_v_py(self) -> Optional[np.ndarray[tuple[int], np.float64]]:
        if self.capacity == 0:
            return None
        cdef double *stat_v = self.get_stat_v()
        arr = np.empty(self.capacity, dtype=np.float64)
        for i in range(self.capacity):
            arr[i] = stat_v[i]
        return arr

    def to_result(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self.capacity == 0:
            return np.array([], dtype=np.float64)

        r = cnp.PyArray_SimpleNewFromData(1, [self.capacity], cnp.NPY_DOUBLE,  self.stat_v)
        cnp.PyArray_ENABLEFLAGS(r, cnp.NPY_ARRAY_OWNDATA)
        self.stat_v = NULL
        self.reset()

        return r

    def to_filtered_result(self) -> IndexedGroupedStatResult:
        cdef:
            size_t i
            size_t[:] indices = self.build_indices()
            size_t num_inds = indices.shape[0]

        if num_inds == 0:
            return IndexedGroupedStatResult(
                index=np.array([], dtype=np.uintp),
                result=np.array([], dtype=np.float64)
            )

        cdef:
            double[:] r = np.empty(num_inds, dtype=np.float64)
            double *stat_v = self.get_stat_v()

        for i in range(num_inds):
            r[i] = stat_v[indices[i]]

        return IndexedGroupedStatResult(index=indices.base, result=r.base)


cdef class GroupedIntStatAccumulator(BaseGroupedStatAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.eltsize = sizeof(size_t)
        self.resize(capacity)

    cdef size_t* get_stat_v(self) noexcept nogil:
        return <size_t *> self.stat_v

    def get_stat_v_py(self) -> Optional[np.ndarray[tuple[int], np.uintp]]:
        if self.capacity == 0:
            return None
        cdef size_t *stat_v = self.get_stat_v()
        arr = np.empty(self.capacity, dtype=np.uintp)
        for i in range(self.capacity):
            arr[i] = stat_v[i]
        return arr

    def to_result(self) -> np.ndarray[tuple[int], np.dtype[np.uintp]]:
        if self.capacity == 0:
            return np.array([], dtype=np.uintp)

        r = cnp.PyArray_SimpleNewFromData(1, [self.capacity], cnp.NPY_UINTP,  self.stat_v)
        cnp.PyArray_ENABLEFLAGS(r, cnp.NPY_ARRAY_OWNDATA)
        self.stat_v = NULL
        self.reset()

        return r

    def to_filtered_result(self) -> IndexedGroupedStatResult:
        cdef:
            size_t i
            size_t[:] indices = self.build_indices()
            size_t num_inds = indices.shape[0]

        if num_inds == 0:
            return IndexedGroupedStatResult(
                index=np.array([], dtype=np.uintp),
                result=np.array([], dtype=np.uintp)
            )

        cdef:
            size_t[:] r = np.empty(num_inds, dtype=np.uintp)
            size_t *stat_v = self.get_stat_v()

        for i in range(num_inds):
            r[i] = stat_v[indices[i]]

        return IndexedGroupedStatResult(index=indices.base, result=r.base)

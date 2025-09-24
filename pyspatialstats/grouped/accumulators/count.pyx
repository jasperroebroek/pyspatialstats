# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from typing import Optional
import numpy as np
cimport numpy as cnp
from libc.math cimport isnan
from pyspatialstats.grouped.indices.max cimport _define_max_ind
from pyspatialstats.results.stats import IndexedGroupedStatResult


cdef class GroupedCountAccumulator(GroupedIntStatAccumulator):
    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1:
        cdef:
            size_t i, n, max_ind
            int resize_result

        with nogil:
            n = ind.shape[0]
            max_ind = _define_max_ind(ind)
            resize_result = self.resize(max_ind + 1)
            if resize_result != 0:
                return -1
            stat_v = self.get_stat_v()

            for i in range(n):
                if isnan(v[i]):
                    continue
                self.count_v[ind[i]] += 1

        return 0

    cdef size_t* get_stat_v(self) noexcept nogil:
        return <size_t *> self.count_v

    def to_result(self) -> np.ndarray[tuple[int], np.dtype[np.uintp]]:
        if self.capacity == 0:
            return np.array([], dtype=np.uintp)

        r = cnp.PyArray_SimpleNewFromData(1, [self.capacity], cnp.NPY_UINTP,  self.count_v)
        cnp.PyArray_ENABLEFLAGS(r, cnp.NPY_ARRAY_OWNDATA)
        self.count_v = NULL
        self.reset()

        return r

    def to_filtered_result(self) -> IndexedGroupedStatResult:
        cdef:
            size_t i, num_inds
            size_t[:] r
            size_t[:] indices = self.build_indices()

        num_inds = indices.shape[0]
        r = np.empty(num_inds, dtype=np.uintp)

        for i in range(num_inds):
            r[i] = self.count_v[indices[i]]

        return IndexedGroupedStatResult(index=indices.base, result=r.base)

    def __add__(self, GroupedCountAccumulator other) -> GroupedCountAccumulator:
        cdef:
            size_t i, count, capacity = self.capacity if self.capacity > other.capacity else other.capacity
            GroupedCountAccumulator r = GroupedCountAccumulator(capacity)
            size_t *result_stat_v = r.get_stat_v()

        for i in range(capacity):
            r.count_v[i] = self.get_count(i) + other.get_count(i)

        return r

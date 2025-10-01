# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from libc.string cimport memset
from libc.math cimport NAN, isnan
from pyspatialstats.grouped.indices.max cimport _define_max_ind
from pyspatialstats.grouped.accumulators.base cimport GroupedFloatStatAccumulator
from pyspatialstats.results.stats import IndexedGroupedStatResult
from pyspatialstats.results.stats import MeanResult


cdef class GroupedSumAccumulator(GroupedFloatStatAccumulator):
    # This method is needed, although it is exactly the same as the GroupedFloatStatAccumulator version, because if not
    # present the _reset_stat_v from the GroupedFloatStatAccumulator class will be called upon initialisation.
    # Initialisation of this class will cause them to be called one after the other, with double allocation. To avoid
    # this, just call resize after initialisation of an empty class.
    def __cinit__(self, capacity: int = 0):
        self.eltsize = sizeof(double)
        self.resize(capacity)

    cdef int _reset_stat_v(self) except -1 nogil:
        if self.stat_v == NULL:
            return -1
        memset(self.stat_v, 0, self.capacity * self.eltsize)
        return 0

    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1:
        cdef:
            size_t i, group, n, max_ind
            double *sum_v
            double value

        with nogil:
            n = ind.shape[0]
            max_ind = _define_max_ind(ind)
            resize_result = self.resize(max_ind + 1)
            if resize_result != 0:
                return -1
            sum_v = self.get_stat_v()

            for i in range(n):
                value = v[i]
                if isnan(value):
                    continue

                group = ind[i]
                self.count_v[group] += 1
                sum_v[group] += value

        return 0

    def __add__(self, GroupedSumAccumulator other) -> GroupedSumAccumulator:
        cdef:
            size_t i, capacity = self.capacity if self.capacity > other.capacity else other.capacity
            GroupedSumAccumulator r = GroupedSumAccumulator()

        # This can't be done in the initialiser, because that causes the GroupedFloatBaseAccumulator _reset_stat_v to
        # be called before calling the version of this class, causing double the work.
        r.resize(capacity)

        cdef:
            double *result_stat_v = r.get_stat_v()
            double *self_stat_v = self.get_stat_v()
            double *other_stat_v = other.get_stat_v()

            size_t self_count, other_count

        for i in range(r.capacity):
            self_count = self.get_count(i)
            other_count = other.get_count(i)
            r.count_v[i] = self_count + other_count

            if self_count > 0:
                result_stat_v[i] += self_stat_v[i]
            if other_count > 0:
                result_stat_v[i] += other_stat_v[i]

        return r

    def to_mean_result(self):
        cdef:
            double *stat_v = self.get_stat_v()
            double[:] result = np.empty(self.capacity)

        for i in range(self.capacity):
            if self.count_v[i] == 0:
                result[i] = NAN
            else:
                result[i] = stat_v[i] / self.count_v[i]

        return MeanResult(mean=result.base)

    def to_mean_filtered_result(self):
        cdef:
            double *stat_v = self.get_stat_v()
            size_t[:] indices = self.build_indices()
            size_t n = indices.shape[0]
            double[:] result = np.empty(n)

        for i in range(n):
            index = indices[i]
            if self.count_v[index] == 0:
                result[i] = NAN
            else:
                result[i] = stat_v[index] / self.count_v[index]

        return IndexedGroupedStatResult(index=indices.base, result=MeanResult(mean=result.base))

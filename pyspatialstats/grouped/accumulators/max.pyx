# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.string cimport memcpy
from libc.math cimport isnan, fmax
from pyspatialstats.grouped.indices.max cimport _define_max_ind
from pyspatialstats.grouped.accumulators.base cimport GroupedFloatStatAccumulator


cdef class GroupedMaxAccumulator(GroupedFloatStatAccumulator):
    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1:
        cdef:
            size_t i, group, n, max_ind
            double value
            double *max_v
            int resize_result

        with nogil:
            n = ind.shape[0]
            max_ind = _define_max_ind(ind)
            resize_result = self.resize(max_ind + 1)
            if resize_result != 0:
                return -1

            max_v = self.get_stat_v()

            for i in range(n):
                value = v[i]
                if isnan(value):
                    continue

                group = ind[i]

                if self.count_v[group] == 0:
                    max_v[group] = value
                elif value > max_v[group]:
                    max_v[group] = value

                self.count_v[group] += 1

        return 0

    def __add__(self, GroupedMaxAccumulator other):
        cdef:
            size_t i
            size_t min_capacity = self.capacity if self.capacity < other.capacity else other.capacity
            size_t max_capacity = self.capacity if self.capacity > other.capacity else other.capacity

        if max_capacity == 0:
            return self

        cdef:
            GroupedMaxAccumulator r = GroupedMaxAccumulator(max_capacity)

            double *result_stat_v = r.get_stat_v()
            double *self_stat_v = self.get_stat_v()
            double *other_stat_v = other.get_stat_v()

            double *source_stat_v
            size_t *source_count_v

        for i in range(min_capacity):
            result_stat_v[i] = fmax(self_stat_v[i], other_stat_v[i])
            r.count_v[i] = self.count_v[i] + other.count_v[i]

        if self.capacity > other.capacity:
            source_stat_v = self_stat_v
            source_count_v = self.count_v
        else:
            source_stat_v = other_stat_v
            source_count_v = other.count_v

        memcpy(result_stat_v + min_capacity, source_stat_v + min_capacity,
               (max_capacity - min_capacity) * sizeof(double))
        memcpy(r.count_v + min_capacity, source_count_v + min_capacity,
               (max_capacity - min_capacity) * sizeof(size_t))

        return r

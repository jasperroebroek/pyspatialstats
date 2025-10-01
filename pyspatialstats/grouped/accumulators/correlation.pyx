# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from pyspatialstats.results.stats import CorrelationResult
from pyspatialstats.results.stats import IndexedGroupedStatResult
from pyspatialstats.stats.p_values import calculate_p_value
import numpy as np

from libc.string cimport memcpy
from pyspatialstats.stats.correlation cimport CorrelationState, crs_array_new, crs_corr, crs_add, crs_merge, crs_new
from pyspatialstats.grouped.indices.max cimport _define_max_ind
from pyspatialstats.grouped.accumulators.base cimport BaseGroupedStatAccumulator
from libc.math cimport isnan


cdef class GroupedCorrelationAccumulator(BaseGroupedStatAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.eltsize = sizeof(CorrelationState)
        self.resize(capacity)

    cdef CorrelationState* get_stat_v(self) noexcept nogil:
        return <CorrelationState *> self.stat_v

    def get_stat_v_py(self):
        cdef:
            size_t i
            CorrelationState *stat_v = self.get_stat_v()
        return np.asarray(
            [
                (stat_v[i].mean_v1, stat_v[i].mean_v2, stat_v[i].m2_v1, stat_v[i].m2_v2, stat_v[i].cov_v1_v2)
                for i in range(self.capacity)
            ]
        )

    cpdef int add_data(self, size_t[:] ind, double[:] v1, double[:] v2) except -1:
        cdef:
            size_t i, group, n, max_ind
            int resize_result
            double value1, value2
            CorrelationState *stat_v

        with nogil:
            n = ind.shape[0]
            max_ind = _define_max_ind(ind)
            resize_result = self.resize(max_ind + 1)
            if resize_result != 0:
                return -1
            stat_v = self.get_stat_v()

            for i in range(n):
                value1 = v1[i]
                value2 = v2[i]
                if isnan(value1) or isnan(value2):
                    continue

                group = ind[i]
                self.count_v[group] += 1
                crs_add(&stat_v[group], value1, value2)

        return 0

    def __add__(self, GroupedCorrelationAccumulator other) -> GroupedCorrelationAccumulator:
        cdef:
            size_t i, capacity = self.capacity if self.capacity > other.capacity else other.capacity
            GroupedCorrelationAccumulator r = GroupedCorrelationAccumulator(capacity)
            size_t self_count, other_count, n

            CorrelationState *self_stat_v = self.get_stat_v()
            CorrelationState *other_stat_v = other.get_stat_v()
            CorrelationState *result_stat_v = r.get_stat_v()

        for i in range(self.capacity):
            self_count = self.get_count(i)
            other_count = other.get_count(i)

            r.count_v[i] = self_count + other_count

            if self_count > 0:
                crs_merge(&result_stat_v[i], &self_stat_v[i])
            if other_count > 0:
                crs_merge(&result_stat_v[i], &other_stat_v[i])

        return r

    def to_result(self) -> CorrelationResult:
        cdef:
            size_t i
            CorrelationState *stat_v = self.get_stat_v()
            double[:] r = np.empty(self.capacity, dtype=np.float64)
            size_t[:] df = np.empty(self.capacity, dtype=np.uintp)

        for i in range(self.capacity):
            r[i] = crs_corr(&stat_v[i])
            df[i] = (self.count_v[i] - 2) if self.count_v[i] > 2 else 0

        np_r = r.base
        np_df = df.base
        np_t = np_r * np.sqrt(np_df / (1 - np_r**2))

        return CorrelationResult(c=np_r, df=np_df, p=calculate_p_value(np_t, np_df))

    def to_filtered_result(self) -> IndexedGroupedStatResult:
        cdef:
            size_t[:] indices = self.build_indices()
            size_t i, idx, num_inds = indices.shape[0]
            double[:] r = np.empty(num_inds, dtype=np.float64)
            size_t[:] df = np.empty(num_inds, dtype=np.uintp)
            CorrelationState *stat_v = self.get_stat_v()

        for i in range(num_inds):
            idx = indices[i]
            r[i] = crs_corr(&stat_v[idx])
            df[i] = (self.count_v[idx] - 2) if self.count_v[idx] > 2 else 0

        np_r = r.base
        np_df = df.base
        np_t = np_r * np.sqrt(np_df / (1 - np_r**2))

        return IndexedGroupedStatResult(
            index=indices.base,
            result=CorrelationResult(c=np_r, df=np_df, p=calculate_p_value(np_t, np_df))
        )

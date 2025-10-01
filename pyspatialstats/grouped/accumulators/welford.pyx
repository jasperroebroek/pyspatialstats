# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from typing import Literal
import numpy as np
from pyspatialstats.results.stats import IndexedGroupedStatResult
from pyspatialstats.results.stats import MeanResult

from libc.string cimport memcpy
from libc.math cimport isnan
from pyspatialstats.grouped.indices.max cimport _define_max_ind
from pyspatialstats.grouped.accumulators.base cimport BaseGroupedStatAccumulator
from pyspatialstats.stats.welford cimport WelfordState, ws_add, ws_std, ws_merge


cdef class GroupedWelfordAccumulator(BaseGroupedStatAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.eltsize = sizeof(WelfordState)
        self.std_df = 1
        self.resize(capacity)

    def post_init(self, std_df: Literal[0, 1] = 1):
        self.std_df = std_df

    cdef WelfordState* get_stat_v(self) noexcept nogil:
        return <WelfordState *> self.stat_v

    def get_stat_v_py(self):
        cdef:
            size_t i
            WelfordState *stat_v = self.get_stat_v()
        return np.asarray([(stat_v[i].mean, stat_v[i].m2) for i in range(self.capacity)])

    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1:
        cdef:
            size_t i, group, n, max_ind
            int resize_result
            double value
            WelfordState *stat_v

        with nogil:
            n = ind.shape[0]
            max_ind = _define_max_ind(ind)
            resize_result = self.resize(max_ind + 1)
            if resize_result != 0:
                return -1
            stat_v = self.get_stat_v()

            for i in range(n):
                value = v[i]
                if isnan(value):
                    continue

                group = ind[i]
                self.count_v[group] += 1
                ws_add(&stat_v[group], value)

        return 0

    def __add__(self, GroupedWelfordAccumulator other) -> GroupedWelfordAccumulator:
        cdef:
            size_t i, capacity = self.capacity if self.capacity > other.capacity else other.capacity
            GroupedWelfordAccumulator r = GroupedWelfordAccumulator(capacity)

            size_t self_count, other_count

            WelfordState *r_stat = r.get_stat_v()
            WelfordState *self_stat = self.get_stat_v()
            WelfordState *other_stat = other.get_stat_v()

        for i in range(capacity):
            self_count = self.get_count(i)
            other_count = other.get_count(i)

            r.count_v[i] = self_count + other_count

            if self_count > 0:
                ws_merge(&r_stat[i], &self_stat[i])
            if other_count > 0:
                ws_merge(&r_stat[i], &other_stat[i])

            if r_stat[i].m2 < 0:
                r_stat[i].m2 = 0

        return r

    def to_std_result(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        cdef:
            double[:] r
            size_t i, n
            WelfordState* stat_v = self.get_stat_v()

        r = np.empty(self.capacity, dtype=np.float64)

        for i in range(self.capacity):
            r[i] = ws_std(&stat_v[i], self.std_df)

        return r.base

    def to_std_filtered_result(self):
        cdef:
            size_t[:] indices = self.build_indices()
            size_t i, idx, num_inds = indices.shape[0]
            double[:] r = np.empty(num_inds, dtype=np.float64)
            WelfordState* stat_v = self.get_stat_v()

        for i in range(num_inds):
            idx = indices[i]
            r[i] = ws_std(&stat_v[idx], self.std_df)

        return IndexedGroupedStatResult(index=indices.base, result=r.base)

    def to_mean_result(self):
        cdef:
            double[:] r
            size_t i, n
            WelfordState* stat_v = self.get_stat_v()

        r = np.empty(self.capacity, dtype=np.float64)

        for i in range(self.capacity):
            r[i] = stat_v[i].mean

        return r.base

    def to_mean_std_result(self):
        return MeanResult(mean=self.to_mean_result(), std=self.to_std_result())

    def to_mean_std_filtered_result(self):
        cdef:
            size_t[:] indices = self.build_indices()
            size_t i, idx, num_inds = indices.shape[0]
            double[:] mean = np.empty(num_inds, dtype=np.float64)
            double[:] std = np.empty(num_inds, dtype=np.float64)
            WelfordState* stat_v = self.get_stat_v()

        for i in range(num_inds):
            idx = indices[i]
            mean[i] = stat_v[idx].mean
            std[i] = ws_std(&stat_v[idx], self.std_df)

        np_mean = mean.base
        np_std = std.base
        np_se = np_std / np_mean

        return IndexedGroupedStatResult(
            index=indices.base, result=MeanResult(mean=np_mean, std=np_std, se=np_se)
        )

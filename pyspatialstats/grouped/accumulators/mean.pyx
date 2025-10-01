# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from pyspatialstats.results.stats import IndexedGroupedStatResult
from pyspatialstats.results.stats import MeanResult
from typing import Optional
import numpy as np

from pyspatialstats.grouped.indices.max cimport _define_max_ind
from pyspatialstats.grouped.accumulators.base cimport BaseGroupedBootstrapAccumulator
from pyspatialstats.stats.welford cimport WelfordState, ws_new, ws_add, ws_std, ws_reset, ws_mean
from libc.math cimport isnan
from libc.stdlib cimport free
from libc.string cimport memcpy


cdef class GroupedBootstrapMeanAccumulator(BaseGroupedBootstrapAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.eltsize = sizeof(double)
        self.bootstrap_resize(capacity)

    cdef double* get_stat_v(self) noexcept nogil:
        return <double *> self.stat_v

    def get_stat_v_py(self) -> Optional[np.ndarray[tuple[int], np.float64]]:
        if self.capacity == 0:
            return None
        cdef double[:] arr = np.empty(self.capacity, dtype=np.float64)
        memcpy(<void*> &arr[0], self.stat_v, self.capacity * sizeof(double))
        return arr.base

    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1:
        cdef:
            size_t i, j, group, max_ind, idx, n = v.shape[0]
            long weight
            double value
            double *sum_v
            size_t *weight_v
            int resize_result

        with nogil:
            max_ind = _define_max_ind(ind)
            resize_result = self.bootstrap_resize(max_ind + 1)
            if resize_result != 0:
                return -1

            sum_v = self.get_stat_v()
            weight_v = self.count_v

            for i in range(n):
                value = v[i]
                if isnan(value):
                    continue

                group = ind[i]

                for j in range(self.n_boot):
                    idx = group * self.n_boot + j
                    weight = self.rng.poisson(1)
                    sum_v[idx] += value * weight
                    weight_v[idx] += weight

        return 0

    def __add__(self, GroupedBootstrapMeanAccumulator other) -> GroupedBootstrapMeanAccumulator:
        cdef:
            size_t i, n_total
            size_t min_cap = self.capacity if self.capacity < other.capacity else other.capacity
            size_t max_cap = self.capacity if self.capacity > other.capacity else other.capacity

            GroupedBootstrapMeanAccumulator r = GroupedBootstrapMeanAccumulator(max_cap // self.n_boot)
            double *result_stat_v = r.get_stat_v()
            double *self_stat_v = self.get_stat_v()
            double *other_stat_v = other.get_stat_v()

            double *source_stat_v
            size_t *source_count_v

        if self.n_boot != other.n_boot:
            raise ValueError("Bootstrap dimensions must match.")

        for i in range(min_cap):
            result_stat_v[i] = self_stat_v[i] + other_stat_v[i]
            r.count_v[i] = self.count_v[i] + other.count_v[i]

        if self.capacity > other.capacity:
            source_stat_v = self_stat_v
            source_count_v = self.count_v
        else:
            source_stat_v = other_stat_v
            source_count_v = other.count_v

        memcpy(result_stat_v + min_cap, source_stat_v + min_cap, (max_cap - min_cap) * sizeof(double))
        memcpy(r.count_v + min_cap, source_count_v + min_cap, (max_cap - min_cap) * sizeof(size_t))

        return r

    cdef inline void merge_bootstraps(self, size_t group_idx, WelfordState* ws_result) noexcept nogil:
        cdef:
            size_t j, boot_idx
            double* stat_v = self.get_stat_v()

        for j in range(self.n_boot):
            boot_idx = group_idx * self.n_boot + j

            if self.count_v[boot_idx] == 0:
                continue

            x = stat_v[boot_idx] / self.count_v[boot_idx]
            ws_add(ws_result, x)

    def to_result(self) -> MeanResult:
        if self.capacity == 0:
            return MeanResult(
                mean=np.array([], dtype=np.float64),
                se=np.array([], dtype=np.float64)
            )

        cdef:
            size_t i
            double *stat_v = self.get_stat_v()
            size_t *count_v = self.count_v

            size_t n_groups = self.capacity // self.n_boot
            double[:] mean_r = np.empty(n_groups, dtype=np.float64)
            double[:] se_r = np.empty(n_groups, dtype=np.float64)

            WelfordState* ws_result = ws_new()

        for i in range(n_groups):
            self.merge_bootstraps(i, ws_result)
            mean_r[i] = ws_mean(ws_result)
            se_r[i] = ws_std(ws_result)
            ws_reset(ws_result)

        free(ws_result)
        return MeanResult(mean=mean_r.base, se=se_r.base)

    def to_filtered_result(self) -> IndexedGroupedStatResult:
        if self.capacity == 0:
            return IndexedGroupedStatResult(
                index=np.array([], dtype=np.uintp),
                result=MeanResult(
                    mean=np.array([], dtype=np.float64)
                )
            )

        cdef:
            size_t i, idx
            double *stat_v = self.get_stat_v()
            size_t *count_v = self.count_v

            size_t[:] indices = self.build_indices()
            size_t n = indices.shape[0]

            double[:] mean_r = np.empty(n, dtype=np.float64)
            double[:] se_r = np.empty(n, dtype=np.float64)

            WelfordState* ws_result = ws_new()

        for i in range(n):
            idx = indices[i]
            self.merge_bootstraps(idx, ws_result)
            mean_r[i] = ws_mean(ws_result)
            se_r[i] = ws_std(ws_result)
            ws_reset(ws_result)

        free(ws_result)

        return IndexedGroupedStatResult(
            indices.base,
            MeanResult(mean=mean_r.base, se=se_r.base)
        )

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free
cimport numpy as cnp
import numpy as np
from libc.math cimport isnan
from pyspatialstats.grouped.indices.max cimport _define_max_ind
from pyspatialstats.grouped.accumulators.base cimport BaseGroupedStatAccumulator, BaseGroupedBootstrapAccumulator
from pyspatialstats.results.stats import IndexedGroupedStatResult
from pyspatialstats.results.stats import RegressionResult
from pyspatialstats.stats.p_values import calculate_p_value
from pyspatialstats.stats.linear_regression cimport (
    LinearRegressionState, lrs_array_free, lrs_add, lrs_merge, lrs_to_result, lrs_array_init, lrs_array_to_bootstrap_result
)
from pyspatialstats.stats.linear_regression cimport LinearRegressionResult, lrr_new, lrr_free


def empty_result(size: int, nf: int, calc_se: bool = True, calc_r2: bool = True):
    return RegressionResult(
        df=np.full(size, fill_value=0, dtype=np.intp),
        beta=np.full((size, nf), fill_value=np.nan, dtype=np.float64),
        beta_se=np.full((size, nf), fill_value=np.nan, dtype=np.float64) if calc_se else None,
        t=np.full((size, nf), fill_value=np.nan, dtype=np.float64) if calc_se else None,
        p=np.full((size, nf), fill_value=np.nan, dtype=np.float64) if calc_se else None,
        r_squared=np.full(size, fill_value=np.nan, dtype=np.float64) if calc_r2 else None
    )


cdef class GroupedLinearRegressionAccumulator(BaseGroupedStatAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.nf = 1
        self.eltsize = sizeof(LinearRegressionState)
        self.resize(capacity)
        self.calc_se = True
        self.calc_r2 = True

    def post_init(self, calc_se: bool = True, calc_r2: bool = True):
        self.calc_se = calc_se
        self.calc_r2 = calc_r2

    cdef int _reset_stat_v(self) except -1 nogil:
        cdef int r = lrs_array_init(self.get_stat_v(), self.capacity, self.nf)
        if r != 0:
            return -1
        return 0

    cdef LinearRegressionState* get_stat_v(self) noexcept nogil:
        return <LinearRegressionState *> self.stat_v

    cdef void _deallocate(self) noexcept nogil:
        if self.count_v != NULL:
            free(self.count_v)
            self.count_v = NULL
        if self.stat_v != NULL:
            lrs_array_free(<LinearRegressionState *> self.stat_v, self.capacity)
            self.stat_v = NULL

    def get_stat_v_py(self):
        cdef:
            size_t i, j, k
            LinearRegressionState *stat_v = self.get_stat_v()

        if self.capacity == 0 or self.nf == 0 or stat_v == NULL:
            return np.full((self.capacity, self.nf, self.nf), fill_value=np.nan, dtype=np.float64)

        return cnp.PyArray_SimpleNewFromData(
            3, [self.capacity, self.nf, self.nf], cnp.NPY_DOUBLE, stat_v.XtX
        ).copy()

    cpdef int add_data(self, size_t[:] ind, double[:, :] x, double[:] y) except -1:
        cdef:
            size_t i, j, max_ind, idx, nf = x.shape[1] + 1
            size_t n_samples = ind.shape[0]
            double xi, xj, yi
            LinearRegressionState* states

        with nogil:
            self.nf = nf
            max_ind = _define_max_ind(ind)
            resize_result = self.resize(max_ind + 1)
            if resize_result != 0:
                return -1
            states = self.get_stat_v()

            for i in range(n_samples):
                yi = y[i]
                if isnan(yi):
                    continue
                for j in range(x.shape[1]):
                    if isnan(x[i, j]):
                        break
                else:
                    idx = ind[i]
                    lrs_add(&states[idx], yi, x[i])
                    self.count_v[idx] += 1

        return 0

    def __add__(self, GroupedLinearRegressionAccumulator other) -> GroupedLinearRegressionAccumulator:
        if self.nf != other.nf:
            raise ValueError("nf must match in both accumulators")

        cdef:
            size_t i
            size_t nf = self.nf
            size_t capacity = self.capacity if self.capacity > other.capacity else other.capacity
            GroupedLinearRegressionAccumulator r = GroupedLinearRegressionAccumulator()
            size_t self_count, other_count, n

            LinearRegressionState *result_stat_v
            LinearRegressionState *self_stat_v = self.get_stat_v()
            LinearRegressionState *other_stat_v = other.get_stat_v()

        r.nf = self.nf
        r.resize(capacity)
        result_stat_v = r.get_stat_v()

        for i in range(capacity):
            self_count = self.count_v[i]
            other_count = other.count_v[i]

            r.count_v[i] = self_count + other_count

            if self_count > 0:
                lrs_merge(&result_stat_v[i], &self_stat_v[i])
            if other_count > 0:
                lrs_merge(&result_stat_v[i], &other_stat_v[i])

        return r

    def to_result(self):
        if self.capacity == 0 or self.nf == 0 or self.stat_v == NULL or self.count_v == NULL:
            return empty_result(0, self.nf, self.calc_se, self.calc_r2)

        cdef:
            size_t i
            LinearRegressionState* stat_v = self.get_stat_v()
            LinearRegressionResult* lrr = lrr_new(self.nf)
            double[:] df = np.full(self.capacity, fill_value=np.nan, dtype=np.float64)
            double[:, :] beta = np.full((self.capacity, self.nf), fill_value=np.nan, dtype=np.float64)
            double[:, :] beta_se
            double[:] r_squared

        if self.calc_se:
            beta_se = np.full((self.capacity, self.nf), fill_value=np.nan, dtype=np.float64)
        if self.calc_r2:
            r_squared = np.full(self.capacity, fill_value=np.nan, dtype=np.float64)

        for i in range(self.capacity):
            lrs_to_result(&stat_v[i], lrr, self.calc_se, self.calc_r2)

            if lrr.status > 0:
                df[i] = -lrr.status
                continue

            df[i] = lrr.df
            if self.calc_r2:
                r_squared[i] = lrr.r_squared
            for j in range(self.nf):
                beta[i, j] = lrr.beta[j]
            if self.calc_se:
                for j in range(self.nf):
                    beta_se[i, j] = lrr.beta_se[j]

        lrr_free(lrr)

        if self.calc_se:
            np_t = beta.base / beta_se.base
            p = calculate_p_value(np_t, df.base[:, np.newaxis])
        else:
            np_t = None
            p = None

        return RegressionResult(
            df=df.base,
            beta=beta.base,
            beta_se=beta_se.base if self.calc_se else None,
            r_squared=r_squared.base if self.calc_r2 else None,
            t=np_t,
            p=p
        )

    def to_filtered_result(self):
        cdef:
            LinearRegressionState* stat_v = self.get_stat_v()
            LinearRegressionResult* lrr = lrr_new(nf=self.nf)
            size_t[:] indices = self.build_indices()
            size_t num_inds = indices.shape[0]

        if num_inds == 0 or self.nf == 0:
            return IndexedGroupedStatResult(index=indices.base, result=empty_result(num_inds, self.nf, self.calc_se, self.calc_r2))

        cdef:
            double[:] df = np.full(num_inds, fill_value=np.nan, dtype=np.float64)
            double[:, :] beta = np.full((num_inds, self.nf), fill_value=np.nan, dtype=np.float64)
            double[:, :] beta_se
            double[:] r_squared
            size_t i, idx

        if self.calc_se:
            beta_se = np.full((num_inds, self.nf), fill_value=np.nan, dtype=np.float64)
        if self.calc_r2:
            r_squared = np.full(num_inds, fill_value=np.nan, dtype=np.float64)

        for i in range(num_inds):
            idx = indices[i]
            lrs_to_result(&stat_v[idx], lrr, self.calc_se, self.calc_r2)

            if lrr.status > 0:
                df[i] = -lrr.status
                continue

            df[i] = lrr.df

            if self.calc_r2:
                r_squared[i] = lrr.r_squared
            for j in range(self.nf):
                beta[i, j] = lrr.beta[j]
            if self.calc_se:
                for j in range(self.nf):
                    beta_se[i, j] = lrr.beta_se[j]

        lrr_free(lrr)

        if self.calc_se:
            np_t = beta.base / beta_se.base
            p = calculate_p_value(np_t, df.base[:, np.newaxis])
        else:
            np_t = None
            p = None

        return IndexedGroupedStatResult(
            index=indices.base,
            result=RegressionResult(
                df=df.base,
                beta=beta.base,
                beta_se=beta_se.base if self.calc_se else None,
                r_squared=r_squared.base if self.calc_r2 else None,
                t=np_t,
                p=p
            )
        )


cdef class GroupedBootstrapLinearRegressionAccumulator(BaseGroupedBootstrapAccumulator):
    def __cinit__(self, capacity: int = 0):
        self.nf = 1
        self.eltsize = sizeof(LinearRegressionState)
        self.bootstrap_resize(capacity)

    cdef int _reset_stat_v(self) except -1 nogil:
        cdef int r = lrs_array_init(self.get_stat_v(), self.capacity, self.nf)
        if r != 0:
            return -1
        return r

    cdef LinearRegressionState* get_stat_v(self) noexcept nogil:
        return <LinearRegressionState *> self.stat_v

    cdef void _deallocate(self) noexcept nogil:
        if self.count_v != NULL:
            free(self.count_v)
            self.count_v = NULL
        if self.stat_v != NULL:
            lrs_array_free(<LinearRegressionState *> self.stat_v, self.capacity)
            self.stat_v = NULL

    def get_stat_v_py(self):
        cdef:
            size_t i, j, k
            LinearRegressionState *stat_v = self.get_stat_v()

        if self.capacity == 0 or self.nf == 0 or stat_v == NULL:
            return np.full((self.capacity, self.nf, self.nf), fill_value=np.nan, dtype=np.float64)

        return cnp.PyArray_SimpleNewFromData(
            3, [self.capacity, self.nf, self.nf], cnp.NPY_DOUBLE, stat_v.XtX
        ).copy()

    cpdef int add_data(self, size_t[:] ind, double[:, :] x, double[:] y) except -1:
        cdef:
            size_t i, j, k, max_ind, boot_idx, idx, n_samples = y.shape[0]
            size_t nf = x.shape[1] + 1
            LinearRegressionState* lrs_array
            long weight
            double yi
            int resize_result

        with nogil:
            self.nf = nf
            max_ind = _define_max_ind(ind)
            resize_result = self.bootstrap_resize(max_ind + 1)
            if resize_result != 0:
                return -1

            lrs_array = self.get_stat_v()

            for i in range(n_samples):
                yi = y[i]
                if isnan(yi):
                    continue
                for j in range(x.shape[1]):
                    if isnan(x[i, j]):
                        break

                else:
                    idx = ind[i]

                    for k in range(self.n_boot):
                        weight = self.rng.poisson(1)
                        if weight == 0:
                            continue
                        boot_idx = idx * self.n_boot + k
                        lrs_add(&lrs_array[boot_idx], yi, x[i], weight)
                        self.count_v[boot_idx] += weight

        return 0

    def to_result(self):
        if self.capacity == 0 or self.nf == 0 or self.stat_v == NULL or self.count_v == NULL:
            return empty_result(0, self.nf)

        cdef:
            size_t i
            size_t n_groups = self.capacity // self.n_boot
            LinearRegressionState* lrs_array = self.get_stat_v()
            LinearRegressionResult* lrr = lrr_new(self.nf)

            double[:] df = np.full(n_groups, np.nan, dtype=np.float64)
            double[:, :] beta = np.full((n_groups, self.nf), np.nan, dtype=np.float64)
            double[:, :] beta_se = np.full((n_groups, self.nf), np.nan, dtype=np.float64)
            double[:] r_squared = np.full(n_groups, np.nan, dtype=np.float64)
            double[:] r_squared_se = np.full(n_groups, np.nan, dtype=np.float64)

        for i in range(n_groups):
            lrs_array_to_bootstrap_result(
                &lrs_array[i * self.n_boot],
                lrr,
                self.n_boot,
            )

            if lrr.status > 0:
                df[i] = -lrr.status
                continue

            df[i] = lrr.df
            r_squared[i] = lrr.r_squared
            r_squared_se[i] = lrr.r_squared_se
            for j in range(self.nf):
                beta[i, j] = lrr.beta[j]
                beta_se[i, j] = lrr.beta_se[j]

        lrr_free(lrr)

        np_t = beta.base / beta_se.base

        return RegressionResult(
            df=df.base,
            beta=beta.base,
            beta_se=beta_se.base,
            r_squared=r_squared.base,
            r_squared_se=r_squared_se.base,
            t=np_t,
            p=calculate_p_value(np_t, df.base[:, np.newaxis])
        )

    def to_filtered_result(self):
        if self.capacity == 0 or self.nf == 0 or self.stat_v == NULL or self.count_v == NULL:
            return IndexedGroupedStatResult(
                index=np.array([], dtype=np.uintp),
                result=empty_result(0, self.nf)
            )

        cdef:
            size_t[:] indices = self.build_indices()
            size_t n_groups = indices.shape[0]
            LinearRegressionState* lrs_array = self.get_stat_v()
            LinearRegressionResult* lrr = lrr_new(self.nf)

            double[:] df = np.empty(n_groups, dtype=np.float64)
            double[:, :] beta = np.full((n_groups, self.nf), np.nan, dtype=np.float64)
            double[:, :] beta_se = np.full((n_groups, self.nf), np.nan, dtype=np.float64)
            double[:] r_squared = np.full(n_groups, np.nan, dtype=np.float64)
            double[:] r_squared_se = np.full(n_groups, np.nan, dtype=np.float64)

            size_t i, idx

        for i in range(n_groups):
            idx = indices[i]

            lrs_array_to_bootstrap_result(
                &lrs_array[idx * self.n_boot],
                lrr,
                self.n_boot,
            )
            if lrr.status > 0:
                df[i] = -lrr.status
                continue

            df[i] = lrr.df
            r_squared[i] = lrr.r_squared
            r_squared_se[i] = lrr.r_squared_se
            for j in range(self.nf):
                beta[i, j] = lrr.beta[j]
                beta_se[i, j] = lrr.beta_se[j]

        lrr_free(lrr)

        np_t = beta.base / beta_se.base

        return IndexedGroupedStatResult(
            index=indices.base,
            result=RegressionResult(
                df=df.base,
                beta=beta.base,
                beta_se=beta_se.base,
                r_squared=r_squared.base,
                r_squared_se=r_squared_se.base,
                t=np_t,
                p=calculate_p_value(np_t, df.base[:, np.newaxis])
            )
        )

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from pyspatialstats.stats.p_values import calculate_p_value
from pyspatialstats.types.results import CorrelationResult, LinearRegressionResult, MeanResult

import numpy as np

from libc.stdlib cimport free
cimport numpy as cnp

from pyspatialstats.types.cy_types cimport numeric_return
from pyspatialstats.grouped.core.count cimport _define_max_ind, _grouped_count
from pyspatialstats.grouped.core.mean cimport _grouped_mean
from pyspatialstats.grouped.core.min cimport _grouped_min
from pyspatialstats.grouped.core.max cimport _grouped_max
from pyspatialstats.grouped.core.std cimport _grouped_std as _grouped_std_internal
from pyspatialstats.grouped.core.correlation cimport (
    _grouped_correlation,
    CyGroupedCorrelationResult
)
from pyspatialstats.grouped.core.linear_regression cimport (
    _grouped_linear_regression,
    CyGroupedLinearRegressionResult
)


cdef double* _grouped_std(size_t[:] ind, double[:] v, size_t max_ind) except NULL nogil:
    cdef:
        double *std_v
        double *mean_v

    max_ind = _define_max_ind(ind)
    mean_v = _grouped_mean(ind, v, max_ind)
    std_v = _grouped_std_internal(ind, v, mean_v, max_ind)

    if std_v == NULL:
        free(mean_v)
        with gil:
            raise MemoryError("std_v memory error")

    free(mean_v)
    return std_v


cdef void _apply_to_target(
    size_t[:] ind,
    double[:] v,
    numeric_return[:, ::1] target,
    size_t rows,
    size_t cols,
    numeric_return* (*f)(size_t[:], double[:], size_t) except NULL nogil
):
    cdef:
        size_t i, j, c
        size_t max_ind
        numeric_return* target_v

    with nogil:
        max_ind = _define_max_ind(ind)
        target_v = f(ind, v, max_ind)

        for i in range(rows):
            for j in range(cols):
                c = ind[i * cols + j]
                if c == 0:
                    continue
                target[i, j] = target_v[c]

    free(target_v)


def _strata_count(size_t[:] ind, double[:] v, size_t rows, size_t cols) -> RasterInt32:
    r = np.full((rows, cols), dtype=np.uintp, fill_value=0)
    _apply_to_target[size_t](ind, v, r, rows, cols, _grouped_count)
    return r


def _strata_min(size_t[:] ind, double[:] v, size_t rows, size_t cols) -> RasterFloat64:
    r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
    _apply_to_target[double](ind, v, r, rows, cols, _grouped_min)
    return r
                    

def _strata_max(size_t[:] ind, double[:] v, size_t rows, size_t cols) -> RasterFloat64:
    r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
    _apply_to_target[double](ind, v, r, rows, cols, _grouped_max)
    return r


def _strata_mean(size_t[:] ind, double[:] v, size_t rows, size_t cols) -> RasterFloat64:
    r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
    _apply_to_target[double](ind, v, r, rows, cols, _grouped_mean)
    return r


def _strata_std(size_t[:] ind, double[:] v, size_t rows, size_t cols) -> RasterFloat64:
    r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
    _apply_to_target[double](ind, v, r, rows, cols, _grouped_std)
    return r


def _strata_mean_std(size_t[:] ind, double[:] v, size_t rows, size_t cols) -> MeanResult:
    cdef:
        double[:, ::1] mean_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] std_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        size_t i, j, c, max_ind
        double *std_v, *mean_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            mean_v = _grouped_mean(ind, v, max_ind)
            std_v = _grouped_std_internal(ind, v, mean_v, max_ind)

            for i in range(rows):
                for j in range(cols):
                    c = ind[i * cols + j]
                    if c == 0:
                        continue
                    mean_r[i, j] = mean_v[c]
                    std_r[i, j] = std_v[c]

    finally:
        free(mean_v)
        free(std_v)

    return MeanResult(
        mean=np.asarray(mean_r),
        std=np.asarray(std_r)
    )


def _strata_correlation(size_t[:] ind,
                        double[:] v1,
                        double[:] v2,
                        size_t rows,
                        size_t cols) -> CorrelationResult:
    cdef:
        size_t i, j, c, max_ind
        double[:] t_v
        double[:] p_v
        size_t[:] df_v

        double[:, ::1] c_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] p_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)

        CyGroupedCorrelationResult r

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r = _grouped_correlation(ind, v1, v2, max_ind)

        t_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.t)
        df_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_UINTP, r.df)

        p_v = calculate_p_value(np.asarray(t_v), np.asarray(df_v))

        with nogil:
            for i in range(rows):
                for j in range(cols):
                    c = ind[i * cols + j]
                    if c == 0:
                        continue
                    c_r[i, j] = r.c[c]
                    p_r[i, j] = p_v[c]

    finally:
        free(r.c)
        free(r.t)
        free(r.df)

    return CorrelationResult(c=np.asarray(c_r), p=np.asarray(p_r))


def _strata_linear_regression(size_t[:] ind,
                              double[:] v1,
                              double[:] v2,
                              size_t rows,
                              size_t cols) -> LinearRegressionResult:
    cdef:
        size_t i, j, c, max_ind
        double[:] t_a_v, t_b_v
        double[:] p_a_v, p_b_v
        size_t[:] df_v

        double[:, ::1] a_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] b_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] se_a_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] se_b_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] t_a_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] t_b_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] p_a_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        double[:, ::1] p_b_r = np.full((rows, cols), dtype=np.float64, fill_value=np.nan)
        size_t[:, ::1] df_r = np.full((rows, cols), dtype=np.uintp, fill_value=np.nan)

        CyGroupedLinearRegressionResult r

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r = _grouped_linear_regression(ind, v1, v2, max_ind)

        t_a_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.t_a)
        t_b_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.t_b)
        df_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_UINTP, r.df)

        p_a_v = calculate_p_value(np.asarray(t_a_v), np.asarray(df_v))
        p_b_v = calculate_p_value(np.asarray(t_b_v), np.asarray(df_v))

        with nogil:
            for i in range(rows):
                for j in range(cols):
                    c = ind[i * cols + j]
                    if c == 0:
                        continue

                    df_r[i, j] = r.df[c]
                    a_r[i, j] = r.a[c]
                    b_r[i, j] = r.b[c]
                    se_a_r[i, j] = r.se_a[c]
                    se_b_r[i, j] = r.se_b[c]
                    t_a_r[i, j] = r.t_a[c]
                    t_b_r[i, j] = r.t_b[c]
                    p_a_r[i, j] = p_a_v[c]
                    p_b_r[i, j] = p_b_v[c]

    finally:
        free(r.a)
        free(r.b)
        free(r.se_a)
        free(r.se_b)
        free(r.t_a)
        free(r.t_b)
        free(r.df)

    return LinearRegressionResult(
        df=np.asarray(df_r),
        a=np.asarray(a_r),
        b=np.asarray(b_r),
        se_a=np.asarray(se_a_r),
        se_b=np.asarray(se_b_r),
        t_a=np.asarray(t_a_r),
        t_b=np.asarray(t_b_r),
        p_a=np.asarray(p_a_r),
        p_b=np.asarray(p_b_r)
    )

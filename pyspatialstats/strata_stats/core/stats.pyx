# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from pyspatialstats.stat_utils import calculate_p_value
from pyspatialstats.results import StrataLinearRegressionResult, StrataCorrelationResult, StrataStatsResult

import numpy as np

from libc.stdlib cimport free
cimport numpy as cnp

from pyspatialstats.grouped_stats.core.count cimport _define_max_ind, _grouped_count
from pyspatialstats.grouped_stats.core.mean cimport _grouped_mean
from pyspatialstats.grouped_stats.core.min cimport _grouped_min
from pyspatialstats.grouped_stats.core.max cimport _grouped_max
from pyspatialstats.grouped_stats.core.std cimport _grouped_std
from pyspatialstats.grouped_stats.core.correlation cimport (
    _grouped_correlation,
    CyGroupedCorrelationResult
)
from pyspatialstats.grouped_stats.core.linear_regression cimport (
    _grouped_linear_regression,
    CyGroupedLinearRegressionResult
)


cdef int _apply_to_target(size_t[:] ind,
                          float[:] v,
                          float[:, ::1] target,
                          long rows,
                          long cols,
                          float* (*f)(size_t[:], float[:], size_t) nogil) nogil:
    cdef:
        size_t i, j, c
        size_t max_ind
        float *target_v

    max_ind = _define_max_ind(ind)
    target_v = f(ind, v, max_ind)

    for i in range(rows):
        for j in range(cols):
            c = ind[i * cols + j]
            if c == 0:
                continue
            target[i, j] = target_v[c]

    free(target_v)


cdef float[:, ::1] _apply_values_to_raster_float32(size_t[:] ind,
                                                   float[:] v,
                                                   long rows,
                                                   long cols,
                                                   float* (*f)(size_t[:], float[:], size_t) except * nogil):
    cdef:
        float[:, ::1] r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)

    with nogil:
        _apply_to_target(ind, v, r, rows=rows, cols=cols, f=f)

    return r


def _strata_count(size_t[:] ind, float[:] v, size_t rows, size_t cols) -> RasterInt32:
    cdef:
        int[:, ::1] count_r = np.full((rows, cols), dtype=np.int32, fill_value=np.nan)
        size_t i, j, c, max_ind
        long *count_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            count_v = _grouped_count(ind, v, max_ind)
        
            for i in range(rows):
                for j in range(cols):
                    c = ind[i * cols + j]
                    if c == 0:
                        continue
                    count_r[i, j] = count_v[c]
        
    finally:
        free(count_v)

    return np.asarray(count_r)


def _strata_min(size_t[:] ind, float[:] v, size_t rows, size_t cols) -> RasterFloat32:
    return np.asarray(
        _apply_values_to_raster_float32(ind, v, rows, cols, _grouped_min)
    )
                    

def _strata_max(size_t[:] ind, float[:] v, size_t rows, size_t cols) -> RasterFloat32:
    return np.asarray(
        _apply_values_to_raster_float32(ind, v, rows, cols, _grouped_max)
    )


def _strata_mean(size_t[:] ind, float[:] v, size_t rows, size_t cols) -> RasterFloat32:
    return np.asarray(
        _apply_values_to_raster_float32(ind, v, rows, cols, _grouped_mean)
    )


def _strata_std(size_t[:] ind, float[:] v, size_t rows, size_t cols) -> RasterFloat32:
    cdef:
        float[:, ::1] std_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        size_t i, j, c, max_ind
        float *std_v, *mean_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            mean_v = _grouped_mean(ind, v, max_ind)
            std_v = _grouped_std(ind, v, mean_v, max_ind)

            for i in range(rows):
                for j in range(cols):
                    c = ind[i * cols + j]
                    if c == 0:
                        continue
                    std_r[i, j] = std_v[c]

    finally:
        free(mean_v)
        free(std_v)

    return np.asarray(std_r)


def _strata_mean_std(size_t[:] ind, float[:] v, size_t rows, size_t cols) -> Tuple[RasterFloat32, RasterFloat32]:
    cdef:
        float[:, ::1] mean_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] std_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        size_t i, j, c, max_ind
        float *std_v, *mean_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            mean_v = _grouped_mean(ind, v, max_ind)
            std_v = _grouped_std(ind, v, mean_v, max_ind)

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

    return StrataStatsResult(
        mean=np.asarray(mean_r),
        std=np.asarray(std_r)
    )


def _strata_correlation(size_t[:] ind,
                        float[:] v1,
                        float[:] v2,
                        size_t rows,
                        size_t cols) -> StrataCorrelationResult:
    cdef:
        size_t i, j, c, max_ind
        float[:] t_v
        double[:] p_v
        long[:] df_v

        float[:, ::1] c_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] p_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)

        CyGroupedCorrelationResult r

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r = _grouped_correlation(ind, v1, v2, max_ind)

        t_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT, r.t)
        df_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_LONG, r.df)

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

    return StrataCorrelationResult(c=np.asarray(c_r), p=np.asarray(p_r))


def _strata_linear_regression(size_t[:] ind,
                              float[:] v1,
                              float[:] v2,
                              size_t rows,
                              size_t cols) -> StrataLinearRegressionResult:
    cdef:
        size_t i, j, c, max_ind
        float[:] t_a_v, t_b_v
        double[:] p_a_v, p_b_v
        long[:] df_v

        float[:, ::1] a_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] b_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] se_a_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] se_b_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] t_a_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] t_b_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] p_a_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        float[:, ::1] p_b_r = np.full((rows, cols), dtype=np.float32, fill_value=np.nan)
        long[:, ::1] df_r = np.full((rows, cols), dtype=np.int64, fill_value=np.nan)

        CyGroupedLinearRegressionResult r

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r = _grouped_linear_regression(ind, v1, v2, max_ind)

        t_a_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT, r.t_a)
        t_b_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT, r.t_b)
        df_v = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_LONG, r.df)

        p_a_v = calculate_p_value(np.asarray(t_a_v), np.asarray(df_v))
        p_b_v = calculate_p_value(np.asarray(t_b_v), np.asarray(df_v))

        with nogil:
            for i in range(rows):
                for j in range(cols):
                    c = ind[i * cols + j]
                    if c == 0:
                        continue
                    a_r[i, j] = r.a[c]
                    b_r[i, j] = r.b[c]
                    se_a_r[i, j] = r.se_a[c]
                    se_b_r[i, j] = r.se_b[c]
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

    return StrataLinearRegressionResult(
        a=np.asarray(a_r),
        b=np.asarray(b_r),
        se_a=np.asarray(se_a_r),
        se_b=np.asarray(se_b_r),
        p_a=np.asarray(p_a_r),
        p_b=np.asarray(p_b_r)
    )

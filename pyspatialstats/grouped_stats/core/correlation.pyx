# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from pyspatialstats.stat_utils import calculate_p_value
from pyspatialstats.results import GroupedCorrelationResult

cimport numpy as cnp
from libc.math cimport sqrt, isnan, NAN
from libc.stdlib cimport calloc, free, malloc

from pyspatialstats.grouped_stats.core.count cimport _define_max_ind, _grouped_count


cdef CyGroupedCorrelationResult _grouped_correlation(size_t[:] ind, float[:] v1, float[:] v2, size_t max_ind) except * nogil: 
    cdef:
        size_t i, k, n = ind.shape[0]
        long *count = <long *> calloc(max_ind + 1, sizeof(long))
        float *sum_v1 = <float *> calloc(max_ind + 1, sizeof(float))
        float *sum_v2 = <float *> calloc(max_ind + 1, sizeof(float))
        float *sum_v1_v2 = <float *> calloc(max_ind + 1, sizeof(float))
        float *sum_v1_squared = <float *> calloc(max_ind + 1, sizeof(float))
        float *sum_v2_squared = <float *> calloc(max_ind + 1, sizeof(float))
        float *num = <float *> calloc(max_ind + 1, sizeof(float))
        float *den = <float *> calloc(max_ind + 1, sizeof(float))
        CyGroupedCorrelationResult result

    result.c = <float *> calloc(max_ind + 1, sizeof(float))
    result.t = <float *> calloc(max_ind + 1, sizeof(float))
    result.df = <long *> calloc(max_ind + 1, sizeof(long))

    if (
        count == NULL or
        sum_v1 == NULL or
        sum_v2 == NULL or
        sum_v1_v2 == NULL or
        sum_v1_squared == NULL or
        sum_v2_squared == NULL or
        num == NULL or
        den == NULL or
        result.c == NULL or
        result.t == NULL or
        result.df == NULL
    ):
        free(count)
        free(sum_v1)
        free(sum_v2)
        free(sum_v1_v2)
        free(sum_v1_squared)
        free(sum_v2_squared)
        free(num)
        free(den)
        free(result.c)
        free(result.t)
        free(result.df)

        with gil:
            raise MemoryError("Memory error")

    for i in range(n):
        if ind[i] == 0:
            continue
        if isnan(v1[i]) or isnan(v2[i]):
            continue

        count[ind[i]] += 1
        sum_v1[ind[i]] += v1[i]
        sum_v2[ind[i]] += v2[i]
        sum_v1_v2[ind[i]] += v1[i] * v2[i]
        sum_v1_squared[ind[i]] += v1[i] * v1[i]
        sum_v2_squared[ind[i]] += v2[i] * v2[i]

    for k in range(max_ind + 1):
        if count[k] <= 2:
            result.c[k] = NAN
            result.df[k] = 0
            continue

        result.df[k] = count[k] - 2

        # Calculate Pearson correlation coefficient
        num[k] = count[k] * sum_v1_v2[k] - sum_v1[k] * sum_v2[k]
        den[k] = (
            sqrt(
                (count[k] * sum_v1_squared[k] - sum_v1[k] * sum_v1[k]) *
                (count[k] * sum_v2_squared[k] - sum_v2[k] * sum_v2[k])
            )
        )

        if den[k] == 0:
            result.c[k] = 0
        else:
            result.c[k] = num[k] / den[k]

        # Calculate t-statistic
        result.t[k] = result.c[k] * sqrt(result.df[k] / (1 - result.c[k] ** 2))

    free(count)
    free(sum_v1)
    free(sum_v2)
    free(sum_v1_v2)
    free(sum_v1_squared)
    free(sum_v2_squared)
    free(num)
    free(den)

    return result


def grouped_correlation_npy(size_t[:] ind, float[:] v1, float[:] v2) -> GroupedCorrelationResult:
    cdef:
        CyGroupedCorrelationResult r
        size_t max_ind

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r = _grouped_correlation(ind, v1, v2, max_ind)

        corr_arr = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT, r.c)
        cnp.PyArray_ENABLEFLAGS(corr_arr, cnp.NPY_ARRAY_OWNDATA)

        t_arr = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT, r.t)
        df_arr = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_LONG, r.df)

        py_r = GroupedCorrelationResult(corr_arr, calculate_p_value(t_arr, df_arr))

    finally:
        free(r.t)
        free(r.df)

    return py_r


def grouped_correlation_npy_filtered(size_t[:] ind, float[:] v1, float[:] v2) -> GroupedCorrelationResult:
    cdef:
        CyGroupedCorrelationResult r
        size_t i, max_ind, c = 0, num_inds = 0
        long *count_v1, *count_v2, *df_f
        float *c_f, *t_f

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            count_v1 = _grouped_count(ind, v1, max_ind)
            count_v2 = _grouped_count(ind, v2, max_ind)
            r = _grouped_correlation(ind, v1, v2, max_ind)

            for i in range(max_ind + 1):
                if count_v1[i] > 0 and count_v2[i] > 0:
                    num_inds += 1

            c_f = <float *> malloc(num_inds * sizeof(float))
            t_f = <float *> malloc(num_inds * sizeof(float))
            df_f = <long *> malloc(num_inds * sizeof(long))

            if c_f == NULL or t_f == NULL or df_f == NULL:
                with gil:
                    raise MemoryError

            for i in range(max_ind + 1):
                if count_v1[i] > 0 and count_v2[i] > 0:
                    c_f[c] = r.c[i]
                    t_f[c] = r.t[i]
                    df_f[c] = r.df[i]
                    c += 1

        corr_arr = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_FLOAT, c_f)
        cnp.PyArray_ENABLEFLAGS(corr_arr, cnp.NPY_ARRAY_OWNDATA)

        t_arr = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_FLOAT, t_f)
        df_arr = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_LONG, df_f)

        py_r = GroupedCorrelationResult(corr_arr, calculate_p_value(t_arr, df_arr))

    except MemoryError:
        free(c_f)

    finally:
        free(r.c)
        free(r.t)
        free(r.df)
        free(count_v1)
        free(count_v2)
        free(t_f)
        free(df_f)

    return py_r

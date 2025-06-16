# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from pyspatialstats.bootstrap.p_values import calculate_p_value
from pyspatialstats.types.results import LinearRegressionResult

from libc.math cimport sqrt, isnan, NAN, pow
from libc.stdlib cimport malloc, calloc, free
cimport numpy as cnp

from pyspatialstats.grouped.core.count cimport _define_max_ind, _grouped_count
from pyspatialstats.types.cy_types cimport CyGroupedLinearRegressionResult


cdef CyGroupedLinearRegressionResult _grouped_linear_regression(size_t[:] ind,
                                                                double[:] v1,
                                                                double[:] v2,
                                                                size_t max_ind) except * nogil:
    cdef:
        size_t i, k, n = ind.shape[0]
        double residual
        size_t *count = <size_t *> calloc(max_ind + 1, sizeof(size_t))
        double *sum_v1 = <double *> calloc(max_ind + 1, sizeof(double))
        double *sum_v2 = <double *> calloc(max_ind + 1, sizeof(double))
        double *sum_v1_v2 = <double *> calloc(max_ind + 1, sizeof(double))
        double *sum_v1_squared = <double *> calloc(max_ind + 1, sizeof(double))
        double *sum_residuals_squared = <double *> calloc(max_ind + 1, sizeof(double))
        double *se = <double *> calloc(max_ind + 1, sizeof(double))
        double *ss_v1_residuals = <double *> calloc(max_ind + 1, sizeof(double))
        CyGroupedLinearRegressionResult result

    result.a = <double *> calloc(max_ind + 1, sizeof(double))
    result.b = <double *> calloc(max_ind + 1, sizeof(double))
    result.se_a = <double *> calloc(max_ind + 1, sizeof(double))
    result.se_b = <double *> calloc(max_ind + 1, sizeof(double))
    result.t_a = <double *> calloc(max_ind + 1, sizeof(double))
    result.t_b = <double *> calloc(max_ind + 1, sizeof(double))
    result.df = <size_t *> calloc(max_ind + 1, sizeof(size_t))

    result.a[0] = result.b[0] = result.se_a[0] = result.se_b[0] = result.t_a[0] = result.t_b[0] = NAN

    if (
        count == NULL or
        sum_v1 == NULL or
        sum_v2 == NULL or
        sum_v1_v2 == NULL or
        sum_v1_squared == NULL or
        sum_residuals_squared == NULL or
        se == NULL or
        ss_v1_residuals == NULL or

        result.df == NULL or
        result.a == NULL or
        result.b == NULL or
        result.se_a == NULL or
        result.se_b == NULL or
        result.t_a == NULL or
        result.t_b == NULL
    ):
        free(count)
        free(sum_v1)
        free(sum_v2)
        free(sum_v1_v2)
        free(sum_v1_squared)
        free(sum_residuals_squared)
        free(se)
        free(ss_v1_residuals)

        free(result.df)
        free(result.a)
        free(result.b)
        free(result.se_a)
        free(result.se_b)
        free(result.t_a)
        free(result.t_b)

        with gil:
            raise MemoryError("Memory error")

    for i in range(n):
        if isnan(v1[i]) or isnan(v2[i]):
            continue

        count[ind[i]] += 1
        sum_v1[ind[i]] += v1[i]
        sum_v2[ind[i]] += v2[i]
        sum_v1_v2[ind[i]] += v1[i] * v2[i]
        sum_v1_squared[ind[i]] += v1[i] * v1[i]

    for k in range(max_ind + 1):
        if count[k] < 2:
            result.a[k] = result.b[k] = result.se_a[k] = result.se_b[k] = result.t_a[k] = result.t_b[k] = NAN
            continue

        result.df[k] = count[k] - 2

        # Calculate coefficients (a and b)
        result.a[k] = (
               (count[k] * sum_v1_v2[k] - sum_v1[k] * sum_v2[k]) /
               (count[k] * sum_v1_squared[k] - pow(sum_v1[k], 2))
        )
        result.b[k] = (sum_v2[k] - result.a[k] * sum_v1[k]) / count[k]

    for i in range(n):
        if isnan(v1[i]) or isnan(v2[i]):
            continue

        residual = v2[i] - (result.a[ind[i]] * v1[i] + result.b[ind[i]])
        sum_residuals_squared[ind[i]] += pow(residual, 2)

    for k in range(max_ind + 1):
        if count[k] < 2:
            result.se_a[k] = result.se_b[k] = result.t_a[k] = result.t_b[k] = NAN
            continue

        se[k] = sqrt(sum_residuals_squared[k] / (count[k] - 2))
        ss_v1_residuals[k] = sum_v1_squared[k] - (sum_v1[k] ** 2) / count[k]

        if ss_v1_residuals[k] == 0:
            result.se_a[k] = result.se_b[k] = result.t_a[k] = result.t_b[k] = NAN
            continue

        # Calculate standard errors
        result.se_a[k] = se[k] / sqrt(ss_v1_residuals[k])
        result.se_b[k] = se[k] * sqrt((1.0 / count[k]) + (pow(sum_v1[k] / count[k], 2) / ss_v1_residuals[k]))

        # T-values
        result.t_a[k] = result.a[k] / result.se_a[k]
        result.t_b[k] = result.b[k] / result.se_b[k]

    free(count)
    free(sum_v1)
    free(sum_v2)
    free(sum_v1_v2)
    free(sum_v1_squared)
    free(sum_residuals_squared)
    free(se)
    free(ss_v1_residuals)

    return result


def grouped_linear_regression_npy(size_t[:] ind, double[:] v1, double[:] v2) -> GroupedLinearRegressionResult:
    cdef:
        CyGroupedLinearRegressionResult r
        size_t max_ind

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r = _grouped_linear_regression(ind, v1, v2, max_ind)

        a = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.a)
        b = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.b)
        se_a = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.se_a)
        se_b = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.se_b)
        t_a = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.t_a)
        t_b = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, r.t_b)
        df = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_UINTP, r.df)

        for x in (a, b, se_a,  se_b, t_a, t_b):
            cnp.PyArray_ENABLEFLAGS(x, cnp.NPY_ARRAY_OWNDATA)

        p_a = calculate_p_value(t_a, df)
        p_b = calculate_p_value(t_b, df)

    finally:
        free(r.df)

    return LinearRegressionResult(a, b, se_a, se_b, t_a, t_b, p_a, p_b)


def grouped_linear_regression_npy_filtered(size_t[:] ind, double[:] v1, double[:] v2) -> GroupedLinearRegressionResult:
    cdef:
        CyGroupedLinearRegressionResult r
        size_t i, max_ind, c = 0, num_inds = 0
        size_t *count_v1, *count_v2, *df_f
        double *a_f, *b_f, *se_a_f, *se_b_f, *t_a_f, *t_b_f

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            count_v1 = _grouped_count(ind, v1, max_ind)
            count_v2 = _grouped_count(ind, v2, max_ind)
            r = _grouped_linear_regression(ind, v1, v2, max_ind)

            for i in range(max_ind + 1):
                if count_v1[i] > 0 and count_v2[i] > 0:
                    num_inds += 1

            a_f = <double *> malloc(num_inds * sizeof(double))
            b_f = <double *> malloc(num_inds * sizeof(double))
            se_a_f = <double *> malloc(num_inds * sizeof(double))
            se_b_f = <double *> malloc(num_inds * sizeof(double))
            t_a_f = <double *> malloc(num_inds * sizeof(double))
            t_b_f = <double *> malloc(num_inds * sizeof(double))
            df_f = <size_t *> malloc(num_inds * sizeof(size_t))

            if (
                a_f == NULL or
                b_f == NULL or
                se_a_f == NULL or
                se_b_f == NULL or
                t_a_f == NULL or
                t_b_f == NULL or
                df_f == NULL
            ):
                with gil:
                    raise MemoryError("rf_v memory error")

            for i in range(max_ind + 1):
                if count_v1[i] > 0 and count_v2[i] > 0:
                    a_f[c] = r.a[i]
                    b_f[c] = r.b[i]
                    se_a_f[c] = r.se_a[i]
                    se_b_f[c] = r.se_b[i]
                    t_a_f[c] = r.t_a[i]
                    t_b_f[c] = r.t_b[i]
                    df_f[c] = r.df[i]
                    c += 1

        a = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, a_f)
        b = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, b_f)
        se_a = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, se_a_f)
        se_b = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, se_b_f)
        t_a = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, t_a_f)
        t_b = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, t_b_f)
        df = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_UINTP, df_f)

        for x in (a, b, se_a,  se_b, t_a, t_b):
            cnp.PyArray_ENABLEFLAGS(x, cnp.NPY_ARRAY_OWNDATA)

        p_a = calculate_p_value(t_a, df)
        p_b = calculate_p_value(t_b, df)

    except MemoryError:
        free(a_f)
        free(b_f)
        free(se_a_f)
        free(se_b_f)
        free(t_a_f)
        free(t_b_f)
        free(df_f)

    finally:
        free(r.df)
        free(r.a)
        free(r.b)
        free(r.se_a)
        free(r.se_b)
        free(r.t_a)
        free(r.t_b)
        free(count_v1)
        free(count_v2)

    return LinearRegressionResult(
        a=a, b=b, se_a=se_a, se_b=se_b, t_a=t_a, t_b=t_b, p_a=p_a, p_b=p_b
    )

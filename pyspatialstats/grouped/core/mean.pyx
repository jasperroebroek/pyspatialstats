# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport isnan
from libc.stdlib cimport calloc, free, malloc
from numpy.math cimport NAN
cimport numpy as cnp

from pyspatialstats.grouped.core.count cimport _define_max_ind, _grouped_count
from pyspatialstats.random.random cimport RandomInts
from pyspatialstats.bootstrap.mean cimport _bootstrap_mean, CyBootstrapMeanResult
from pyspatialstats.types.results import MeanResult


cdef double* _grouped_mean(size_t[:] ind, double[:] v, size_t max_ind) except NULL nogil:
    cdef:
        size_t i, k, n = ind.shape[0]
        size_t *count_v = <size_t *> calloc(max_ind + 1, sizeof(size_t))
        double *sum_v = <double *> calloc(max_ind + 1, sizeof(double))
        double *mean_v = sum_v

    if count_v == NULL or sum_v == NULL:
        free(count_v)
        free(sum_v)
        with gil:
            raise MemoryError("count_v or sum_v memory error")

    for i in range(n):
        if isnan(v[i]):
            continue
        sum_v[ind[i]] += v[i]
        count_v[ind[i]] += 1

    for k in range(max_ind + 1):
        if count_v[k] == 0:
            mean_v[k] = NAN
        else:
            mean_v[k] /= count_v[k]

    free(count_v)

    return mean_v


cdef CyGroupedBootstrapMeanResult _grouped_mean_bootstrap(
    size_t[:] ind,
    double[:] v,
    size_t max_ind,
    size_t n_bootstraps,
    RandomInts rng
) nogil:
    cdef:
        size_t i, c, k, max_count = 0, n = ind.shape[0]
        size_t *count_v = <size_t *> calloc(max_ind + 1, sizeof(size_t))
        double *mean_v = <double *> malloc((max_ind + 1) * sizeof(double))
        double *se_v = <double *> malloc((max_ind + 1) * sizeof(double))
        double *means_v = <double *> malloc(n_bootstraps * sizeof(double))
        double *values
        CyBootstrapMeanResult r
        CyGroupedBootstrapMeanResult result

    if count_v == NULL or mean_v == NULL or se_v == NULL or means_v == NULL:
        free(count_v)
        free(mean_v)
        free(se_v)
        free(means_v)
        raise MemoryError

    result.mean_v = mean_v
    result.se_v = se_v

    for i in range(n):
        if isnan(v[i]):
            continue
        count_v[ind[i]] += 1

    for i in range(max_ind + 1):
        if count_v[i] > 0:
            max_count = count_v[i] if count_v[i] > max_count else max_count

    values = <double *> malloc(max_count * sizeof(double))

    if values == NULL:
        free(count_v)
        free(means_v)
        free(mean_v)
        free(se_v)
        raise MemoryError

    for i in range(max_ind + 1):
        if count_v[i] == 0:
            mean_v[i] = NAN
            se_v[i] = NAN
            continue

        c = 0
        for k in range(n):
            if ind[k] == i:
                values[c] = v[k]
                c += 1
        r = _bootstrap_mean(values, c, n_bootstraps, rng, means_v)
        mean_v[i] = r.mean
        se_v[i] = r.se

    free(count_v)
    free(means_v)
    free(values)

    return result


def grouped_mean_npy(size_t[:] ind, double[:] v) -> np.ndarray:
    cdef:
        size_t max_ind
        double *r

    with nogil:
        max_ind = _define_max_ind(ind)
        r = _grouped_mean(ind, v, max_ind)

    result_array = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE,  r)
    cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    return result_array


def grouped_mean_bootstrap_npy(size_t[:] ind, double[:] v, size_t n_bootstraps, size_t seed) -> np.ndarray:
    cdef:
        size_t max_ind
        RandomInts rng = RandomInts(seed)
        CyGroupedBootstrapMeanResult result
        
    with nogil:
        max_ind = _define_max_ind(ind)
        result = _grouped_mean_bootstrap(ind, v, max_ind, n_bootstraps, rng)

    mean_array = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, <void *> result.mean_v)
    cnp.PyArray_ENABLEFLAGS(mean_array, cnp.NPY_ARRAY_OWNDATA)

    se_array = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_DOUBLE, <void *> result.se_v)
    cnp.PyArray_ENABLEFLAGS(se_array, cnp.NPY_ARRAY_OWNDATA)

    return MeanResult(mean=mean_array, se=se_array)


def grouped_mean_npy_filtered(size_t[:] ind, double[:] v) -> np.ndarray[tuple[int], np.float64]:
    cdef:
        size_t i, max_ind, c = 0, num_inds = 0
        size_t *count_v
        double *r_v
        double *rf_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            r_v = _grouped_mean(ind, v, max_ind)
            count_v = _grouped_count(ind, v, max_ind)

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    num_inds += 1

            rf_v = <double *> malloc(num_inds * sizeof(double))

            if rf_v == NULL:
                with gil:
                    raise MemoryError("rf_v memory error")

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    rf_v[c] = r_v[i]
                    c += 1

        result_array = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, rf_v)
        cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    finally:
        free(r_v)
        free(count_v)

    return result_array


def grouped_mean_bootstrap_npy_filtered(size_t[:] ind, double[:] v, n_bootstraps: int, seed: int) -> np.ndarray[tuple[int], np.float64]:
    cdef:
        size_t i, max_ind, c = 0, num_inds = 0
        size_t *count_v
        double *r_mean_v
        double *r_se_v
        RandomInts rng = RandomInts(seed)
        CyGroupedBootstrapMeanResult result
        size_t cy_n_bootstraps = n_bootstraps

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            result = _grouped_mean_bootstrap(ind, v, max_ind, cy_n_bootstraps, rng)
            count_v = _grouped_count(ind, v, max_ind)

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    num_inds += 1

            r_mean_v = <double *> malloc(num_inds * sizeof(double))
            r_se_v = <double *> malloc(num_inds * sizeof(double))

            if r_mean_v == NULL or r_se_v == NULL:
                with gil:
                    raise MemoryError

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    r_mean_v[c] = result.mean_v[i]
                    r_se_v[c] = result.se_v[i]
                    c += 1

        mean_array = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, r_mean_v)
        cnp.PyArray_ENABLEFLAGS(mean_array, cnp.NPY_ARRAY_OWNDATA)

        se_array = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_DOUBLE, r_se_v)
        cnp.PyArray_ENABLEFLAGS(se_array, cnp.NPY_ARRAY_OWNDATA)

    except MemoryError:
        free(r_mean_v)
        free(r_se_v)

    finally:
        free(count_v)
        free(result.mean_v)
        free(result.se_v)

    return MeanResult(mean=mean_array, se=se_array)

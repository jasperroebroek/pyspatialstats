cimport numpy as cnp


cdef struct LinearRegressionState:
    size_t nf
    double count
    double* XtX          # shape (nf, nf)
    double* Xty          # shape (nf,)
    double yty
    double y_sum


cdef struct LinearRegressionResult:
    # 0: success
    # 1: too few observations
    # 2: memory errors
    # 3: dpotrf error
    # 4: dpotrs error
    # 5: dpotri error

    size_t nf
    double df
    double* beta
    double* beta_se
    double r_squared
    double r_squared_se
    int status
    bint calc_se
    bint calc_r2


# LinearRegressionState functions
cdef LinearRegressionState* lrs_new(size_t nf) noexcept nogil
cdef int lrs_init(LinearRegressionState* lrs, size_t nf) noexcept nogil
cdef void lrs_free(LinearRegressionState* lrs) noexcept nogil
cdef void lrs_reset(LinearRegressionState* lrs) noexcept nogil
cdef void lrs_add(LinearRegressionState* lrs, double y, double[:] x, double weight=?) noexcept nogil
cdef void lrs_merge(LinearRegressionState* lrs_into, LinearRegressionState* lrs_from) noexcept nogil
cdef void lrs_to_result(LinearRegressionState* lrs, LinearRegressionResult* result, bint calc_se=?, bint calc_r2=?) noexcept nogil

cdef LinearRegressionState* lrs_array_new(size_t count, size_t nf) noexcept nogil
cdef void lrs_array_free(LinearRegressionState* lrs_array, size_t count) noexcept nogil
cdef int lrs_array_init(LinearRegressionState* lrs_array, size_t count, size_t nf) noexcept nogil
cdef void lrs_array_to_bootstrap_result(LinearRegressionState* lrs_array, LinearRegressionResult* lrr, size_t n_boot) noexcept nogil

# RegressionResult functions
cdef LinearRegressionResult* lrr_new(size_t nf) noexcept nogil
cdef int lrr_init(LinearRegressionResult* lrr, size_t nf) noexcept nogil
cdef void lrr_free(LinearRegressionResult* lrr) noexcept nogil
cdef void lrr_reset(LinearRegressionResult* lrr) noexcept nogil

cdef LinearRegressionResult* lrr_array_new(size_t count, size_t nf) noexcept nogil
cdef void lrr_array_free(LinearRegressionResult* lrr_array, size_t count) noexcept nogil

cdef class CyRegressionResult:
    cdef readonly int nf
    cdef readonly cnp.ndarray beta
    cdef readonly cnp.ndarray beta_se
    cdef readonly double r_squared
    cdef readonly double r_squared_se
    cdef readonly int df
    cdef readonly int status
    cdef readonly bint calc_se
    cdef readonly bint calc_r2
    @staticmethod
    cdef CyRegressionResult _from_c_struct(LinearRegressionResult* result, int n_params)

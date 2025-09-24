cdef struct LinearRegressionState:
    size_t nf
    double count
    double* XtX          # shape (nf, nf)
    double* Xty          # shape (nf,)
    double yty
    double y_sum


cdef struct LinearRegressionResult:
    size_t nf
    double df
    double* beta
    double* beta_se
    double r_squared
    double r_squared_se


# LinearRegressionState functions
cdef LinearRegressionState* lrs_new(size_t nf) noexcept nogil
cdef int lrs_init(LinearRegressionState* lrs, size_t nf) noexcept nogil
cdef void lrs_free(LinearRegressionState* lrs) noexcept nogil
cdef void lrs_reset(LinearRegressionState* lrs) noexcept nogil
cdef void lrs_add(LinearRegressionState* lrs, double y, double[:] x, double weight=?) noexcept nogil
cdef void lrs_merge(LinearRegressionState* lrs_into, LinearRegressionState* lrs_from) noexcept nogil
cdef void lrs_to_result(LinearRegressionState* lrs, LinearRegressionResult* result) noexcept nogil

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

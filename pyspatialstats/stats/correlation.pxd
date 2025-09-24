cdef struct CorrelationState:
    size_t count
    double mean_v1
    double mean_v2
    double m2_v1
    double m2_v2
    double cov_v1_v2

# State lifecycle
cdef CorrelationState* crs_new() noexcept nogil
cdef void crs_init(CorrelationState* crs) noexcept nogil
cdef void crs_free(CorrelationState* crs) noexcept nogil
cdef void crs_reset(CorrelationState* crs) noexcept nogil

# Update / merge
cdef void crs_add(CorrelationState* crs, double v1, double v2) noexcept nogil
cdef void crs_merge(CorrelationState* crs_into, CorrelationState* crs_from) noexcept nogil

# Accessors
cdef double crs_corr(CorrelationState* crs) noexcept nogil
cdef double crs_cov(CorrelationState* crs) noexcept nogil
cdef double crs_var1(CorrelationState* crs) noexcept nogil
cdef double crs_var2(CorrelationState* crs) noexcept nogil

# Array utilities
cdef CorrelationState* crs_array_new(size_t count) noexcept nogil

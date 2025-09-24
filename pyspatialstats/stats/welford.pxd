cdef struct WelfordState:
    double count
    double mean
    double m2

cdef WelfordState* ws_new() noexcept nogil
cdef void ws_init(WelfordState* ws) noexcept nogil
cdef void ws_free(WelfordState* ws) noexcept nogil
cdef void ws_reset(WelfordState* ws) noexcept nogil
cdef void ws_add(WelfordState* ws, double v, double weight=?) noexcept nogil
cdef void ws_merge(WelfordState* ws_into, WelfordState* ws_from) noexcept nogil
cdef double ws_mean(WelfordState* ws) noexcept nogil
cdef double ws_std(WelfordState* ws, size_t ddof=?) noexcept nogil
cdef WelfordState* ws_array_new(size_t count) noexcept nogil

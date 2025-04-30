cdef struct IterParams:
    size_t[2] stop
    size_t[2] step
    size_t[2] shape
    size_t[2] fringe
    size_t[2] iter
    long num_values
    double threshold

cdef IterParams*  define_iter_params(
    size_t[2] shape, size_t[2] window_size, double fraction_accepted, bint reduce
) noexcept nogil

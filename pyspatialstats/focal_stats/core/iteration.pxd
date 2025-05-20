cimport numpy as np


cdef struct IterParams:
    size_t[2] stop
    size_t[2] step
    size_t[2] shape
    size_t[2] fringe
    size_t[2] iter
    size_t num_values
    double threshold


cdef IterParams*  define_iter_params(
    size_t[2] shape, size_t[2] window_size, double fraction_accepted, bint reduce
)


cdef struct FocalWindow:
    bint skip
    double* values
    size_t num_values


cdef class Iterator:
    cdef:
        bint reduce
        size_t iterations
        size_t num_values
        double threshold
        size_t[2] shape
        size_t[2] output_shape
        size_t[2] window_shape
        size_t[2] xyiter
        size_t[2] step
        size_t[2] stop
        size_t[2] fringe
        np.uint8_t[:, ::1] mask
        FocalWindow window
    cdef FocalWindow* get_window_values_v1(self, size_t iteration, double[:, ::1] a) noexcept nogil
    cdef void get_output_index(self, size_t iteration, size_t* yx) noexcept nogil

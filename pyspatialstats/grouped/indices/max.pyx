# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


cdef size_t _define_max_ind(size_t[:] ind) noexcept nogil:
    cdef:
        size_t i, n = ind.shape[0]
        size_t max_ind = ind[0]

    for i in range(1, n):
        if ind[i] > max_ind:
            max_ind = ind[i]

    return max_ind


def define_max_ind(size_t[:] ind):
    cdef size_t max_ind

    if ind.shape[0] == 0:
        return -1

    max_ind = _define_max_ind(ind)

    return max_ind

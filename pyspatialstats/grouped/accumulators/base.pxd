from pyspatialstats.random.random cimport Random


cdef class BaseGroupedStatAccumulator:
    cdef:
        size_t eltsize
        size_t capacity
        size_t *count_v
        void *stat_v
    cdef void _deallocate(self) noexcept nogil
    cdef int _reset_stat_v(self) except -1 nogil
    cdef void reset(self) noexcept nogil
    cdef int resize(self, size_t max_ind) except -1 nogil
    cdef size_t get_count(self, size_t ind) noexcept nogil
    cdef size_t get_num_inds(self) noexcept nogil
    cpdef size_t[:] build_indices(self)

cdef class GroupedFloatStatAccumulator(BaseGroupedStatAccumulator):
    cdef double* get_stat_v(self) noexcept nogil

cdef class GroupedIntStatAccumulator(BaseGroupedStatAccumulator):
    cdef size_t* get_stat_v(self) noexcept nogil

cdef class BaseGroupedBootstrapAccumulator(BaseGroupedStatAccumulator):
    cdef:
        size_t n_boot
        Random rng
    cdef int bootstrap_resize(self, size_t capacity) except -1 nogil

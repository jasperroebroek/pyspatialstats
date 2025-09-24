from pyspatialstats.grouped.accumulators.base cimport GroupedIntStatAccumulator


cdef class GroupedCountAccumulator(GroupedIntStatAccumulator):
    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1

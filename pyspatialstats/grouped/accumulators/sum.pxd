from pyspatialstats.grouped.accumulators.base cimport GroupedFloatStatAccumulator


cdef class GroupedSumAccumulator(GroupedFloatStatAccumulator):
    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1

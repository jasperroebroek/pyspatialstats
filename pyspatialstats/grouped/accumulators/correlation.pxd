from pyspatialstats.grouped.accumulators.base cimport BaseGroupedStatAccumulator
from pyspatialstats.stats.correlation cimport CorrelationState


cdef class GroupedCorrelationAccumulator(BaseGroupedStatAccumulator):
    cdef CorrelationState* get_stat_v(self) noexcept nogil
    cpdef int add_data(self, size_t[:] ind, double[:] v1, double[:] v2) except -1

from pyspatialstats.grouped.accumulators.base cimport BaseGroupedStatAccumulator
from pyspatialstats.stats.welford cimport WelfordState


cdef class GroupedWelfordAccumulator(BaseGroupedStatAccumulator):
    cdef size_t std_df
    cdef WelfordState* get_stat_v(self) noexcept nogil
    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1

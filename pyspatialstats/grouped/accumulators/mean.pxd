from pyspatialstats.grouped.accumulators.base cimport BaseGroupedBootstrapAccumulator
from pyspatialstats.stats.welford cimport WelfordState


cdef class GroupedBootstrapMeanAccumulator(BaseGroupedBootstrapAccumulator):
    cdef double* get_stat_v(self) noexcept nogil
    cdef void merge_bootstraps(self, size_t group_idx, WelfordState* ws_result) noexcept nogil
    cpdef int add_data(self, size_t[:] ind, double[:] v) except -1

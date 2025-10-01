from pyspatialstats.grouped.accumulators.base cimport BaseGroupedStatAccumulator, BaseGroupedBootstrapAccumulator
from pyspatialstats.stats.linear_regression cimport LinearRegressionState


cdef class GroupedLinearRegressionAccumulator(BaseGroupedStatAccumulator):
    cdef size_t nf
    cdef bint calc_se
    cdef bint calc_r2
    cdef LinearRegressionState* get_stat_v(self) noexcept nogil
    cpdef int add_data(self, size_t[:] ind, double[:, :] x, double[:] y) except -1


cdef class GroupedBootstrapLinearRegressionAccumulator(BaseGroupedBootstrapAccumulator):
    cdef size_t nf
    cdef LinearRegressionState* get_stat_v(self) noexcept nogil
    cpdef int add_data(self, size_t[:] ind, double[:, :] x, double[:] y) except -1

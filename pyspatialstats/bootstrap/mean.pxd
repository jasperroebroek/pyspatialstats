from pyspatialstats.random.random cimport Random
from pyspatialstats.stats.welford cimport WelfordState


cdef void bootstrap_mean(double* v, size_t n_samples, size_t n_boot, Random rng, WelfordState *result) noexcept nogil

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

extensions = [
    ### FOCAL STATS
    Extension(
        "pyspatialstats.focal_stats.core.iteration",
        ["pyspatialstats/focal_stats/core/iteration.pyx"],
    ),
    Extension(
        "pyspatialstats.focal_stats.core.correlation",
        ["pyspatialstats/focal_stats/core/correlation.pyx"],
    ),
    Extension(
        "pyspatialstats.focal_stats.core.linear_regression",
        ["pyspatialstats/focal_stats/core/linear_regression.pyx"],
    ),
    Extension(
        "pyspatialstats.focal_stats.core.stats",
        ["pyspatialstats/focal_stats/core/stats.pyx"],
    ),
    Extension(
        "pyspatialstats.focal_stats.core.mean",
        ["pyspatialstats/focal_stats/core/mean.pyx"],
    ),
    ### GROUPED STATS
    Extension(
        "pyspatialstats.grouped_stats.core.count",
        ["pyspatialstats/grouped_stats/core/count.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped_stats.core.min",
        ["pyspatialstats/grouped_stats/core/min.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped_stats.core.max",
        ["pyspatialstats/grouped_stats/core/max.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped_stats.core.mean",
        ["pyspatialstats/grouped_stats/core/mean.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped_stats.core.std",
        ["pyspatialstats/grouped_stats/core/std.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped_stats.core.correlation",
        ["pyspatialstats/grouped_stats/core/correlation.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped_stats.core.linear_regression",
        ["pyspatialstats/grouped_stats/core/linear_regression.pyx"],
    ),
    ### STRATA STATS
    Extension(
        "pyspatialstats.strata_stats.core.stats",
        ["pyspatialstats/strata_stats/core/stats.pyx"],
    ),
    ### RANDOM
    Extension(
        "pyspatialstats.random.random",
        ["pyspatialstats/random/random.pyx"],
    ),
    ### STATS + BOOTSTRAP
    Extension(
        "pyspatialstats.stats.mean",
        ["pyspatialstats/stats/mean.pyx"],
    ),
]


setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)

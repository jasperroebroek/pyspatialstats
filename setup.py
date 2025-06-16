import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

misc_extensions = [
    Extension(
        "pyspatialstats.types.cy_types",
        ["pyspatialstats/types/cy_types.pyx"],
    ),
    Extension(
        "pyspatialstats.random.random",
        ["pyspatialstats/random/random.pyx"],
    ),
    Extension(
        "pyspatialstats.bootstrap.mean",
        ["pyspatialstats/bootstrap/mean.pyx"],
    ),
]

focal_stat_extensions = [
    Extension(
        "pyspatialstats.focal.core.correlation",
        ["pyspatialstats/focal/core/correlation.pyx"],
    ),
    Extension(
        "pyspatialstats.focal.core.linear_regression",
        ["pyspatialstats/focal/core/linear_regression.pyx"],
    ),
    Extension(
        "pyspatialstats.focal.core.majority",
        ["pyspatialstats/focal/core/majority.pyx"],
    ),
    Extension(
        "pyspatialstats.focal.core.mean",
        ["pyspatialstats/focal/core/mean.pyx"],
    ),
    Extension(
        "pyspatialstats.focal.core.min",
        ["pyspatialstats/focal/core/min.pyx"],
    ),
    Extension(
        "pyspatialstats.focal.core.max",
        ["pyspatialstats/focal/core/max.pyx"],
    ),
    Extension(
        "pyspatialstats.focal.core.sum",
        ["pyspatialstats/focal/core/sum.pyx"],
    ),
    Extension(
        "pyspatialstats.focal.core.std",
        ["pyspatialstats/focal/core/std.pyx"],
    ),
]

grouped_stat_extensions = [
    Extension(
        "pyspatialstats.grouped.core.count",
        ["pyspatialstats/grouped/core/count.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped.core.min",
        ["pyspatialstats/grouped/core/min.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped.core.max",
        ["pyspatialstats/grouped/core/max.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped.core.mean",
        ["pyspatialstats/grouped/core/mean.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped.core.std",
        ["pyspatialstats/grouped/core/std.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped.core.correlation",
        ["pyspatialstats/grouped/core/correlation.pyx"],
    ),
    Extension(
        "pyspatialstats.grouped.core.linear_regression",
        ["pyspatialstats/grouped/core/linear_regression.pyx"],
    ),
]

strata_stat_extensions = [
    Extension(
        "pyspatialstats.strata.core.stats",
        ["pyspatialstats/strata/core/stats.pyx"],
    )
]


setup(
    packages=find_packages(),
    ext_modules=cythonize(
        misc_extensions
        + grouped_stat_extensions
        + strata_stat_extensions
        + focal_stat_extensions
    ),
    include_dirs=[numpy.get_include()],
)

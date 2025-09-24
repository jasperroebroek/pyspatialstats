import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

misc_extensions = [
    Extension(
        'pyspatialstats.types.cy_types',
        ['pyspatialstats/types/cy_types.pyx'],
    ),
    Extension(
        'pyspatialstats.random.random',
        ['pyspatialstats/random/random.pyx'],
    ),
    Extension(
        'pyspatialstats.bootstrap.mean',
        ['pyspatialstats/bootstrap/mean.pyx'],
    ),
]

stat_extensions = [
    Extension('pyspatialstats.stats.linear_regression',
              ['pyspatialstats/stats/linear_regression.pyx']
    ),
    Extension('pyspatialstats.stats.correlation',
              ['pyspatialstats/stats/correlation.pyx']
    ),
    Extension('pyspatialstats.stats.welford',
              ['pyspatialstats/stats/welford.pyx']
    ),
]

focal_stat_extensions = [
    Extension(
        'pyspatialstats.focal.core.correlation',
        ['pyspatialstats/focal/core/correlation.pyx'],
    ),
    Extension(
        'pyspatialstats.focal.core.linear_regression',
        ['pyspatialstats/focal/core/linear_regression.pyx'],
    ),
    Extension(
        'pyspatialstats.focal.core.majority',
        ['pyspatialstats/focal/core/majority.pyx'],
    ),
    Extension(
        'pyspatialstats.focal.core.mean',
        ['pyspatialstats/focal/core/mean.pyx'],
    ),
    Extension(
        'pyspatialstats.focal.core.max',
        ['pyspatialstats/focal/core/max.pyx'],
    ),
    Extension(
        'pyspatialstats.focal.core.min',
        ['pyspatialstats/focal/core/min.pyx'],
    ),
    Extension(
        'pyspatialstats.focal.core.sum',
        ['pyspatialstats/focal/core/sum.pyx'],
    ),
    Extension(
        'pyspatialstats.focal.core.std',
        ['pyspatialstats/focal/core/std.pyx'],
    ),
]

grouped_stat_extensions = [
    Extension(
        'pyspatialstats.grouped.indices.max',
        ['pyspatialstats/grouped/indices/max.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.base',
        ['pyspatialstats/grouped/accumulators/base.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.welford',
        ['pyspatialstats/grouped/accumulators/welford.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.sum',
        ['pyspatialstats/grouped/accumulators/sum.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.min',
        ['pyspatialstats/grouped/accumulators/min.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.count',
        ['pyspatialstats/grouped/accumulators/count.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.max',
        ['pyspatialstats/grouped/accumulators/max.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.mean',
        ['pyspatialstats/grouped/accumulators/mean.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.correlation',
        ['pyspatialstats/grouped/accumulators/correlation.pyx'],
    ),
    Extension(
        'pyspatialstats.grouped.accumulators.linear_regression',
        ['pyspatialstats/grouped/accumulators/linear_regression.pyx'],
    ),
]


setup(
    packages=find_packages(),
    ext_modules=cythonize(misc_extensions + stat_extensions + grouped_stat_extensions + focal_stat_extensions),
    include_dirs=[numpy.get_include()],
)

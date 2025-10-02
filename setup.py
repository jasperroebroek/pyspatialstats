import os

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

numpy_path = os.path.dirname(np.__file__)
npyrandom_path = os.path.join(numpy_path, 'random', 'lib')
npy_include = os.path.join(numpy_path, '_core', 'include')


misc_extensions = [
    Extension(
        'pyspatialstats.random.random',
        ['pyspatialstats/random/random.pyx'],
        include_dirs=[npy_include],
        library_dirs=[npyrandom_path],
        libraries=['npyrandom'],
    ),
]


bootstrap_extensions = [
    Extension(
        'pyspatialstats.bootstrap.mean',
        ['pyspatialstats/bootstrap/mean.pyx'],
    ),
    Extension(
        'pyspatialstats.bootstrap.linear_regression',
        ['pyspatialstats/bootstrap/linear_regression.pyx'],
    ),
]

stat_extensions = [
    Extension('pyspatialstats.stats.linear_regression', ['pyspatialstats/stats/linear_regression.pyx']),
    Extension('pyspatialstats.stats.correlation', ['pyspatialstats/stats/correlation.pyx']),
    Extension('pyspatialstats.stats.welford', ['pyspatialstats/stats/welford.pyx']),
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


extensions = misc_extensions + bootstrap_extensions + stat_extensions + grouped_stat_extensions + focal_stat_extensions

for ext in extensions:
    ext.define_macros = ext.define_macros or []
    ext.define_macros.append(('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'))


setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    language_level=3,
)

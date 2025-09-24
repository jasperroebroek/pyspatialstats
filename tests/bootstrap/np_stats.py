import numpy as np

from pyspatialstats.random.random import Random
from pyspatialstats.results.stats import MeanResult


def np_bootstrap_mean(v, n_bootstraps, seed=None, rng=None):
    if rng is None:
        rng = Random(seed)
    means = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        sample_idx = rng.np_randints(bound=v.size, n=v.size)
        sample = v[sample_idx]
        means[i] = np.mean(sample)

    return MeanResult(mean=means.mean(), se=means.std())

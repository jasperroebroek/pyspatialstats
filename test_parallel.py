import numpy as np

from pyspatialstats.focal import focal_mean
from pyspatialstats.grouped import grouped_mean_pd, grouped_mean_std_pd

a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)

ind = np.random.randint(0, 10, size=(1000, 1000))

focal_mean(a, window=5)
c = grouped_mean_std_pd(ind, a)

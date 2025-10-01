import numpy as np

from pyspatialstats.bootstrap.config import BootstrapConfig
from pyspatialstats.focal import focal_linear_regression

y = np.random.rand(500, 500)
x = np.random.rand(500, 500, 10)


lr1 = focal_linear_regression(x=x, y=y, window=5, verbose=True, error=None)
lr2 = focal_linear_regression(x=x, y=y, window=5, verbose=True, error='parametric')
l3 = focal_linear_regression(x=x, y=y, window=5, verbose=True, error='bootstrap', bootstrap_config=BootstrapConfig(n_bootstraps=100))

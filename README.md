[![Documentation Status](https://readthedocs.org/projects/pyspatialstats/badge/?version=latest)](https://pyspatialstats.readthedocs.io/en/latest/?badge=latest)

This module provides spatial statistics for python, that runs without the installation of extensive GIS packages.

The package implements three different categories of spatial statistics:
- focal statistics, which are calculated as a moving window over input rasters (2D)
- grouped statistics, which calculates the statistics based on zonal (xD)
- zonal statistics, which calculates the statistics for each stratum and reapplies it to the index data (xD).

Furthermore, this package provides functionality to implement your own focal statistics functions, based on the 
mechanisms of a rolling window (xD).

# Installation

The package can be installed with conda:

```
conda install --channel conda-forge pyspatialstats
```

# Usage example

Focal mean of a 100x100 random numpy array.

```
from pyspatialstats.focal import focal_mean
import numpy as np

x = np.random.rand(100, 100)
fm = focal_mean(x, window=5)
```

# Important links

- API reference: https://pyspatialstats.readthedocs.io/en/latest/api.html
- Documentation: https://pyspatialstats.readthedocs.io/en/latest/index.html

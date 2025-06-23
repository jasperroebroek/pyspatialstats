import numpy as np
import scipy

from pyspatialstats.enums import MajorityMode
from pyspatialstats.focal import focal_majority


def test_focal_majority(rs):
    # majority modes
    a = rs.integers(0, 10, 25).reshape(5, 5)

    # Value when reducing
    mode = scipy.stats.mode(a.flatten()).mode
    if isinstance(mode, np.ndarray):
        mode = mode[0]

    # Values when reducing
    assert focal_majority(a, window=5, majority_mode=MajorityMode.ASCENDING)[2, 2] == mode
    # Values when not reducing
    assert focal_majority(a, window=5, reduce=True, majority_mode=MajorityMode.ASCENDING)[0, 0] == mode

    # Same number of observations in several classes lead to NaN in majority_mode='nan'
    a = np.arange(100).reshape(10, 10)
    assert np.isnan(focal_majority(a, window=10, reduce=True, majority_mode=MajorityMode.NAN))

    # Same number of observations in several classes lead to lowest number in majority_mode='ascending'
    assert focal_majority(a, window=10, reduce=True, majority_mode=MajorityMode.ASCENDING) == 0

    # Same number of observations in several classes lead to highest number in majority_mode='descending'
    assert focal_majority(a, window=10, reduce=True, majority_mode=MajorityMode.DESCENDING) == 99


def test_focal_stats_nan_behaviour_majority():
    a = np.ones((5, 5)).astype(float)
    a[1, 1] = np.nan
    assert focal_majority(a, window=5)[2, 2] == 1
    assert not np.isnan(focal_majority(a, window=5, fraction_accepted=0)[2, 2])
    assert np.isnan(focal_majority(a, window=5, fraction_accepted=1)[2, 2])

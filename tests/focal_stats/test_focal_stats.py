import numpy as np
import pytest
import scipy.stats
from numpy.ma.testutils import assert_array_almost_equal
from pydantic import ValidationError

from pyspatialstats.focal_stats import (
    focal_majority,
    focal_max,
    focal_mean,
    focal_min,
    focal_std,
    focal_sum,
)
from pyspatialstats.rolling import rolling_window


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (focal_mean, np.mean),
        (focal_sum, np.sum),
        (focal_min, np.min),
        (focal_max, np.max),
        (focal_std, np.std),
    ],
)
def test_focal_stats_values(rs, fs, np_fs):
    a = rs.random((5, 5))
    assert np.allclose(fs(a, window=5)[2, 2], np_fs(a))
    assert np.allclose(fs(a, window=5, reduce=True)[0, 0], np_fs(a))

    a = rs.random((100, 100))
    assert np.allclose(
        fs(a, window=5, reduce=False)[2:-2, 2:-2],
        np_fs(rolling_window(a, window=5, reduce=False, flatten=True), axis=-1),
    )
    assert np.allclose(
        fs(a, window=5, reduce=True),
        np_fs(rolling_window(a, window=5, reduce=True, flatten=True), axis=-1),
    )


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (focal_mean, np.nanmean),
        (focal_sum, np.nansum),
        (focal_min, np.nanmin),
        (focal_max, np.nanmax),
        (focal_std, np.nanstd),
    ],
)
def test_focal_stats_values_mask(rs, fs, np_fs):
    a = rs.random((5, 5))
    a[1, 2] = np.nan

    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = False
    mask[-1, -1] = False
    mask[0, -1] = False
    mask[-1, 0] = False

    assert fs(a, window=mask, reduce=True, fraction_accepted=0) == np_fs(a[mask])


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (focal_mean, np.nanmean),
        (focal_sum, np.nansum),
        (focal_min, np.nanmin),
        (focal_max, np.nanmax),
        (focal_std, np.nanstd),
    ],
)
def test_focal_stats_values_mask(rs, fs, np_fs):
    a = rs.random((100, 100))
    assert_array_almost_equal(
        fs(a, window=5)[2:-2, 2:-2],
        np_fs(rolling_window(a, window=5, flatten=True), axis=-1),
    )


@pytest.mark.parametrize(
    "fs",
    [
        focal_mean,
        focal_sum,
        focal_min,
        focal_max,
        focal_std,
        focal_majority,
    ],
)
def test_focal_stats_shape(rs, fs):
    a = rs.random((10, 10))
    assert a.shape == fs(a, window=3).shape
    assert fs(a, window=10, reduce=True).shape == (1, 1)


@pytest.mark.parametrize(
    "fs",
    [
        focal_mean,
        focal_sum,
        focal_min,
        focal_max,
        focal_std,
        focal_majority,
    ],
)
def test_focal_stats_errors(rs, fs):
    # Not boolean
    with pytest.raises(ValidationError):
        fs(rs.random((10, 10)), window=5, verbose=2)

    with pytest.raises(ValidationError):
        fs(rs.random((10, 10)), window=5, reduce=2)

    # not 2D
    with pytest.raises(IndexError):
        a = rs.random((10, 10, 10))
        fs(a, window=5)

    with pytest.raises(ValidationError):
        a = rs.random((10, 10))
        fs(a, window="x")

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        fs(a, window=1)

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        fs(a, window=11)

    # uneven window_shape is not supported
    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        fs(a, window=4)

    # Not exactly divided in reduce mode
    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        fs(a, window=4, reduce=True)

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        fs(a, window=5, fraction_accepted=-0.1)

    with pytest.raises(ValueError):
        a = rs.random((10, 10))
        fs(a, window=5, fraction_accepted=1.1)


@pytest.mark.parametrize(
    "fs",
    [
        focal_mean,
        focal_sum,
        focal_min,
        focal_max,
        focal_std,
        focal_majority,
    ],
)
def test_focal_stats_nan_propagation(rs, fs):
    a = rs.random((5, 5))
    a[2, 2] = np.nan
    assert np.isnan(fs(a, window=5)[2, 2])


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (focal_mean, np.nanmean),
        (focal_sum, np.nansum),
        (focal_min, np.nanmin),
        (focal_max, np.nanmax),
        (focal_std, np.nanstd),
    ],
)
def test_focal_stats_nan_behaviour_fraction_accepted(rs, fs, np_fs):
    a = rs.random((5, 5))
    a[1, 1] = np.nan

    assert np.allclose(fs(a, window=5)[2, 2], np_fs(a))
    assert not np.isnan(fs(a, window=5, fraction_accepted=0)[2, 2])
    assert np.isnan(fs(a, window=5, fraction_accepted=1)[2, 2])


@pytest.mark.parametrize(
    "fs",
    [
        focal_mean,
        focal_sum,
        focal_min,
        focal_max,
        focal_std,
        focal_majority,
    ],
)
def test_focal_stats_dtype(rs, fs):
    a = rs.random((5, 5)).astype(np.int32)
    assert fs(a, window=5).dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    assert fs(a, window=5).dtype == np.float64


def test_focal_majority(rs):
    # majority modes
    a = rs.integers(0, 10, 25).reshape(5, 5)

    # Value when reducing
    mode = scipy.stats.mode(a.flatten()).mode
    if isinstance(mode, np.ndarray):
        mode = mode[0]

    # Values when reducing
    assert focal_majority(a, window=5, majority_mode="ascending")[2, 2] == mode
    # Values when not reducing
    assert (
        focal_majority(a, window=5, reduce=True, majority_mode="ascending")[0, 0]
        == mode
    )

    # Same number of observations in several classes lead to NaN in majority_mode='nan'
    a = np.arange(100).reshape(10, 10)
    assert np.isnan(focal_majority(a, window=10, reduce=True, majority_mode="nan"))

    # Same number of observations in several classes lead to lowest number in majority_mode='ascending'
    assert focal_majority(a, window=10, reduce=True, majority_mode="ascending") == 0

    # Same number of observations in several classes lead to highest number in majority_mode='descending'
    assert focal_majority(a, window=10, reduce=True, majority_mode="descending") == 99


def test_focal_stats_nan_behaviour_majority():
    a = np.ones((5, 5)).astype(float)
    a[1, 1] = np.nan
    assert focal_majority(a, window=5)[2, 2] == 1
    assert not np.isnan(focal_majority(a, window=5, fraction_accepted=0)[2, 2])
    assert np.isnan(focal_majority(a, window=5, fraction_accepted=1)[2, 2])

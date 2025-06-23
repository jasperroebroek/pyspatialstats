import numpy as np
import pytest
import xarray as xr

from pyspatialstats.enums import Uncertainty
from pyspatialstats.focal import focal_mean
from pyspatialstats.types.results import MeanResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.windows import define_window


def test_focal_stats_values(rs):
    a = rs.random((5, 5))
    assert np.allclose(focal_mean(a, window=5).mean[2, 2], a.mean())
    assert np.allclose(focal_mean(a, window=5, reduce=True).mean[0, 0], a.mean())

    assert np.allclose(focal_mean(a, window=5, uncertainty=Uncertainty.STD).std[2, 2], a.std())
    assert np.allclose(focal_mean(a, window=5, reduce=True, uncertainty=Uncertainty.STD).std[0, 0], a.std())

    a = rs.random((100, 100))
    assert np.allclose(
        focal_mean(a, window=5, reduce=False).mean[2:-2, 2:-2],
        np.mean(rolling_window(a, window=5, reduce=False, flatten=True), axis=-1),
    )
    assert np.allclose(
        focal_mean(a, window=5, reduce=True).mean,
        np.mean(rolling_window(a, window=5, reduce=True, flatten=True), axis=-1),
    )
    assert np.allclose(
        focal_mean(a, window=5, reduce=False, uncertainty=Uncertainty.STD).std[2:-2, 2:-2],
        np.std(rolling_window(a, window=5, reduce=False, flatten=True), axis=-1),
    )
    assert np.allclose(
        focal_mean(a, window=5, reduce=True, uncertainty=Uncertainty.STD).std,
        np.std(rolling_window(a, window=5, reduce=True, flatten=True), axis=-1),
    )


def test_focal_mean_values_mask(rs):
    a = rs.random((5, 5))
    a[1, 2] = np.nan

    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = False
    mask[-1, -1] = False
    mask[0, -1] = False
    mask[-1, 0] = False

    assert focal_mean(a, window=mask, fraction_accepted=0, reduce=True).mean == np.nanmean(a[mask])


@pytest.mark.parametrize('reduce', (True, False))
def test_parallel(rs, reduce):
    a = rs.random((100, 100))

    linear_r = focal_mean(a, window=5, reduce=reduce, uncertainty=Uncertainty.STD)
    parallel_r = focal_mean(a, window=5, reduce=reduce, uncertainty=Uncertainty.STD, chunks=75)

    np.testing.assert_allclose(linear_r.mean, parallel_r.mean, equal_nan=True)
    np.testing.assert_allclose(linear_r.std, parallel_r.std, equal_nan=True)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
def test_xarray_output(rs, reduce, chunks):
    a = xr.DataArray(rs.random((100, 100)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a)

    output = MeanResult(
        mean=xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=np.float64)),
        std=xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=np.float64)),
    )

    r = focal_mean(a, window=5, out=output, reduce=reduce, chunks=chunks, uncertainty=Uncertainty.STD)
    r_expected = focal_mean(a, window=5, out=None, reduce=reduce, chunks=chunks, uncertainty=Uncertainty.STD)

    assert id(r.mean) == id(output.mean)
    assert id(r.std) == id(output.std)

    np.testing.assert_allclose(r.mean.values, r_expected.mean, equal_nan=True)
    np.testing.assert_allclose(r.std.values, r_expected.std, equal_nan=True)

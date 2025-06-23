import numpy as np
import pytest
import xarray as xr
from numpy.ma.testutils import assert_array_almost_equal

from pyspatialstats.focal import focal_correlation as focal_correlation_internal
from pyspatialstats.focal import focal_linear_regression as focal_linear_regression_internal
from pyspatialstats.focal import (
    focal_majority,
    focal_max,
    focal_min,
    focal_std,
    focal_sum,
)
from pyspatialstats.focal import focal_mean as focal_mean_internal
from pyspatialstats.focal.result_config import (
    FocalArrayResultConfig,
    FocalCorrelationResultConfig,
    FocalLinearRegressionResultConfig,
    FocalMeanResultConfig,
)
from pyspatialstats.rolling import rolling_window
from pyspatialstats.utils import get_dtype
from pyspatialstats.windows import define_window


def focal_mean(*args, **kwargs):
    return focal_mean_internal(*args, **kwargs).mean


def focal_correlation(a, **kwargs):
    return focal_correlation_internal(a1=a, a2=a, **kwargs).c


def focal_linear_regression(a, **kwargs):
    return focal_linear_regression_internal(a1=a, a2=a, **kwargs).a


focal_stat_functions = (
    focal_mean,
    focal_min,
    focal_max,
    focal_sum,
    focal_std,
    focal_correlation,
    focal_linear_regression,
    focal_majority,
)


@pytest.mark.parametrize(
    'fs,np_fs',
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
    'fs,np_fs',
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
    'fs,np_fs',
    [
        (focal_mean, np.nanmean),
        (focal_sum, np.nansum),
        (focal_min, np.nanmin),
        (focal_max, np.nanmax),
        (focal_std, np.nanstd),
    ],
)
def test_focal_stats_values_against_rolling(rs, fs, np_fs):
    a = rs.random((100, 100))
    assert_array_almost_equal(
        fs(a, window=5)[2:-2, 2:-2],
        np_fs(rolling_window(a, window=5, flatten=True), axis=-1),
    )


@pytest.mark.parametrize('fs', focal_stat_functions)
def test_focal_stats_shape(rs, fs):
    a = rs.random((10, 10))
    assert a.shape == fs(a, window=3).shape
    assert fs(a, window=10, reduce=True).shape == (1, 1)


@pytest.mark.parametrize('fs', focal_stat_functions)
def test_focal_stats_errors(rs, fs):
    # not 2D
    with pytest.raises(IndexError):
        a = rs.random((10, 10, 10))
        fs(a, window=5)

    with pytest.raises(TypeError):
        a = rs.random((10, 10))
        fs(a, window='x')

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


@pytest.mark.parametrize('fs', focal_stat_functions)
def test_focal_stats_nan_propagation(rs, fs):
    a = rs.random((5, 5))
    a[2, 2] = np.nan
    assert np.isnan(fs(a, window=5)[2, 2])


@pytest.mark.parametrize(
    'fs,np_fs',
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


@pytest.mark.parametrize('fs', focal_stat_functions)
def test_focal_stats_dtype(rs, fs):
    a = rs.random((5, 5)).astype(np.int32)
    assert fs(a, window=5).dtype == np.float64

    a = rs.random((5, 5)).astype(np.float64)
    assert fs(a, window=5).dtype == np.float64


@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize('fs', focal_stat_functions)
def test_parallel(rs, reduce, fs):
    a = rs.random((100, 100))

    linear_r = fs(a, window=5, reduce=reduce)
    parallel_r = fs(a, window=5, reduce=reduce, chunks=75)

    np.testing.assert_allclose(linear_r, parallel_r, equal_nan=True)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize(
    'fs,config',
    [
        (focal_mean, FocalMeanResultConfig()),
        (focal_sum, FocalArrayResultConfig()),
        (focal_min, FocalArrayResultConfig()),
        (focal_max, FocalArrayResultConfig()),
        (focal_std, FocalArrayResultConfig()),
        (focal_majority, FocalArrayResultConfig()),
        (focal_linear_regression, FocalLinearRegressionResultConfig(p_values=False)),
        (focal_correlation, FocalCorrelationResultConfig(p_values=False)),
    ],
)
def test_xarray_output(rs, reduce, fs, config, chunks):
    a = xr.DataArray(rs.random((100, 100)))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a)

    if isinstance(config, FocalArrayResultConfig):
        out = xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=np.float64))
    else:
        out = config.return_type(
            **{
                field: xr.DataArray(np.full(output_shape, fill_value=np.nan, dtype=get_dtype(field)))
                for field in config.active_fields
            }
        )

    r = fs(a, window=5, out=out, reduce=reduce, chunks=chunks)
    r_expected = fs(a, window=5, out=None, reduce=reduce, chunks=chunks)

    ids = (
        (id(out),)
        if isinstance(out, xr.DataArray)
        else tuple(id(getattr(out, field)) for field in config.active_fields)
    )
    assert id(r) in ids
    np.testing.assert_allclose(r.values, r_expected, equal_nan=True)


@pytest.mark.parametrize('chunks', (None, 75))
@pytest.mark.parametrize('reduce', (True, False))
@pytest.mark.parametrize(
    'fs,config',
    [
        (focal_mean, FocalMeanResultConfig()),
        (focal_sum, FocalArrayResultConfig()),
        (focal_min, FocalArrayResultConfig()),
        (focal_max, FocalArrayResultConfig()),
        (focal_std, FocalArrayResultConfig()),
        (focal_majority, FocalArrayResultConfig()),
        (focal_linear_regression, FocalLinearRegressionResultConfig(p_values=False)),
        (focal_correlation, FocalCorrelationResultConfig(p_values=False)),
    ],
)
def test_numpy_output(rs, reduce, fs, config, chunks):
    a = rs.random((100, 100))
    window = define_window(5)
    output_shape = window.define_windowed_shape(reduce=reduce, a=a)

    if isinstance(config, FocalArrayResultConfig):
        out = np.full(output_shape, fill_value=np.nan, dtype=np.float64)
    else:
        out = config.return_type(
            **{
                field: np.full(output_shape, fill_value=np.nan, dtype=get_dtype(field))
                for field in config.active_fields
            }
        )

    ids = (
        (id(out),)
        if isinstance(out, np.ndarray)
        else tuple(id(getattr(out, field)) for field in config.active_fields)
    )

    r = fs(a, window=5, out=out, reduce=reduce, chunks=chunks)
    r_expected = fs(a, window=5, out=None, reduce=reduce, chunks=chunks)

    assert id(r) in ids

    np.testing.assert_allclose(r, r_expected, equal_nan=True)

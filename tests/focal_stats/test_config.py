import numpy as np
import pytest

from pyspatialstats.focal.result_config import (
    FocalArrayResultConfig,
    FocalMeanResultConfig,
)
from pyspatialstats.results.stats import MeanResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.windows import define_window


@pytest.fixture
def window():
    return define_window(5)


@pytest.mark.parametrize('reduce', (True, False))
def test_array_result_config_create_output(v, window, reduce):
    config = FocalArrayResultConfig()
    out = config.create_output(v.shape, window=window, reduce=reduce)
    assert isinstance(out, np.ndarray)
    np.testing.assert_equal(out.shape, window.define_windowed_shape(reduce, a=v))


@pytest.mark.parametrize('reduce', (True, False))
def test_array_result_config_get_cython_input(v, window, reduce):
    config = FocalArrayResultConfig()
    out = config.create_output(v.shape, window, reduce=reduce)
    cy_input = config.get_cython_input(v.shape, window=window, reduce=reduce, out=out)
    v_windowed = rolling_window(v, window=window, reduce=reduce)
    assert isinstance(cy_input, dict)
    assert 'r' in cy_input
    np.testing.assert_equal(cy_input['r'].shape, v_windowed.shape[:2])


def test_array_result_config_validate_output_valid(v, window):
    config = FocalArrayResultConfig()
    out = config.create_output(v.shape, window, reduce=False)
    config.validate_output(v.shape, window, reduce=False, out=out)


def test_mean_result_config_active_fields():
    cfg_none = FocalMeanResultConfig(error=None)
    cfg_se = FocalMeanResultConfig(error='bootstrap')
    cfg_std = FocalMeanResultConfig(error='parametric')

    assert cfg_none.active_fields == ('mean',)
    assert cfg_se.active_fields == ('mean', 'se')
    assert cfg_std.active_fields == ('mean', 'std')


@pytest.mark.parametrize('reduce', (True, False))
def test_mean_result_config_create_output(v, window, reduce):
    config = FocalMeanResultConfig(error='parametric')
    out = config.create_output(v.shape, window, reduce=reduce)
    assert isinstance(out, MeanResult)
    np.testing.assert_equal(out.mean.shape, window.define_windowed_shape(reduce, a=v))
    np.testing.assert_equal(out.std.shape, window.define_windowed_shape(reduce, a=v))


def test_parse_output_creates_when_none(v, window):
    config = FocalMeanResultConfig(error=None)
    out = config.parse_output(v.shape[:2], out=None, window=window, reduce=False)
    assert isinstance(out, MeanResult)


def test_parse_output_validates_existing(v, window):
    config = FocalMeanResultConfig(error=None)
    shape = v.shape[0], v.shape[1]
    out = config.create_output(raster_shape=shape, window=window, reduce=False)
    result = config.parse_output(raster_shape=shape, out=out, window=window, reduce=False)
    assert result is out

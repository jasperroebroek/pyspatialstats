import numpy as np
import pytest

from pyspatialstats.enums import Uncertainty
from pyspatialstats.focal.result_config import (
    FocalArrayResultConfig,
    FocalCorrelationResultConfig,
    FocalLinearRegressionResultConfig,
    FocalMeanResultConfig,
)
from pyspatialstats.types.results import MeanResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.windows import define_window


@pytest.fixture
def window():
    return define_window(5)


@pytest.mark.parametrize('reduce', (True, False))
def test_array_result_config_create_output(v, window, reduce):
    config = FocalArrayResultConfig()
    out = config.create_output(v, window=window, reduce=reduce)
    assert isinstance(out, np.ndarray)
    np.testing.assert_equal(out.shape, window.define_windowed_shape(reduce, a=v))


@pytest.mark.parametrize('reduce', (True, False))
def test_array_result_config_get_cython_input(v, window, reduce):
    config = FocalArrayResultConfig()
    out = config.create_output(v, window, reduce=reduce)
    cy_input = config.get_cython_input(v, window=window, reduce=reduce, out=out)
    v_windowed = rolling_window(v, window=window, reduce=reduce)
    assert isinstance(cy_input, dict)
    assert 'r' in cy_input
    np.testing.assert_equal(cy_input['r'].shape, v_windowed.shape[:2])


def test_array_result_config_validate_output_valid(v, window):
    config = FocalArrayResultConfig()
    out = config.create_output(v, window, reduce=False)
    config.validate_output(v, window, reduce=False, out=out)


def test_mean_result_config_active_fields():
    cfg_none = FocalMeanResultConfig(uncertainty=None)
    cfg_se = FocalMeanResultConfig(uncertainty=Uncertainty.SE)
    cfg_std = FocalMeanResultConfig(uncertainty=Uncertainty.STD)

    assert cfg_none.active_fields == ('mean',)
    assert cfg_se.active_fields == ('mean', 'se')
    assert cfg_std.active_fields == ('mean', 'std')


@pytest.mark.parametrize('reduce', (True, False))
def test_mean_result_config_create_output(v, window, reduce):
    config = FocalMeanResultConfig(uncertainty=Uncertainty.STD)
    out = config.create_output(v, window, reduce=reduce)
    assert isinstance(out, MeanResult)
    np.testing.assert_equal(out.mean.shape, window.define_windowed_shape(reduce, a=v))
    np.testing.assert_equal(out.std.shape, window.define_windowed_shape(reduce, a=v))


def test_correlation_result_config_active_fields():
    config = FocalCorrelationResultConfig(p_values=True)
    assert 'p' in config.active_fields
    config_no_p = FocalCorrelationResultConfig(p_values=False)
    assert 'p' not in config_no_p.active_fields


def test_regression_result_config_active_fields():
    config = FocalLinearRegressionResultConfig(p_values=True)
    assert any(field.startswith('p_') for field in config.active_fields)
    config_no_p = FocalLinearRegressionResultConfig(p_values=False)
    assert all(not field.startswith('p_') for field in config_no_p.active_fields)


def test_parse_output_creates_when_none(v, window):
    config = FocalMeanResultConfig(uncertainty=None)
    out = config.parse_output(a=v, out=None, window=window, reduce=False)
    assert isinstance(out, MeanResult)


def test_parse_output_validates_existing(v, window):
    config = FocalMeanResultConfig(uncertainty=None)
    out = config.create_output(v, window, reduce=False)
    result = config.parse_output(a=v, out=out, window=window, reduce=False)
    assert result is out

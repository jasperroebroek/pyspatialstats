import warnings
from typing import Dict

import numpy as np
import pytest

from pyspatialstats.focal import focal_mean
from pyspatialstats.focal.function import focal_function


def mean_fun_t(x: np.ndarray, key: str) -> Dict[str, float]:
    """Mean of first output is fed to the first output"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {key: np.nanmean(x).item()}


def test_focal_function(rs):
    inputs = {'x': rs.random((10, 10))}
    outputs = {'y': np.full((2, 2), np.nan)}

    focal_function(mean_fun_t, inputs, outputs, window=5, reduce=True, key='y')

    # Check for equality
    m = focal_mean(inputs['x'], window=5, fraction_accepted=0, reduce=True)
    np.allclose(m.mean, outputs['y'], equal_nan=True)


def test_focal_function_keyword_only(rs):
    def _mean_fun_t_keyword_only(*, x: np.ndarray, key: str) -> Dict[str, float]:
        """Mean of first output is fed to the first output"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return {key: np.nanmean(x).item()}

    inputs = {'x': rs.random((10, 10))}
    outputs = {'y': np.full((2, 2), np.nan)}

    focal_function(_mean_fun_t_keyword_only, inputs, outputs, window=5, reduce=True, key='y')


def test_window_definition_errors(rs):
    inputs = {'x': rs.random((10, 10))}
    outputs = {'y': np.full((10, 10), np.nan)}

    with pytest.raises(ValueError):
        focal_function(mean_fun_t, inputs, outputs, window=(0, 0), key='y')

    with pytest.raises(ValueError):
        focal_function(mean_fun_t, inputs, outputs, window=(11, 11), key='y')

    with pytest.raises(ValueError):
        focal_function(mean_fun_t, inputs, outputs, window=(2, 2), key='y')


def test_dimension_errors():
    # todo
    pass

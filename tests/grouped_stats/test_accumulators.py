import numpy as np
import pytest

from pyspatialstats.grouped.accumulators.base import (
    GroupedFloatStatAccumulator,
    GroupedIntStatAccumulator,
)
from pyspatialstats.grouped.accumulators.correlation import GroupedCorrelationAccumulator
from pyspatialstats.grouped.accumulators.count import GroupedCountAccumulator
from pyspatialstats.grouped.accumulators.linear_regression import (
    GroupedBootstrapLinearRegressionAccumulator,
    GroupedLinearRegressionAccumulator,
)
from pyspatialstats.grouped.accumulators.max import GroupedMaxAccumulator
from pyspatialstats.grouped.accumulators.mean import GroupedBootstrapMeanAccumulator
from pyspatialstats.grouped.accumulators.min import GroupedMinAccumulator
from pyspatialstats.grouped.accumulators.sum import GroupedSumAccumulator
from pyspatialstats.grouped.accumulators.welford import GroupedWelfordAccumulator

ACCUMULATORS = [
    GroupedFloatStatAccumulator,
    GroupedIntStatAccumulator,
    GroupedCountAccumulator,
    GroupedMaxAccumulator,
    GroupedMinAccumulator,
    GroupedSumAccumulator,
    GroupedWelfordAccumulator,
    GroupedBootstrapMeanAccumulator,
    GroupedCorrelationAccumulator,
    GroupedLinearRegressionAccumulator,
    GroupedBootstrapLinearRegressionAccumulator,
]


TO_RESULT_FUNC = {
    GroupedWelfordAccumulator: lambda x: x.to_std_result(),
    GroupedBootstrapMeanAccumulator: lambda x: x.to_result().mean,
    GroupedCorrelationAccumulator: lambda x: x.to_result().c,
    GroupedLinearRegressionAccumulator: lambda x: x.to_result().beta[:, 0],
    GroupedBootstrapLinearRegressionAccumulator: lambda x: x.to_result().r_squared,
}


TO_FILTERED_RESULT_FUNC = {
    GroupedWelfordAccumulator: lambda x: x.to_std_filtered_result(),
}


ELTSIZE = {
    GroupedWelfordAccumulator: 8 * 2 + np.dtype(np.uintp).itemsize,
    GroupedCorrelationAccumulator: 8 * 5 + np.dtype(np.uintp).itemsize,
    GroupedLinearRegressionAccumulator: 8 * 4 + np.dtype(np.uintp).itemsize * 2,
    GroupedBootstrapLinearRegressionAccumulator: 8 * 4 + np.dtype(np.uintp).itemsize * 2,
}


CHECK_NULL = {
    GroupedCountAccumulator: lambda x: x == 0,
    GroupedIntStatAccumulator: lambda x: x == 0,
    GroupedSumAccumulator: lambda x: x == 0
}

DEFAULT_CHECK_NULL = lambda x: np.isnan(x)  # noqa


@pytest.mark.parametrize('accumulator', ACCUMULATORS)
def test_accumulator_initialization(accumulator):
    a = accumulator(5)
    result = TO_RESULT_FUNC.get(accumulator, lambda x: x.to_result())(a)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64 or result.dtype == np.uintp
    assert result.shape == (5,)
    assert np.all(CHECK_NULL.get(accumulator, DEFAULT_CHECK_NULL)(result))

    a = accumulator(5)
    filtered_result = TO_FILTERED_RESULT_FUNC.get(accumulator, lambda x: x.to_filtered_result())(a)
    assert filtered_result.result.size == 0


@pytest.mark.parametrize('r', ACCUMULATORS)
def test_to_dict(r):
    result_obj = r(2)
    d = result_obj.to_dict()
    assert isinstance(d, dict)
    assert {'capacity', 'eltsize', 'count_v', 'stat_v', 'num_inds', 'indices'} <= d.keys()
    assert isinstance(d['count_v'], np.ndarray)
    assert isinstance(d['stat_v'], np.ndarray)
    assert d['num_inds'] == 0
    assert d['count_v'].shape[0] == d['capacity']
    assert d['stat_v'].shape[0] == d['capacity']
    assert d['eltsize'] == ELTSIZE.get(r, d['stat_v'].dtype.itemsize)


@pytest.mark.parametrize('accumulator', ACCUMULATORS)
def test_empty(accumulator):
    a = accumulator()
    assert TO_RESULT_FUNC.get(accumulator, lambda x: x.to_result())(a).size == 0
    assert TO_FILTERED_RESULT_FUNC.get(accumulator, lambda x: x.to_filtered_result())(a).result.size == 0


@pytest.mark.parametrize('accumulator', ACCUMULATORS)
def test_resizing(accumulator):
    a = accumulator()
    assert a.py_resize(0) == 0
    assert a.py_resize(1) == 0
    assert a.py_resize(2) == 0

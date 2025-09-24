from typing import Dict

import numpy as np

from pyspatialstats.types.arrays import Array
from pyspatialstats.utils import get_dtype


def parse_array(name: str, dv: Array) -> np.ndarray[tuple[int, ...], np.generic]:
    return np.ascontiguousarray(dv, dtype=get_dtype(name))


def parse_data(ind: Array, **kwargs: Array) -> Dict[str, np.ndarray[tuple[int], np.generic]]:
    parsed_data = {'ind': parse_array('ind', ind)}
    parsed_data.update({k: parse_array(k, kwargs[k]) for k in kwargs})

    for d in parsed_data:
        if parsed_data[d].shape != parsed_data['ind'].shape:
            raise IndexError(f'Arrays are not all of the same shape: {ind.shape=} {parsed_data[d].shape=}')

    return {k: v.ravel() for k, v in parsed_data.items()}


def parse_data_linear_regression(ind: Array, y: Array, x: Array) -> Dict[str, np.ndarray[tuple[int, ...], np.generic]]:
    """X contains features, which are stored in the last dimension"""
    if (
        (ind.shape != y.shape)
        or (x.ndim == y.ndim and not x.shape == y.shape)
        or (x.ndim == y.ndim + 1 and not x.shape[:-1] == y.shape)
    ):
        raise IndexError(
            f'Arrays are not compatible: {ind.shape=} {x.shape=} {y.shape=}. Ind and y must be the same shape, while x '
            f'needs be either the same shape as y or have one dimension more than y.'
        )

    n_features = 1 if x.size == y.size else x.shape[-1]

    return {
        'ind': parse_array('ind', ind).flatten(),
        'y': parse_array('y', y).flatten(),
        'x': np.ascontiguousarray(x, dtype=np.float64).reshape(-1, n_features),
    }


def define_max_ind(ind: np.ndarray[tuple[int, ...], np.generic]) -> int:
    from pyspatialstats.grouped.indices.max import define_max_ind as cydefine_max_ind

    ind_flat = np.ascontiguousarray(ind, dtype=np.uintp).ravel()
    return cydefine_max_ind(ind_flat)

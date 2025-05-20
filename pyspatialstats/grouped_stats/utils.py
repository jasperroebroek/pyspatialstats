from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from numpydantic import NDArray, Shape
from numpydantic.dtype import Int64

from pyspatialstats.grouped_stats.core.count import define_max_ind as cydefine_max_ind
from pyspatialstats.types import Result


def define_max_ind(ind: NDArray) -> int:
    ind_flat = np.ascontiguousarray(ind, dtype=np.uintp).ravel()
    return cydefine_max_ind(ind_flat)


def generate_index(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Int64]:
    from pyspatialstats.grouped_stats import grouped_count

    return np.argwhere(grouped_count(ind, v)).ravel()


def parse_array(name: str, dv: NDArray) -> NDArray:
    if name != "ind" and not name.startswith("v"):
        raise ValueError("Only ind an variables are valid keywords")

    dtype = np.uintp if name == "ind" else np.float64
    return np.ascontiguousarray(dv, dtype=dtype)


def parse_data(ind: NDArray, **data) -> Dict[str, NDArray[Shape["*"], Any]]:
    parsed_data = {"ind": parse_array("ind", ind)}
    parsed_data.update({d: parse_array(d, data[d]) for d in data})

    for d in parsed_data:
        if parsed_data[d].shape != parsed_data["ind"].shape:
            raise IndexError("Arrays are not all of the same shape")

    return {k: v.ravel() for k, v in parsed_data.items()}


def grouped_fun(
    fun: Callable,
    ind: NDArray,
    n_bootstraps: Optional[int] = None,
    seed: Optional[int] = None,
    **data,
) -> NDArray[Shape["*"], Any] | Result:
    kwargs = {}
    if n_bootstraps is not None:
        kwargs["n_bootstraps"] = n_bootstraps
    if seed is not None:
        kwargs["seed"] = seed

    parsed_data = parse_data(ind, **data)

    return fun(**parsed_data, **kwargs)


def grouped_fun_pd(fun: Callable, name: str, ind, n_bootstraps=None, seed=None, **data) -> pd.DataFrame:
    kwargs = {}
    if n_bootstraps is not None:
        kwargs["n_bootstraps"] = n_bootstraps
    if seed is not None:
        kwargs["seed"] = seed
    
    parsed_data = parse_data(ind, **data)

    nan_mask = np.zeros_like(parsed_data["ind"], dtype=np.float64)

    for d in parsed_data:
        if d == "ind":
            continue
        nan_mask[np.isnan(parsed_data[d])] = np.nan

    index = generate_index(parsed_data["ind"], nan_mask)
    r = fun(**parsed_data, **kwargs)

    if isinstance(r, np.ndarray):
        return pd.DataFrame(data={name: r}, index=index)

    # assume r is a namedtuple
    return pd.DataFrame(data=r._asdict(), index=index)

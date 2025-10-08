from typing import Optional

from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from pyspatialstats.grouped.config import GroupedResultConfig
from pyspatialstats.grouped.accumulators.base import BaseGroupedStatAccumulator
from pyspatialstats.results.stats import StatResult
from pyspatialstats.types.arrays import VectorT
from pyspatialstats.views import ArrayView, construct_tile_views


def grouped_stats_base(ind: ArrayLike, config: GroupedResultConfig, **kwargs) -> BaseGroupedStatAccumulator:
    parsed_data = config.parse_data_fun(ind=ind, **kwargs)
    acc = config.get_accumulator()
    if ind.size > 0:
        acc.add_data(**parsed_data)
    return acc


def grouped_stats_parallel_tile(
    ind: ArrayLike,
    config: GroupedResultConfig,
    tile_view: ArrayView,
    **kwargs,
) -> BaseGroupedStatAccumulator:
    return grouped_stats_base(
        ind=ind[tuple(tile_view.slices)], config=config, **{k: v[tuple(tile_view.slices)] for k, v in kwargs.items()}
    )


def grouped_stats_parallel(
    ind: ArrayLike,
    config: GroupedResultConfig,
    tile_shape: tuple[int, ...],
    **kwargs,
) -> BaseGroupedStatAccumulator:
    tile_views = construct_tile_views(ind.shape, tile_shape)

    return sum(
        Parallel(prefer='threads', return_as='generator_unordered')(
            delayed(grouped_stats_parallel_tile)(ind=ind, config=config, tile_view=tile_view, **kwargs)
            for tile_view in tile_views
        )
    )


def grouped_stats(
    ind: ArrayLike,
    config: GroupedResultConfig,
    filtered: bool,
    chunks: Optional[int | tuple[int, ...]] = None,
    **kwargs,
) -> Optional[VectorT | StatResult]:
    if chunks is None:
        cy_stat_r = grouped_stats_base(ind=ind, config=config, **kwargs)
    else:
        tile_shape = chunks if isinstance(chunks, tuple) else (chunks,) * ind.ndim
        cy_stat_r = grouped_stats_parallel(ind=ind, config=config, tile_shape=tile_shape, **kwargs)

    if not filtered:
        return getattr(cy_stat_r, config.to_result_func)()

    return getattr(cy_stat_r, config.to_filtered_result_func)().to_dataframe()

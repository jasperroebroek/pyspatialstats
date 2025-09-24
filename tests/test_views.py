import pytest

from pyspatialstats.views import (
    ArrayView,
    ArrayViewPair,
    construct_window_views,
    construct_windowed_tile_views,
)
from pyspatialstats.windows import define_window


def test_window_view_definition_errors():
    with pytest.raises(ValueError):
        construct_window_views((1, 1), window=0, reduce=False)


@pytest.mark.parametrize('ws', [3, 5, 7])
def test_window_view_definition_reduce(ws):
    wps = list(construct_window_views((ws * 2, ws * 2), window=ws, reduce=True))

    assert len(wps) == 4

    for wp in wps:
        assert wp in [
            ArrayViewPair(
                input=ArrayView(offset=(0, 0), shape=(ws, ws)), output=ArrayView(offset=(0, 0), shape=(1, 1))
            ),
            ArrayViewPair(
                input=ArrayView(offset=(ws, 0), shape=(ws, ws)), output=ArrayView(offset=(1, 0), shape=(1, 1))
            ),
            ArrayViewPair(
                input=ArrayView(offset=(0, ws), shape=(ws, ws)), output=ArrayView(offset=(0, 1), shape=(1, 1))
            ),
            ArrayViewPair(
                input=ArrayView(offset=(ws, ws), shape=(ws, ws)), output=ArrayView(offset=(1, 1), shape=(1, 1))
            ),
        ]


@pytest.mark.parametrize('ws', [3, 5, 7])
def test_window_view_definition_non_reduce(ws):
    window = define_window(ws)
    fringes = window.get_fringes(reduce=False)

    wps = list(construct_window_views((ws + 1, ws + 1), window=ws, reduce=False))

    assert len(wps) == 4

    for wp in wps:
        assert wp in [
            ArrayViewPair(
                input=ArrayView(offset=(0, 0), shape=(ws, ws)),
                output=ArrayView(offset=(fringes[1], fringes[0]), shape=(1, 1)),
            ),
            ArrayViewPair(
                input=ArrayView(offset=(1, 0), shape=(ws, ws)),
                output=ArrayView(offset=(fringes[1] + 1, fringes[0]), shape=(1, 1)),
            ),
            ArrayViewPair(
                input=ArrayView(offset=(0, 1), shape=(ws, ws)),
                output=ArrayView(offset=(fringes[1], fringes[0] + 1), shape=(1, 1)),
            ),
            ArrayViewPair(
                input=ArrayView(offset=(1, 1), shape=(ws, ws)),
                output=ArrayView(offset=(fringes[1] + 1, fringes[0] + 1), shape=(1, 1)),
            ),
        ]


def test_tile_view_perfect_fit():
    ws = 3
    window = define_window(ws)
    fringes = window.get_fringes(reduce=False)

    raster_shape = (10, 10)
    tile_shape = (6, 6)
    tile_output_shape = (tile_shape[0] - 2 * fringes[0], tile_shape[1] - 2 * fringes[1])

    views = construct_windowed_tile_views(raster_shape, tile_shape, 3, reduce=False)

    assert len(views) == 4

    assert views[0] == ArrayViewPair(
        input=ArrayView(offset=(0, 0), shape=(6, 6)),
        output=ArrayView(offset=(fringes[1], fringes[0]), shape=tile_output_shape),
    )

    assert views[-1] == ArrayViewPair(
        input=ArrayView(
            offset=(raster_shape[1] - tile_shape[1], raster_shape[0] - tile_shape[0]),
            shape=(6, 6),
        ),
        output=ArrayView(
            offset=(fringes[1] + raster_shape[1] - tile_shape[1], fringes[0] + raster_shape[0] - tile_shape[0]),
            shape=tile_output_shape,
        ),
    )


def test_tile_view_not_fitting():
    ws = 3
    window = define_window(ws)
    fringes = window.get_fringes(reduce=False)

    raster_shape = (13, 13)
    tile_shape = (6, 6)
    tile_output_shape = (tile_shape[0] - 2 * fringes[0], tile_shape[1] - 2 * fringes[1])

    views = construct_windowed_tile_views(raster_shape, tile_shape, window, reduce=False)

    assert len(views) == 9

    assert views[0] == ArrayViewPair(
        input=ArrayView(offset=(0, 0), shape=(6, 6)),
        output=ArrayView(offset=(fringes[1], fringes[0]), shape=tile_output_shape),
    )

    assert views[-1] == ArrayViewPair(
        input=ArrayView(offset=(8, 8), shape=(5, 5)), output=ArrayView(offset=(9, 9), shape=(3, 3))
    )


def test_tiles_view_reduce_perfect_fit():
    ws = 2
    window = define_window(ws)

    raster_shape = (12, 12)
    tile_shape = (6, 6)

    views = construct_windowed_tile_views(raster_shape, tile_shape, window, reduce=True)

    assert len(views) == 4

    assert views[0] == ArrayViewPair(
        input=ArrayView(offset=(0, 0), shape=(6, 6)),
        output=ArrayView(offset=(0, 0), shape=(3, 3)),
    )

    assert views[-1] == ArrayViewPair(
        input=ArrayView(offset=(6, 6), shape=(6, 6)), output=ArrayView(offset=(3, 3), shape=(3, 3))
    )


def test_tiles_view_reduce_not_fitting():
    ws = 2
    window = define_window(ws)

    raster_shape = (10, 10)
    tile_shape = (6, 6)

    views = construct_windowed_tile_views(raster_shape, tile_shape, window, reduce=True)

    assert len(views) == 4

    assert views[0] == ArrayViewPair(
        input=ArrayView(offset=(0, 0), shape=(6, 6)),
        output=ArrayView(offset=(0, 0), shape=(3, 3)),
    )

    assert views[-1] == ArrayViewPair(
        input=ArrayView(offset=(6, 6), shape=(4, 4)), output=ArrayView(offset=(3, 3), shape=(2, 2))
    )

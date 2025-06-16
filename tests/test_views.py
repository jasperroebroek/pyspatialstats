import pytest

from pyspatialstats.views import (
    RasterView,
    RasterViewPair,
    construct_tile_views,
    construct_window_views,
)
from pyspatialstats.windows import define_window


def test_window_view_definition_errors():
    with pytest.raises(ValueError):
        construct_window_views((1, 1), window=0, reduce=False)


@pytest.mark.parametrize("ws", [3, 5, 7])
def test_window_view_definition_reduce(ws):
    wps = list(construct_window_views((ws * 2, ws * 2), window=ws, reduce=True))

    assert len(wps) == 4

    for wp in wps:
        assert wp in [
            RasterViewPair(
                input=RasterView(col_off=0, row_off=0, width=ws, height=ws),
                output=RasterView(col_off=0, row_off=0, width=1, height=1),
            ),
            RasterViewPair(
                input=RasterView(col_off=ws, row_off=0, width=ws, height=ws),
                output=RasterView(col_off=1, row_off=0, width=1, height=1),
            ),
            RasterViewPair(
                input=RasterView(col_off=0, row_off=ws, width=ws, height=ws),
                output=RasterView(col_off=0, row_off=1, width=1, height=1),
            ),
            RasterViewPair(
                input=RasterView(col_off=ws, row_off=ws, width=ws, height=ws),
                output=RasterView(col_off=1, row_off=1, width=1, height=1),
            ),
        ]


@pytest.mark.parametrize("ws", [3, 5, 7])
def test_window_view_definition_non_reduce(ws):
    window = define_window(ws)
    fringes = window.get_fringes(reduce=False)

    wps = list(construct_window_views((ws + 1, ws + 1), window=ws, reduce=False))

    assert len(wps) == 4

    for wp in wps:
        assert wp in [
            RasterViewPair(
                input=RasterView(col_off=0, row_off=0, width=ws, height=ws),
                output=RasterView(
                    col_off=fringes[1], row_off=fringes[0], width=1, height=1
                ),
            ),
            RasterViewPair(
                input=RasterView(col_off=1, row_off=0, width=ws, height=ws),
                output=RasterView(
                    col_off=fringes[1] + 1, row_off=fringes[0], width=1, height=1
                ),
            ),
            RasterViewPair(
                input=RasterView(col_off=0, row_off=1, width=ws, height=ws),
                output=RasterView(
                    col_off=fringes[1], row_off=fringes[0] + 1, width=1, height=1
                ),
            ),
            RasterViewPair(
                input=RasterView(col_off=1, row_off=1, width=ws, height=ws),
                output=RasterView(
                    col_off=fringes[1] + 1, row_off=fringes[0] + 1, width=1, height=1
                ),
            ),
        ]


def test_tile_view_perfect_fit():
    ws = 3
    window = define_window(ws)
    fringes = window.get_fringes(reduce=False)

    raster_shape = (10, 10)
    tile_shape = (6, 6)
    tile_output_shape = (tile_shape[0] - 2 * fringes[0], tile_shape[1] - 2 * fringes[1])

    views = construct_tile_views(raster_shape, tile_shape, 3, reduce=False)

    assert len(views) == 4

    assert views[0] == RasterViewPair(
        input=RasterView(col_off=0, row_off=0, width=6, height=6),
        output=RasterView(
            col_off=fringes[1],
            row_off=fringes[0],
            width=tile_output_shape[1],
            height=tile_output_shape[0],
        ),
    )

    assert views[-1] == RasterViewPair(
        input=RasterView(
            col_off=raster_shape[1] - tile_shape[1],
            row_off=raster_shape[0] - tile_shape[0],
            width=6,
            height=6,
        ),
        output=RasterView(
            col_off=fringes[1] + raster_shape[1] - tile_shape[1],
            row_off=fringes[0] + raster_shape[0] - tile_shape[0],
            width=tile_output_shape[1],
            height=tile_output_shape[0],
        ),
    )


def test_tile_view_not_fitting():
    ws = 3
    window = define_window(ws)
    fringes = window.get_fringes(reduce=False)

    raster_shape = (13, 13)
    tile_shape = (6, 6)
    tile_output_shape = (tile_shape[0] - 2 * fringes[0], tile_shape[1] - 2 * fringes[1])

    views = construct_tile_views(raster_shape, tile_shape, window, reduce=False)

    assert len(views) == 9

    assert views[0] == RasterViewPair(
        input=RasterView(col_off=0, row_off=0, width=6, height=6),
        output=RasterView(
            col_off=fringes[1],
            row_off=fringes[0],
            width=tile_output_shape[1],
            height=tile_output_shape[0],
        ),
    )

    assert views[-1] == RasterViewPair(
        input=RasterView(col_off=8, row_off=8, width=5, height=5),
        output=RasterView(col_off=9, row_off=9, width=3, height=3),
    )


def test_tiles_view_reduce_perfect_fit():
    ws = 2
    window = define_window(ws)

    raster_shape = (12, 12)
    tile_shape = (6, 6)

    views = construct_tile_views(raster_shape, tile_shape, window, reduce=True)

    assert len(views) == 4

    assert views[0] == RasterViewPair(
        input=RasterView(col_off=0, row_off=0, width=6, height=6),
        output=RasterView(col_off=0, row_off=0, width=3, height=3),
    )

    assert views[-1] == RasterViewPair(
        input=RasterView(col_off=6, row_off=6, width=6, height=6),
        output=RasterView(col_off=3, row_off=3, width=3, height=3),
    )


def test_tiles_view_reduce_not_fitting():
    ws = 2
    window = define_window(ws)

    raster_shape = (10, 10)
    tile_shape = (6, 6)

    views = construct_tile_views(raster_shape, tile_shape, window, reduce=True)

    assert len(views) == 4

    assert views[0] == RasterViewPair(
        input=RasterView(col_off=0, row_off=0, width=6, height=6),
        output=RasterView(col_off=0, row_off=0, width=3, height=3),
    )

    assert views[-1] == RasterViewPair(
        input=RasterView(col_off=6, row_off=6, width=4, height=4),
        output=RasterView(col_off=3, row_off=3, width=2, height=2),
    )

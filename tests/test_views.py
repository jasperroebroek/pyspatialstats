import pytest

from pyspatialstats.views import RasterView, RasterViewPair, construct_views


def test_view_definition_errors():
    with pytest.raises(ValueError):
        construct_views((1, 1), view_shape=(0, 0), reduce=False)


def test_view_definition_reduce():
    assert list(construct_views((5, 5), view_shape=(5, 5), reduce=True)) == [
        RasterViewPair(
            input=RasterView(col_off=0, row_off=0, width=5, height=5),
            output=RasterView(col_off=0, row_off=0, width=1, height=1),
        )
    ]


def test_view_definition_non_reduce():
    for wp in list(construct_views((2, 2), view_shape=(1, 1), reduce=False)):
        assert wp in [
            RasterViewPair(
                input=RasterView(col_off=0, row_off=0, width=1, height=1),
                output=RasterView(col_off=0, row_off=0, width=1, height=1),
            ),
            RasterViewPair(
                input=RasterView(col_off=1, row_off=0, width=1, height=1),
                output=RasterView(col_off=1, row_off=0, width=1, height=1),
            ),
            RasterViewPair(
                input=RasterView(col_off=0, row_off=1, width=1, height=1),
                output=RasterView(col_off=0, row_off=1, width=1, height=1),
            ),
            RasterViewPair(
                input=RasterView(col_off=1, row_off=1, width=1, height=1),
                output=RasterView(col_off=1, row_off=1, width=1, height=1),
            ),
        ]

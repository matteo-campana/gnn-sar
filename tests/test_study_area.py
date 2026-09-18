from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from gnn_sar.data.raster import RasterGrid
from gnn_sar.data.segmentation import NodeGeometries
from gnn_sar.data.sentinel import PeriodGrid
from gnn_sar.data.study_area import (
    SourceStack,
    align_annual_labels,
    build_study_area,
    composite_sources,
    load_study_area,
)
from gnn_sar.data.targets import IGNORE_INDEX


@pytest.fixture
def geometries() -> NodeGeometries:
    return NodeGeometries.regular_grid(8, 8, block=4)


def _stack(dates, height=8, width=8, channels=1, fill=1.0, names=None, **kwargs):
    values = np.full((len(dates), channels, height, width), fill, dtype=float)
    return SourceStack(
        values=values,
        dates=dates,
        names=names or [f"c{i}" for i in range(channels)],
        **kwargs,
    )


def test_two_sources_acquired_on_different_dates_become_channels_of_one_series():
    radar = _stack([datetime(2023, 1, 5), datetime(2023, 2, 5)], fill=1.0, names=["vv"])
    optical = _stack([datetime(2023, 2, 20)], fill=2.0, names=["ndvi"])
    periods = PeriodGrid.covering([datetime(2023, 1, 1), datetime(2023, 2, 28)], "month")

    composites, names = composite_sources([radar, optical], periods)

    assert names == ["vv", "ndvi"]
    assert composites.shape == (2, 2, 8, 8)
    # January has no optical acquisition, so the February one is back-filled.
    assert composites[0, 1].tolist() == composites[1, 1].tolist()


def test_the_composites_of_one_period_are_reduced_together(geometries):
    dates = [datetime(2023, 1, 5), datetime(2023, 1, 25)]
    source = _stack(dates, names=["vv"])
    source.values[0] = 10.0
    source.values[1] = 20.0

    area = build_study_area([source], geometries, period="month", statistics=("mean",))

    assert area.graph.n_time == 1
    assert area.graph.features[0, :, 0] == pytest.approx(15.0)  # the median of the two


def test_the_feature_tensor_names_every_channel_and_statistic(geometries):
    source = _stack([datetime(2023, 1, 5)], channels=2, names=["vv", "vh"])

    area = build_study_area([source], geometries, statistics=("mean", "std"))

    assert area.channel_names == ["vv_mean", "vv_std", "vh_mean", "vh_std"]
    assert area.graph.n_features == 4


def test_a_masked_source_is_pooled_over_the_pixels_that_were_observed(geometries):
    source = _stack([datetime(2023, 1, 5)], names=["ndvi"], ignore_nan=True)
    source.values[0, 0, :4, :2] = np.nan  # half of node 0 under cloud
    source.values[0, 0, :4, 2:4] = 3.0

    area = build_study_area([source], geometries, statistics=("mean",))

    assert area.graph.features[0, 0, 0] == pytest.approx(3.0)


def test_a_node_never_observed_is_refused_rather_than_trained_on(geometries):
    source = _stack([datetime(2023, 1, 5)], names=["ndvi"], ignore_nan=True)
    source.values[0, 0, :4, :4] = np.nan  # node 0, every pixel, every composite

    with pytest.raises(ValueError, match="never observed"):
        build_study_area([source], geometries)


def test_annual_labels_are_laid_out_on_the_years_the_composites_span():
    periods = PeriodGrid.covering([datetime(2022, 11, 1), datetime(2023, 2, 1)], "month")

    labels = align_annual_labels({2023: np.array([1, 0])}, periods, n_nodes=2)

    assert labels.shape == (2, 2)
    assert labels[0].tolist() == [IGNORE_INDEX, IGNORE_INDEX]  # 2022 has no cartography
    assert labels[1].tolist() == [1, 0]


def test_labels_for_a_year_outside_the_series_are_an_error():
    periods = PeriodGrid.covering([datetime(2023, 1, 1)], "month")

    with pytest.raises(ValueError, match="do not cover"):
        align_annual_labels({2021: np.zeros(2)}, periods, n_nodes=2)


def test_annual_labels_of_the_wrong_shape_say_what_was_expected(geometries):
    source = _stack([datetime(2023, 1, 5)], names=["vv"])

    with pytest.raises(ValueError, match=r"annual_labels must be \(1, 4\)"):
        build_study_area([source], geometries, annual_labels=np.zeros((2, 4), dtype=int))


def test_the_graph_carries_the_year_each_composite_is_supervised_by(geometries):
    dates = [datetime(2022, 12, 5), datetime(2023, 1, 5)]
    source = _stack(dates, names=["vv"])

    area = build_study_area(
        [source], geometries, annual_labels={2023: np.ones(4, dtype=int)}
    )

    assert area.graph.year_index.tolist() == [0, 1]
    assert area.year_labels == [2022, 2023]


def test_a_study_area_survives_a_round_trip_through_disk(tmp_path, geometries):
    source = _stack([datetime(2023, 1, 5), datetime(2023, 2, 5)], channels=2)
    grid = RasterGrid(8, 8, (300000.0, 10.0, 0.0, 4600000.0, 0.0, -10.0), "EPSG:32633")
    area = build_study_area(
        [source],
        geometries,
        grid=grid,
        corridor=np.array([True, False, True, False]),
        annual_labels={2023: np.array([1, 0, 0, 0])},
        metadata={"note": "round trip"},
    )

    restored = load_study_area(area.save(tmp_path / "area.npz"))

    assert np.allclose(restored.graph.features, area.graph.features)
    assert restored.graph.annual_labels.tolist() == area.graph.annual_labels.tolist()
    assert restored.channel_names == area.channel_names
    assert restored.periods == area.periods
    assert restored.grid == grid
    assert restored.corridor.tolist() == [True, False, True, False]
    assert restored.metadata == {"note": "round trip"}


def test_sources_on_different_rasters_are_refused(geometries):
    small = _stack([datetime(2023, 1, 5)], height=4, width=4)
    large = _stack([datetime(2023, 1, 5)])

    with pytest.raises(ValueError, match="different rasters"):
        composite_sources(
            [small, large], PeriodGrid.covering([datetime(2023, 1, 5)], "month")
        )


def test_the_rasters_must_match_the_geometries(geometries):
    source = _stack([datetime(2023, 1, 5)], height=4, width=4)

    with pytest.raises(ValueError, match="do not match the geometries"):
        build_study_area([source], geometries)

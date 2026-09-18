from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.labels import (
    annual_land_take_labels,
    corridor_from_raster,
    node_fraction,
    node_state_from_rasters,
)
from gnn_sar.data.segmentation import NodeGeometries
from gnn_sar.data.targets import IGNORE_INDEX


@pytest.fixture
def geometries() -> NodeGeometries:
    """Four 2x2 nodes over a 4x4 raster."""
    return NodeGeometries.regular_grid(4, 4, block=2)


def test_the_fraction_is_measured_inside_each_node(geometries):
    mask = np.zeros((4, 4))
    mask[0, :2] = 1  # half of node 0

    fraction, coverage = node_fraction(mask, geometries)

    assert fraction.tolist() == [0.5, 0.0, 0.0, 0.0]
    assert coverage.tolist() == [1.0] * 4


def test_nodata_lowers_the_coverage_instead_of_counting_as_zero(geometries):
    mask = np.zeros((4, 4))
    mask[0, :2] = 1
    valid = np.ones((4, 4), dtype=bool)
    valid[1, :2] = False  # the unsealed half of node 0 was not observed

    fraction, coverage = node_fraction(mask, geometries, valid)

    assert fraction[0] == pytest.approx(1.0)
    assert coverage[0] == pytest.approx(0.5)


def test_a_node_below_the_coverage_floor_has_no_state(geometries):
    mask = np.zeros((4, 4))
    valid = np.ones((4, 4), dtype=bool)
    valid[:2, :2] = False  # node 0 entirely unobserved

    state = node_state_from_rasters([mask], geometries, valid_masks=[valid])

    assert state[0, 0] == IGNORE_INDEX
    assert state[0, 1:].tolist() == [0, 0, 0]


def test_a_node_counts_as_sealed_by_majority(geometries):
    mask = np.zeros((4, 4))
    mask[:2, 0] = 1  # half of node 0
    mask[:2, 2:] = 1  # all of node 1

    state = node_state_from_rasters([mask], geometries, threshold=0.5)

    assert state[0, :2].tolist() == [1, 1]

    stricter = node_state_from_rasters([mask], geometries, threshold=0.75)

    assert stricter[0, :2].tolist() == [0, 1]


def test_the_label_marks_the_year_of_the_transition_and_not_the_years_after():
    state = np.array([[0], [1], [1]])

    assert annual_land_take_labels(state)[:, 0].tolist() == [0, 1, 0]


def test_land_already_sealed_before_the_series_is_excluded_by_default():
    state = np.array([[1], [1], [1]])

    assert annual_land_take_labels(state)[:, 0].tolist() == [IGNORE_INDEX] * 3


def test_pre_existing_land_can_instead_be_scored_as_a_negative():
    state = np.array([[1], [1], [1]])

    labels = annual_land_take_labels(state, pre_existing="negative")

    assert labels[:, 0].tolist() == [0, 0, 0]


def test_a_transition_hidden_by_a_gap_is_unknown_in_every_year_it_could_be_in():
    # Unobserved in the middle year, sealed in the last: the transition may have
    # happened in either, so neither may be called a positive or a negative.
    state = np.array([[0], [IGNORE_INDEX], [1]])

    assert annual_land_take_labels(state)[:, 0].tolist() == [0, IGNORE_INDEX, IGNORE_INDEX]


def test_an_observed_negative_closes_the_gap_again():
    state = np.array([[0], [IGNORE_INDEX], [0], [1]])

    labels = annual_land_take_labels(state)

    assert labels[:, 0].tolist() == [0, IGNORE_INDEX, 0, 1]


def test_a_node_never_sealed_is_negative_in_every_year():
    state = np.zeros((4, 1), dtype=int)

    assert annual_land_take_labels(state)[:, 0].tolist() == [0, 0, 0, 0]


def test_at_most_one_land_take_per_node():
    state = np.array([[0, 0], [1, 0], [1, 1]])

    labels = annual_land_take_labels(state)

    assert (labels == 1).sum(axis=0).tolist() == [1, 1]


def test_the_corridor_keeps_any_node_the_road_touches(geometries):
    roads = np.zeros((4, 4))
    roads[0, 0] = 1  # a quarter of node 0

    assert corridor_from_raster(roads, geometries).tolist() == [True, False, False, False]


def test_the_corridor_can_demand_a_minimum_share(geometries):
    roads = np.zeros((4, 4))
    roads[0, 0] = 1  # 25% of node 0
    roads[:2, 2:] = 1  # 100% of node 1

    corridor = corridor_from_raster(roads, geometries, min_fraction=0.5)

    assert corridor.tolist() == [False, True, False, False]

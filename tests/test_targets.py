from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.targets import (
    IGNORE_INDEX,
    annual_to_horizon_target,
    onset_to_state,
    state_to_horizon_target,
    state_to_onset,
)


def test_onset_to_state_is_monotone_and_starts_at_the_onset():
    state = onset_to_state(np.array([2, -1, 0]), n_time=4)

    assert state[:, 0].tolist() == [0, 0, 1, 1]
    assert state[:, 1].tolist() == [0, 0, 0, 0]
    assert state[:, 2].tolist() == [1, 1, 1, 1]


def test_state_to_onset_inverts_onset_to_state():
    onset = np.array([3, -1, 0, 1])

    assert state_to_onset(onset_to_state(onset, 5)).tolist() == onset.tolist()


def test_horizon_target_flags_only_the_composites_before_a_transition():
    state = onset_to_state(np.array([3]), n_time=6)

    target = state_to_horizon_target(state, horizon=2)

    # t=1 and t=2 see the onset at t=3 inside their 2-composite horizon; t=0
    # does not, and the last two composites have no observable future.
    assert target[:, 0].tolist() == [0, 1, 1, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_horizon_target_ignores_nodes_that_are_already_sealed():
    state = onset_to_state(np.array([0]), n_time=4)

    target = state_to_horizon_target(state, horizon=1)

    assert set(target[:, 0].tolist()) == {IGNORE_INDEX}


def test_horizon_target_rejects_a_zero_horizon():
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        state_to_horizon_target(np.zeros((3, 1), dtype=int), horizon=0)


def test_weak_horizon_target_spans_the_whole_land_take_year():
    annual = np.array([[0], [1]])
    year_index = np.array([0, 0, 1, 1])

    target = annual_to_horizon_target(annual, year_index, horizon=1)

    # The annual label cannot localise the transition, so the target is 1 from
    # the composite preceding the land-take year onwards.
    assert target[:, 0].tolist() == [0, 1, 1, IGNORE_INDEX]


def test_weak_horizon_target_propagates_unknown_years():
    annual = np.array([[0], [IGNORE_INDEX]])
    year_index = np.array([0, 1])

    target = annual_to_horizon_target(annual, year_index, horizon=1)

    assert target[0, 0] == IGNORE_INDEX

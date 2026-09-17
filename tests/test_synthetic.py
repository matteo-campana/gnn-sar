from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.synthetic import make_synthetic_scenario


def test_the_same_seed_gives_the_same_scenario():
    first = make_synthetic_scenario(height=16, width=16, n_time=6, seed=3)
    second = make_synthetic_scenario(height=16, width=16, n_time=6, seed=3)

    assert np.array_equal(first.onset, second.onset)
    assert np.allclose(first.graph.features, second.graph.features)


def test_different_seeds_give_different_scenarios():
    first = make_synthetic_scenario(height=16, width=16, n_time=6, seed=3)
    second = make_synthetic_scenario(height=16, width=16, n_time=6, seed=4)

    assert not np.array_equal(first.onset, second.onset)


def test_the_graph_carries_annual_labels_and_withholds_the_exact_dates(scenario):
    assert scenario.graph.annual_labels is not None
    assert scenario.graph.node_labels is None, "weak supervision is the default regime"
    assert scenario.with_exact_labels().node_labels is not None


def test_annual_labels_mark_exactly_the_year_of_each_onset(scenario):
    graph = scenario.graph
    sealed = np.flatnonzero(scenario.onset >= 0)

    for node in sealed:
        year = graph.year_index[scenario.onset[node]]
        assert graph.annual_labels[year, node] == 1
        assert graph.annual_labels[:, node].sum() == 1

    never = scenario.onset < 0
    assert graph.annual_labels[:, never].sum() == 0


def test_the_sealed_state_never_reverts(scenario):
    assert np.all(np.diff(scenario.state_labels, axis=0) >= 0)


def test_sealing_is_relational_rather_than_uniformly_random(scenario):
    """New sealing clusters next to existing sealing — the project's hypothesis."""
    edges = scenario.graph.spatial_edge_index
    sealed = scenario.onset >= 0

    neighbour_sealed = sealed[edges[0]]
    share_next_to_sealed = neighbour_sealed[sealed[edges[1]]].mean()
    share_next_to_natural = neighbour_sealed[~sealed[edges[1]]].mean()

    assert share_next_to_sealed > 2 * share_next_to_natural


def test_the_sealed_signature_is_visible_in_the_backscatter(scenario):
    final = scenario.graph.features[-1]
    sealed = scenario.onset >= 0

    # Channel 0 is the mean VV in dB: built surfaces backscatter more.
    assert final[sealed, 0].mean() > final[~sealed, 0].mean() + 3.0
    # Channel 4 is the mean coherence: built surfaces are stable.
    assert final[sealed, 4].mean() > final[~sealed, 4].mean()


def test_lowering_the_contrast_shrinks_the_separation():
    easy = make_synthetic_scenario(height=48, width=48, n_time=12, contrast=1.0, seed=1)
    hard = make_synthetic_scenario(height=48, width=48, n_time=12, contrast=0.2, seed=1)

    def separation(scenario) -> float:
        final = scenario.graph.features[-1]
        sealed = scenario.onset >= 0
        return float(final[sealed, 0].mean() - final[~sealed, 0].mean())

    # The process is identical — only the radiometric signature shrinks.
    assert np.array_equal(easy.onset, hard.onset)
    assert separation(hard) < 0.5 * separation(easy)


def test_the_corridor_seals_faster_than_the_rest_of_the_domain():
    scenario = make_synthetic_scenario(
        height=64, width=64, n_time=24, corridor_hazard=0.15, base_hazard=0.0, seed=2
    )
    sealed = scenario.onset >= 0

    assert sealed[scenario.corridor].mean() > sealed[~scenario.corridor].mean()


def test_the_stack_has_one_frame_per_composite(scenario):
    assert scenario.stack.shape == (scenario.graph.n_time, 3, 32, 32)
    assert np.all((scenario.stack[:, 2] >= 0.0) & (scenario.stack[:, 2] <= 1.0))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"contrast": 0.0}, "contrast must be > 0"),
        ({"looks": 0}, "looks must be >= 1"),
        ({"periods_per_year": 0}, "periods_per_year must be >= 1"),
        ({"construction_periods": -1}, "construction_periods must be >= 0"),
        ({"heterogeneity_db": -1.0}, "heterogeneity_db must be >= 0"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_synthetic_scenario(height=16, width=16, n_time=4, **kwargs)

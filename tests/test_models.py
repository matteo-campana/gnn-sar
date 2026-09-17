from __future__ import annotations

import numpy as np
import pytest
import torch

from gnn_sar.config import ModelConfig
from gnn_sar.models.baselines import MLPBaseline, RandomForestBaseline, TemporalBaseline
from gnn_sar.models.layers import replicate_edges
from gnn_sar.models.stgnn import SpatioTemporalGNN, sealed_probability

N_TIME, N_NODES, N_FEATURES = 6, 12, 4


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    features = torch.randn(N_TIME, N_NODES, N_FEATURES)
    edge_index = torch.tensor(
        [[i for i in range(N_NODES - 1)], [i + 1 for i in range(N_NODES - 1)]]
    )
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return features, edge_index


def _config(**overrides) -> ModelConfig:
    base = {"in_channels": N_FEATURES, "hidden_channels": 16, "dropout": 0.0}
    return ModelConfig(**{**base, **overrides})


def test_replicate_edges_offsets_each_frame():
    edges = torch.tensor([[0, 1], [1, 0]])

    replicated = replicate_edges(edges, n_nodes=2, n_time=3)

    assert replicated.shape == (2, 6)
    assert replicated[:, 2:4].tolist() == [[2, 3], [3, 2]]


def test_replicate_edges_rejects_a_malformed_edge_index():
    with pytest.raises(ValueError, match=r"must be \(2, E\)"):
        replicate_edges(torch.zeros(3, 4, dtype=torch.long), 2, 2)


@pytest.mark.parametrize("spatial_conv", ["sage", "gat"])
@pytest.mark.parametrize("temporal_encoder", ["lstm", "attention"])
def test_forward_produces_one_decision_per_node_and_composite(
    batch, spatial_conv, temporal_encoder
):
    features, edge_index = batch
    model = SpatioTemporalGNN(
        _config(spatial_conv=spatial_conv, temporal_encoder=temporal_encoder)
    )

    output = model(features, edge_index)

    assert output.onset_logits.shape == (N_TIME, N_NODES)
    assert output.horizon_logits.shape == (N_TIME, N_NODES)
    assert output.embeddings.shape == (N_TIME, N_NODES, 16)
    assert torch.isfinite(output.onset_logits).all()


def test_sealed_probability_is_non_decreasing_by_construction():
    # Arbitrary logits, including a strong "nothing is happening" tail.
    logits = torch.tensor([[2.0, -8.0], [-6.0, 3.0], [-1.0, -9.0]])

    probability = sealed_probability(logits)

    assert torch.all(probability.diff(dim=0) >= -1e-7)
    assert torch.all((probability >= 0) & (probability <= 1))


def test_sealed_probability_equals_one_minus_the_survival_product():
    logits = torch.randn(5, 3)
    expected = 1.0 - torch.cumprod(1.0 - torch.sigmoid(logits), dim=0)

    assert torch.allclose(sealed_probability(logits), expected, atol=1e-6)


@pytest.mark.parametrize("temporal_encoder", ["lstm", "attention"])
def test_the_past_never_depends_on_the_future(batch, temporal_encoder):
    features, edge_index = batch
    model = SpatioTemporalGNN(_config(temporal_encoder=temporal_encoder)).eval()

    with torch.no_grad():
        before = model(features, edge_index).onset_logits
        perturbed = features.clone()
        perturbed[-1] += 50.0
        after = model(perturbed, edge_index).onset_logits

    assert torch.allclose(before[:-1], after[:-1], atol=1e-5)
    assert not torch.allclose(before[-1], after[-1], atol=1e-5)


def test_the_model_is_inductive_over_different_node_sets(batch):
    features, edge_index = batch
    model = SpatioTemporalGNN(_config()).eval()

    smaller = torch.randn(N_TIME, 5, N_FEATURES)
    smaller_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

    with torch.no_grad():
        # Same weights, a study area of a different size: what OS4 needs.
        assert model(features, edge_index).onset_logits.shape == (N_TIME, N_NODES)
        assert model(smaller, smaller_edges).onset_logits.shape == (N_TIME, 5)


def test_an_isolated_node_still_gets_a_prediction():
    model = SpatioTemporalGNN(_config()).eval()
    features = torch.randn(3, 4, N_FEATURES)

    with torch.no_grad():
        output = model(features, torch.zeros((2, 0), dtype=torch.long))

    assert torch.isfinite(output.onset_logits).all()


def test_forward_rejects_a_wrong_channel_count(batch):
    _, edge_index = batch
    model = SpatioTemporalGNN(_config())

    with pytest.raises(ValueError, match="expects 4 input channels"):
        model(torch.randn(N_TIME, N_NODES, 3), edge_index)


def test_forward_rejects_edges_outside_the_node_set(batch):
    features, _ = batch
    model = SpatioTemporalGNN(_config())

    with pytest.raises(ValueError, match="outside the graph"):
        model(features, torch.tensor([[0], [N_NODES + 3]]))


def test_predict_onset_reports_minus_one_when_nothing_is_detected(batch):
    features, edge_index = batch
    model = SpatioTemporalGNN(_config())

    onset = model.predict_onset(features, edge_index, threshold=1.01)

    assert onset.shape == (N_NODES,)
    assert torch.all(onset == -1)


def test_the_predictive_head_can_be_switched_off(batch):
    features, edge_index = batch
    model = SpatioTemporalGNN(_config(predictive_head=False))

    assert model(features, edge_index).horizon_logits is None


@pytest.mark.parametrize("baseline", [TemporalBaseline, MLPBaseline])
def test_baselines_ignore_the_graph_entirely(batch, baseline):
    features, edge_index = batch
    model = baseline(_config()).eval()

    with torch.no_grad():
        with_edges = model(features, edge_index).onset_logits
        without_edges = model(features, None).onset_logits
        scrambled = model(features, edge_index.flip(0)).onset_logits

    assert torch.allclose(with_edges, without_edges)
    assert torch.allclose(with_edges, scrambled)


def test_the_mlp_baseline_has_no_temporal_context():
    model = MLPBaseline(_config()).eval()
    features = torch.randn(4, 3, N_FEATURES)

    with torch.no_grad():
        before = model(features).onset_logits
        perturbed = features.clone()
        perturbed[0] += 10.0
        after = model(perturbed).onset_logits

    assert torch.allclose(before[1:], after[1:], atol=1e-6)


def test_random_forest_summarises_each_year(scenario):
    graph = scenario.graph

    summaries = RandomForestBaseline.year_features(graph.features, graph.year_index)

    assert summaries.shape == (graph.n_years, graph.n_nodes, 4 * graph.n_features)


def test_random_forest_predicts_one_probability_per_year_and_node(scenario):
    graph = scenario.graph
    forest = RandomForestBaseline(n_estimators=10).fit(
        graph.features, graph.year_index, graph.annual_labels
    )

    proba = forest.predict_proba(graph.features, graph.year_index)

    assert proba.shape == (graph.n_years, graph.n_nodes)
    assert np.all((proba >= 0) & (proba <= 1))


def test_random_forest_refuses_a_single_class_split(scenario):
    graph = scenario.graph

    with pytest.raises(ValueError, match="single class"):
        RandomForestBaseline().fit(
            graph.features,
            graph.year_index,
            np.zeros_like(graph.annual_labels),
        )


def test_random_forest_refuses_to_predict_before_fitting(scenario):
    with pytest.raises(RuntimeError, match="must be fitted"):
        RandomForestBaseline().predict_proba(
            scenario.graph.features, scenario.graph.year_index
        )

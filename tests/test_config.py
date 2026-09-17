from __future__ import annotations

import pytest

from gnn_sar.config import (
    ExperimentConfig,
    GraphConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)


def test_a_config_survives_a_json_round_trip(tmp_path):
    config = ExperimentConfig(
        graph=GraphConfig(n_neighbors=7),
        model=ModelConfig(hidden_channels=32, temporal_encoder="attention"),
        training=TrainingConfig(max_epochs=3, loss=LossConfig(aggregation="max")),
    )
    path = tmp_path / "experiment.json"

    config.to_json(path)

    assert ExperimentConfig.from_json(path) == config


def test_an_unknown_key_is_reported_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown keys for GraphConfig"):
        ExperimentConfig.from_dict({"graph": {"n_neighbours": 4}})


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: GraphConfig(temporal_order=0), "temporal_order must be >= 1"),
        (lambda: GraphConfig(n_neighbors=-1), "n_neighbors must be >= 0"),
        (lambda: GraphConfig(connectivity_radius=0.0), "connectivity_radius"),
        (lambda: ModelConfig(in_channels=0), "must be >= 1"),
        (lambda: ModelConfig(dropout=1.0), r"dropout must be in \[0, 1\)"),
        (lambda: ModelConfig(spatial_layers=0), "layer counts must be >= 1"),
        (
            lambda: ModelConfig(spatial_conv="gat", hidden_channels=10, gat_heads=4),
            "divisible by gat_heads",
        ),
        (lambda: ModelConfig(prediction_horizon=0), "prediction_horizon must be >= 1"),
        (lambda: LossConfig(smoothness_weight=-1.0), "loss weights must be >= 0"),
        (lambda: LossConfig(logsumexp_temperature=0.0), "temperature must be > 0"),
    ],
)
def test_invalid_settings_fail_at_construction(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_a_disabled_predictive_head_ignores_the_horizon():
    assert ModelConfig(predictive_head=False, prediction_horizon=0).prediction_horizon == 0

from __future__ import annotations

import pytest

from gnn_sar.config import (
    PRECISIONS,
    ExperimentConfig,
    GraphConfig,
    LossConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)


def test_a_config_survives_a_json_round_trip(tmp_path):
    config = ExperimentConfig(
        graph=GraphConfig(n_neighbors=7),
        model=ModelConfig(hidden_channels=32, temporal_encoder="attention"),
        training=TrainingConfig(
            max_epochs=3,
            precision="bf16-mixed",
            accelerator="gpu",
            loss=LossConfig(aggregation="max"),
        ),
        tracking=TrackingConfig(
            experiment_name="viterbo", tracking_uri="sqlite:///runs.db", log_maps=False
        ),
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
        (lambda: TrainingConfig(max_epochs=0), "max_epochs must be >= 1"),
        (lambda: TrainingConfig(learning_rate=0.0), "learning_rate must be > 0"),
        (lambda: TrainingConfig(precision="fp8"), "precision must be one of"),
        (
            lambda: TrainingConfig(decision_threshold=1.5),
            r"decision_threshold must be in \[0, 1\]",
        ),
    ],
)
def test_invalid_settings_fail_at_construction(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_a_disabled_predictive_head_ignores_the_horizon():
    assert ModelConfig(predictive_head=False, prediction_horizon=0).prediction_horizon == 0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_every_advertised_precision_is_accepted(precision):
    assert TrainingConfig(precision=precision).precision == precision


def test_tracking_is_on_by_default_with_no_uri_committed_to():
    """The default records runs, but decides *where* only at run time."""
    tracking = ExperimentConfig().tracking

    assert tracking.enabled
    assert tracking.tracking_uri is None
    assert tracking.experiment_name == "gnn-sar"


def test_the_default_hardware_settings_are_portable():
    """'auto' on a single device is what makes a GPU run need no extra flag."""
    training = ExperimentConfig().training

    assert (training.accelerator, training.devices) == ("auto", 1)
    assert training.precision == "32-true"

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from gnn_sar.config import LossConfig, ModelConfig, TrainingConfig
from gnn_sar.data.targets import IGNORE_INDEX
from gnn_sar.models.baselines import TemporalBaseline
from gnn_sar.models.stgnn import SpatioTemporalGNN
from gnn_sar.training.cli import build_parser, run_demo
from gnn_sar.training.datamodule import SpatioTemporalGraphDataModule
from gnn_sar.training.lightning_module import SoilSealingModule


@pytest.fixture
def datamodule(scenario) -> SpatioTemporalGraphDataModule:
    module = SpatioTemporalGraphDataModule(scenario.graph, prediction_horizon=2)
    module.setup()
    return module


def _trainer(max_epochs: int = 3) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def test_the_datamodule_needs_annual_labels(scenario):
    from gnn_sar.data.graph import SpatioTemporalGraph

    bare = SpatioTemporalGraph(
        features=scenario.graph.features,
        spatial_edge_index=scenario.graph.spatial_edge_index,
        spatial_edge_type=scenario.graph.spatial_edge_type,
        geometries=scenario.geometries,
    )

    with pytest.raises(ValueError, match="supervision signal"):
        SpatioTemporalGraphDataModule(bare)


def test_every_split_serves_the_same_graph_and_a_different_mask(datamodule):
    batches = {split: datamodule.batch(split) for split in ("train", "val", "test")}

    masks = torch.stack([batch["node_mask"] for batch in batches.values()])
    assert torch.all(masks.sum(dim=0) == 1)
    for batch in batches.values():
        assert torch.equal(
            batch["spatial_edge_index"], batches["train"]["spatial_edge_index"]
        )


def test_standardisation_is_fitted_on_the_training_nodes_only(scenario, datamodule):
    train_mask = datamodule.masks["train"]
    raw = scenario.graph.features

    assert datamodule.standardizer is not None
    assert np.allclose(
        datamodule.standardizer.mean, raw[:, train_mask].reshape(-1, raw.shape[2]).mean(0)
    )
    standardised = datamodule.batch("train")["features"][:, torch.tensor(train_mask)]
    assert standardised.reshape(-1, raw.shape[2]).mean(0).abs().max() < 1e-4


def test_standardisation_can_be_switched_off(scenario):
    module = SpatioTemporalGraphDataModule(scenario.graph, standardize=False)
    module.setup()

    assert module.standardizer is None
    assert torch.allclose(
        module.batch("train")["features"],
        torch.as_tensor(scenario.graph.features, dtype=torch.float32),
    )


def test_the_weak_regime_carries_no_exact_labels(datamodule):
    batch = datamodule.batch("train")

    assert batch["state_labels"] is None
    assert set(np.unique(batch["horizon_target"].numpy())) <= {IGNORE_INDEX, 0, 1}


def test_exact_labels_are_passed_through_when_they_exist(scenario):
    module = SpatioTemporalGraphDataModule(scenario.with_exact_labels())
    module.setup()

    batch = module.batch("train")
    assert batch["state_labels"] is not None
    assert batch["state_labels"].shape == (scenario.graph.n_time, scenario.graph.n_nodes)


def test_the_dataloader_yields_the_graph_without_an_extra_batch_axis(datamodule):
    batch = next(iter(datamodule.train_dataloader()))

    assert batch["features"].shape == (
        datamodule.graph.n_time,
        datamodule.graph.n_nodes,
        datamodule.n_features,
    )


def test_training_reduces_the_objective(scenario, datamodule):
    pl.seed_everything(0, workers=True)
    module = SoilSealingModule(
        model=SpatioTemporalGNN(
            ModelConfig(in_channels=datamodule.n_features, hidden_channels=16)
        ),
        training_config=TrainingConfig(learning_rate=5e-3),
    )
    batch = datamodule.batch("train")

    with torch.no_grad():
        before = module._loss(batch)["total"].item()
    _trainer(max_epochs=25).fit(module, datamodule=datamodule)
    with torch.no_grad():
        after = module._loss(batch)["total"].item()

    assert after < before


def test_a_full_fit_test_cycle_reports_the_annual_metrics(datamodule):
    pl.seed_everything(0, workers=True)
    module = SoilSealingModule(
        model=TemporalBaseline(
            ModelConfig(in_channels=datamodule.n_features, hidden_channels=16)
        )
    )
    trainer = _trainer()

    trainer.fit(module, datamodule=datamodule)
    metrics = trainer.test(module, datamodule=datamodule, verbose=False)[0]

    for name in ("accuracy", "f1", "iou", "kappa"):
        assert f"test/annual_{name}" in metrics
        assert 0.0 <= metrics[f"test/annual_{name}"] <= 1.0
    # No exact dates in the weak regime, so no sub-annual claim is made.
    assert not any(key.startswith("test/subannual") for key in metrics)


def test_sub_annual_metrics_appear_only_with_exact_labels(scenario):
    pl.seed_everything(0, workers=True)
    module = SpatioTemporalGraphDataModule(scenario.with_exact_labels())
    module.setup()
    lightning_module = SoilSealingModule(
        model=TemporalBaseline(
            ModelConfig(in_channels=module.n_features, hidden_channels=16)
        )
    )
    trainer = _trainer(max_epochs=2)

    trainer.fit(lightning_module, datamodule=module)
    metrics = trainer.test(lightning_module, datamodule=module, verbose=False)[0]

    assert "test/subannual_f1" in metrics
    assert "test/delay_detection_rate" in metrics


def test_predict_maps_returns_a_sub_annual_product(datamodule):
    module = SoilSealingModule(
        model=SpatioTemporalGNN(
            ModelConfig(in_channels=datamodule.n_features, hidden_channels=16)
        )
    )

    maps = module.predict_maps(datamodule.batch("test"))

    n_time, n_nodes = datamodule.graph.n_time, datamodule.graph.n_nodes
    assert maps["sealed_probability"].shape == (n_time, n_nodes)
    assert maps["onset"].shape == (n_nodes,)
    assert maps["early_warning"].shape == (n_time, n_nodes)
    assert np.all(np.diff(maps["sealed_probability"], axis=0) >= -1e-6)


def test_predicted_maps_can_be_painted_back_onto_the_raster(datamodule, scenario):
    maps = SoilSealingModule(
        model=SpatioTemporalGNN(
            ModelConfig(in_channels=datamodule.n_features, hidden_channels=16)
        )
    ).predict_maps(datamodule.batch("test"))

    raster = scenario.geometries.rasterise(maps["sealed_probability"][-1])

    assert raster.shape == scenario.geometries.shape


def test_the_loss_config_reaches_the_module():
    module = SoilSealingModule(
        model_config=ModelConfig(in_channels=3),
        training_config=TrainingConfig(loss=LossConfig(aggregation="max")),
    )

    assert module.criterion.config.aggregation == "max"


def test_the_cli_parses_a_demo_invocation():
    args = build_parser().parse_args(["demo", "--models", "mlp", "--max-epochs", "2"])

    assert args.command == "demo"
    assert args.models == ["mlp"]
    assert args.max_epochs == 2


def test_the_demo_runs_every_requested_model_end_to_end(tmp_path):
    args = build_parser().parse_args(
        [
            "demo",
            "--models",
            "stgnn",
            "mlp",
            "--max-epochs",
            "2",
            "--nodes-per-side",
            "8",
            "--n-time",
            "8",
            "--periods-per-year",
            "4",
        ]
    )

    results = run_demo(args)

    assert set(results) == {"stgnn", "mlp"}
    for metrics in results.values():
        assert "test/annual_f1" in metrics
        assert "truth/delay_detection_rate" in metrics

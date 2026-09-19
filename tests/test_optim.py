"""The optimiser and learning-rate-schedule registries."""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch

from gnn_sar.config import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from gnn_sar.training.datamodule import SpatioTemporalGraphDataModule
from gnn_sar.training.lightning_module import SoilSealingModule
from gnn_sar.training.optim import (
    OPTIMIZERS,
    SCHEDULERS,
    build_optimizer,
    build_scheduler,
    parameter_groups,
)


@pytest.fixture
def datamodule(scenario) -> SpatioTemporalGraphDataModule:
    module = SpatioTemporalGraphDataModule(scenario.graph, prediction_horizon=2)
    module.setup()
    return module


def _module(datamodule, training_config: TrainingConfig) -> SoilSealingModule:
    return SoilSealingModule(
        model_config=ModelConfig(in_channels=datamodule.n_features),
        training_config=training_config,
    )


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


def test_the_default_config_still_builds_an_adamw_and_no_schedule(datamodule):
    """The non-regression guarantee: the registry changed nothing by default."""
    optimizer = _module(datamodule, TrainingConfig()).configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    assert len(optimizer.param_groups) == 1


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_registered_optimizer_completes_an_epoch(name, datamodule):
    module = _module(
        datamodule, TrainingConfig(optimizer=OptimizerConfig(name=name, learning_rate=1e-3))
    )
    before = module.model.detection_head.weight.detach().clone()

    _trainer(max_epochs=2).fit(module, datamodule=datamodule)

    assert not torch.allclose(before, module.model.detection_head.weight)


@pytest.mark.parametrize("name", sorted(SCHEDULERS))
def test_every_registered_schedule_drives_the_rate_through_a_fit(name, datamodule):
    config = TrainingConfig(max_epochs=4, scheduler=SchedulerConfig(name=name))
    module = _module(datamodule, config)
    trainer = _trainer(max_epochs=4)

    trainer.fit(module, datamodule=datamodule)

    # ``cosine`` reaches its minimum — zero, by torch's default — exactly at the
    # end of the run, so the rate *after* the last step may legitimately be 0.
    rate = trainer.optimizers[0].param_groups[0]["lr"]
    assert rate >= 0 and torch.isfinite(torch.tensor(rate))


def test_a_schedule_actually_lowers_the_rate(datamodule):
    config = TrainingConfig(max_epochs=4, scheduler=SchedulerConfig(name="cosine"))
    module = _module(datamodule, config)
    trainer = _trainer(max_epochs=4)

    trainer.fit(module, datamodule=datamodule)

    assert trainer.optimizers[0].param_groups[0]["lr"] < config.learning_rate


def test_a_warm_up_starts_below_the_configured_rate(datamodule):
    config = TrainingConfig(
        max_epochs=6,
        scheduler=SchedulerConfig(name="cosine", warmup_epochs=3, warmup_start_factor=0.1),
    )
    optimizer = build_optimizer(_module(datamodule, config), config.optimizer)
    entry = build_scheduler(optimizer, config.scheduler, max_epochs=config.max_epochs)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(config.learning_rate * 0.1)
    entry["scheduler"].step()
    assert optimizer.param_groups[0]["lr"] > config.learning_rate * 0.1


def test_a_plateau_schedule_is_handed_the_metric_it_watches(datamodule):
    config = TrainingConfig(scheduler=SchedulerConfig(name="plateau", monitor="val/total"))
    optimizer = build_optimizer(_module(datamodule, config), config.optimizer)

    entry = build_scheduler(optimizer, config.scheduler, max_epochs=config.max_epochs)

    assert entry["monitor"] == "val/total"


def test_a_plateau_schedule_rejects_a_metric_nothing_logs(datamodule):
    config = TrainingConfig(scheduler=SchedulerConfig(name="plateau", monitor="val/invented"))

    with pytest.raises(ValueError, match="never logs"):
        _module(datamodule, config).configure_optimizers()


def test_a_warm_up_cannot_precede_a_plateau_schedule(datamodule):
    config = TrainingConfig(
        scheduler=SchedulerConfig(name="plateau", warmup_epochs=2)
    )
    optimizer = build_optimizer(_module(datamodule, config), config.optimizer)

    with pytest.raises(ValueError, match="cannot precede"):
        build_scheduler(optimizer, config.scheduler, max_epochs=10)


def test_the_logged_metric_names_cover_what_the_module_logs(datamodule):
    """The plateau check is only as good as the list it checks against."""
    module = _module(datamodule, TrainingConfig(max_epochs=1))
    trainer = _trainer(max_epochs=1)
    trainer.fit(module, datamodule=datamodule)

    logged = set(trainer.logged_metrics) - {"epoch", "step"}
    assert logged <= module.logged_metric_names()


def test_the_terms_of_a_configured_objective_are_monitorable(datamodule):
    """A schedule can watch any term, not only the total."""
    config = TrainingConfig(scheduler=SchedulerConfig(name="plateau", monitor="val/smoothness"))

    entry = _module(datamodule, config).configure_optimizers()

    assert entry["lr_scheduler"]["monitor"] == "val/smoothness"


def test_sparing_the_biases_from_the_decay_makes_two_groups(datamodule):
    config = OptimizerConfig(decay_on_norm_and_bias=False, weight_decay=0.1)
    module = _module(datamodule, TrainingConfig(optimizer=config))

    groups = parameter_groups(module, config)

    assert [group["weight_decay"] for group in groups] == [0.1, 0.0]
    assert all(parameter.ndim > 1 for parameter in groups[0]["params"])
    assert all(parameter.ndim <= 1 for parameter in groups[1]["params"])
    assert sum(len(group["params"]) for group in groups) == len(
        [p for p in module.parameters() if p.requires_grad]
    )


def test_an_optimizers_own_arguments_reach_it(datamodule):
    config = OptimizerConfig(name="sgd", params={"momentum": 0.9, "nesterov": True})

    optimizer = build_optimizer(_module(datamodule, TrainingConfig(optimizer=config)), config)

    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert optimizer.param_groups[0]["nesterov"] is True


def test_an_argument_the_optimizer_does_not_take_is_reported_by_name(datamodule):
    config = OptimizerConfig(name="adamw", params={"momentum": 0.9})

    with pytest.raises(ValueError, match="'adamw' does not accept"):
        build_optimizer(_module(datamodule, TrainingConfig()), config)


def test_an_unknown_optimizer_names_the_ones_that_exist(datamodule):
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(_module(datamodule, TrainingConfig()), OptimizerConfig(name="lion"))


def test_an_unknown_schedule_names_the_ones_that_exist(datamodule):
    optimizer = build_optimizer(_module(datamodule, TrainingConfig()), OptimizerConfig())

    with pytest.raises(ValueError, match="unknown scheduler"):
        build_scheduler(optimizer, SchedulerConfig(name="triangular"), max_epochs=10)


def test_a_schedule_spans_the_run_without_the_caller_computing_anything(datamodule):
    """``max_epochs`` is the total step count: one full-batch graph, one step."""
    config = TrainingConfig(max_epochs=17, scheduler=SchedulerConfig(name="cosine"))
    optimizer = build_optimizer(_module(datamodule, config), config.optimizer)

    entry = build_scheduler(optimizer, config.scheduler, max_epochs=config.max_epochs)

    assert entry["scheduler"].T_max == 17

"""The training loop (proposal phases 3–5).

Validation is deliberately reported at two levels:

*annual* — always available, and the only honest comparison against the
ISPRA/SNPA cartography, which is annual. This is the headline number.

*sub-annual* — reported only where exact dates exist (synthetic data, or the
validation points whose date is known), together with the detection delay.
Quoting a sub-annual F1 computed against annual labels would be measuring the
model against a target that does not exist, so the module simply does not
produce one.

Nothing here is device-specific: the loss, the metrics and the heads all follow
the module, and :meth:`SoilSealingModule.predict_maps` moves the batch it is
handed onto the module's own device. That is what lets the same code run on a
CPU, a CUDA card or Apple's MPS with only ``accelerator`` changing.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryCohenKappa,
    BinaryF1Score,
    BinaryJaccardIndex,
)

from gnn_sar.config import ModelConfig, TrainingConfig
from gnn_sar.data.targets import IGNORE_INDEX, state_to_onset
from gnn_sar.losses.aggregation import iter_year_bags
from gnn_sar.losses.registry import build_loss
from gnn_sar.metrics.delay import detection_delay, onset_from_probability
from gnn_sar.models.stgnn import STGNNOutput, SpatioTemporalGNN
from gnn_sar.training.datamodule import move_batch_to_device
from gnn_sar.training.optim import build_optimizer, build_scheduler

__all__ = ["SoilSealingModule"]


class SoilSealingModule(pl.LightningModule):
    """Wraps any model with the ``(features, edge_index) -> STGNNOutput`` contract.

    The full :class:`~gnn_sar.models.stgnn.SpatioTemporalGNN` and every baseline
    in :mod:`gnn_sar.models.baselines` satisfy it, so the comparison of OS3 runs
    the same loop, the same loss and the same metrics for all of them — the only
    difference is the model passed in.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        model_config: ModelConfig | None = None,
        training_config: TrainingConfig | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.training_config = training_config or TrainingConfig()
        if model is None:
            model = SpatioTemporalGNN(model_config or ModelConfig())
        self.model = model
        # The objective normally comes from the config, through the term
        # registry; ``criterion`` is for the one written in Python instead.
        self.criterion = criterion or build_loss(self.training_config.loss)
        self.save_hyperparameters(
            {"model": type(model).__name__, "training": asdict(self.training_config)}
        )

        self._metrics = nn.ModuleDict(
            {
                f"{split}_{level}": _metric_bundle()
                for split in ("val", "test")
                for level in ("annual", "subannual")
            }
        )

    def forward(self, features: Tensor, spatial_edge_index: Tensor) -> STGNNOutput:
        return self.model(features, spatial_edge_index)

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        """The optimiser and schedule the config names.

        A flat learning rate — the project's default, and what every run before
        the registry used — returns the optimiser on its own, which is the form
        Lightning treats as "no schedule at all".
        """
        training = self.training_config
        optimizer = build_optimizer(self, training.optimizer)
        scheduler = build_scheduler(
            optimizer,
            training.scheduler,
            max_epochs=training.max_epochs,
            available_metrics=self.logged_metric_names(),
        )
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def logged_metric_names(self) -> set[str]:
        """Every metric this module logs, so a schedule can be checked against it.

        A plateau schedule that watches a name nothing produces would stall the
        run at the first validation epoch with a Lightning error about a missing
        key; naming the metrics here turns that into a configuration error
        raised before the fit starts.
        """
        names = {
            f"{split}/{term}"
            for split in ("train", "val", "test")
            for term in (*self.criterion.terms.keys(), "total")
        }
        names |= {
            f"{split}/{level}_{metric}"
            for split in ("val", "test")
            for level in ("annual", "subannual")
            for metric in ("accuracy", "f1", "iou", "kappa")
        }
        names |= {
            f"{split}/delay_{name}"
            for split in ("val", "test")
            for name in (
                "mean_delay",
                "median_delay",
                "detection_rate",
                "false_alarm_rate",
                "n_evaluated",
                "n_missed",
            )
        }
        return names

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        terms = self._loss(batch)
        for name, value in terms.items():
            self.log(f"train/{name}", value, batch_size=1, prog_bar=name == "total")
        return terms["total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluate(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluate(batch, "test")

    def _loss(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        output = self(batch["features"], batch["spatial_edge_index"])
        return self.criterion(
            output,
            year_index=batch["year_index"],
            annual_labels=batch["annual_labels"],
            node_mask=batch["node_mask"],
            spatial_edge_index=batch["spatial_edge_index"],
            state_labels=batch.get("state_labels"),
            horizon_target=batch.get("horizon_target"),
        )

    @torch.no_grad()
    def _evaluate(self, batch: dict[str, Any], split: str) -> None:
        terms = self._loss(batch)
        for name, value in terms.items():
            self.log(f"{split}/{name}", value, batch_size=1, prog_bar=name == "total")

        output = self(batch["features"], batch["spatial_edge_index"])
        mask = batch["node_mask"].bool()
        threshold = self.training_config.decision_threshold

        annual_probability, annual_target = self._annual_predictions(output, batch, mask)
        self._update_and_log(
            f"{split}_annual", annual_probability, annual_target, split, "annual"
        )

        state_labels = batch.get("state_labels")
        if state_labels is None:
            return

        probability = output.sealed_probability[:, mask]
        target = state_labels[:, mask]
        valid = target != IGNORE_INDEX
        self._update_and_log(
            f"{split}_subannual",
            probability[valid],
            target[valid],
            split,
            "subannual",
        )

        delay = detection_delay(
            onset_from_probability(output.sealed_probability[:, mask], threshold),
            state_to_onset(target.cpu().numpy()),
            n_time=int(output.onset_logits.size(0)),
        )
        for name, value in delay.items():
            if not np.isnan(value):
                self.log(f"{split}/delay_{name}", float(value), batch_size=1)

    def _annual_predictions(
        self, output: STGNNOutput, batch: dict[str, Any], mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Land-take probability per (year, node), aligned with the annual labels.

        The bags are the ones the ``weak_annual`` term trains on — the same
        function builds both, so the rule the model is measured under cannot
        drift from the rule it was trained under.
        ``LossConfig.metric_aggregation`` is the way to separate them on purpose.
        """
        cfg = self.training_config.loss
        probabilities, targets = [], []
        for logit, target in iter_year_bags(
            output.onset_logits,
            batch["year_index"],
            batch["annual_labels"],
            mask,
            aggregation=cfg.effective_metric_aggregation,
            temperature=cfg.logsumexp_temperature,
        ):
            probabilities.append(torch.sigmoid(logit))
            targets.append(target)
        if not probabilities:
            empty = output.onset_logits.new_zeros(0)
            return empty, empty.long()
        return torch.cat(probabilities), torch.cat(targets)

    def _update_and_log(
        self, key: str, probability: Tensor, target: Tensor, split: str, level: str
    ) -> None:
        if probability.numel() == 0:
            return
        bundle = self._metrics[key]
        for name, metric in bundle.items():
            metric.update(probability, target.long())
            self.log(f"{split}/{level}_{name}", metric, batch_size=1, prog_bar=name == "f1")

    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """The sub-annual products, so ``Trainer.predict`` handles the device."""
        del batch_idx, dataloader_idx
        return self.predict_maps(batch)

    @torch.no_grad()
    def predict_maps(
        self, batch: dict[str, Any], threshold: float | None = None
    ) -> dict[str, np.ndarray]:
        """Sub-annual outputs for a whole study area, as NumPy arrays.

        The batch is moved onto the module's device first, so a batch straight
        from :meth:`SpatioTemporalGraphDataModule.batch` — which builds its
        tensors on the CPU — can be fed to a module that ``Trainer.fit`` has
        already moved to a GPU.

        Returns:
            ``sealed_probability`` ``(T, N)``, ``onset`` ``(N,)`` and, when the
            predictive head is enabled, ``early_warning`` ``(T, N)``.
        """
        was_training = self.training
        self.eval()
        threshold = self.training_config.decision_threshold if threshold is None else threshold
        batch = move_batch_to_device(batch, self.device)
        try:
            output = self(batch["features"], batch["spatial_edge_index"])
        finally:
            self.train(was_training)
        maps = {
            "sealed_probability": output.sealed_probability.float().cpu().numpy(),
            "onset": onset_from_probability(output.sealed_probability, threshold),
        }
        if output.horizon_logits is not None:
            maps["early_warning"] = (
                torch.sigmoid(output.horizon_logits).float().cpu().numpy()
            )
        return maps


def _metric_bundle() -> nn.ModuleDict:
    """Overall accuracy, F1, IoU and kappa — the four the SNPA validation uses."""
    return nn.ModuleDict(
        {
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "iou": BinaryJaccardIndex(),
            "kappa": BinaryCohenKappa(),
        }
    )

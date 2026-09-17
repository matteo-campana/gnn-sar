"""Weak temporal supervision (proposal §5.3).

The problem the proposal names as the critical one: the labels are annual, the
predictions are sub-annual. The objective here resolves it without inventing
dates. The detection head emits per-composite onset probabilities
:math:`q_t`; the probability that land take occurred *anywhere inside year*
:math:`y` is an aggregation of the :math:`q_t` over that year's composites, and
*that* is what the annual label supervises. The sub-annual placement of the
transition is never supervised directly — it is left to the data, constrained
by three terms:

* a **graph smoothness** term, which is how the signal diffuses from certain to
  uncertain areas along the spatial edges;
* a **single-onset** term, which stops the model from spreading a year's
  probability mass thinly across all its composites;
* an optional **strong** term on the few nodes whose exact date is known.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gnn_sar.config import LossConfig
from gnn_sar.data.targets import IGNORE_INDEX
from gnn_sar.models.stgnn import STGNNOutput, sealed_probability

__all__ = ["WeakTemporalSupervisionLoss", "aggregate_year_logit"]

_EPS = 1e-7


def aggregate_year_logit(
    onset_logits: Tensor,
    aggregation: Literal["noisy_or", "logsumexp", "max"] = "noisy_or",
    temperature: float = 1.0,
) -> Tensor:
    """Collapse a year's onset logits into one land-take logit per node.

    Args:
        onset_logits: ``(K, N)`` logits of the ``K`` composites of one year.
        aggregation: ``noisy_or`` is the probabilistically faithful choice —
            "at least one onset" under independence. ``max`` is the hard
            multiple-instance rule and passes gradient to a single composite.
            ``logsumexp`` interpolates between them.
        temperature: smoothing of ``logsumexp``; lower is closer to ``max``.

    Returns:
        ``(N,)`` logit that land take occurred in the year.
    """
    if onset_logits.dim() != 2:
        raise ValueError(f"onset_logits must be (K, N), got {tuple(onset_logits.shape)}")
    if onset_logits.size(0) == 0:
        raise ValueError("a year must contain at least one composite")

    if aggregation == "max":
        return onset_logits.max(dim=0).values
    if aggregation == "logsumexp":
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        return temperature * torch.logsumexp(onset_logits / temperature, dim=0)
    if aggregation == "noisy_or":
        # log P(no onset all year) = sum of log(1 - q_t); the logit of the
        # complement follows without ever forming a product of probabilities.
        log_survival = F.logsigmoid(-onset_logits).sum(dim=0)
        log_event = torch.log((-torch.expm1(log_survival)).clamp_min(_EPS))
        return log_event - log_survival
    raise ValueError(
        f"unknown aggregation {aggregation!r}, "
        "expected 'noisy_or', 'logsumexp' or 'max'"
    )


class WeakTemporalSupervisionLoss(nn.Module):
    """The full training objective, returned decomposed so it can be logged."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        output: STGNNOutput,
        *,
        year_index: Tensor,
        annual_labels: Tensor,
        node_mask: Tensor | None = None,
        spatial_edge_index: Tensor | None = None,
        state_labels: Tensor | None = None,
        horizon_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the objective.

        Args:
            output: the model's forward pass.
            year_index: ``(T,)`` year of each composite.
            annual_labels: ``(Y, N)`` labels in ``{0, 1}``, ``-1`` for unknown.
            node_mask: ``(N,)`` boolean mask of the nodes to train on — the
                training split of the spatially disjoint partition.
            spatial_edge_index: ``(2, E)`` edges used by the smoothness term.
            state_labels: optional ``(T, N)`` exact sealed state, ``-1`` unknown.
            horizon_target: optional ``(T, N)`` early-warning target,
                ``-1`` unknown.

        Returns:
            A dict with ``total`` and every component, all scalars.
        """
        cfg = self.config
        onset_logits = output.onset_logits
        n_time, n_nodes = onset_logits.shape
        if year_index.shape != (n_time,):
            raise ValueError(
                f"year_index must be ({n_time},), got {tuple(year_index.shape)}"
            )
        if annual_labels.dim() != 2 or annual_labels.size(1) != n_nodes:
            raise ValueError(
                f"annual_labels must be (Y, {n_nodes}), "
                f"got {tuple(annual_labels.shape)}"
            )

        mask = (
            torch.ones(n_nodes, dtype=torch.bool, device=onset_logits.device)
            if node_mask is None
            else node_mask.bool()
        )
        zero = onset_logits.new_zeros(())
        terms = {
            "weak_annual": self._weak_annual(onset_logits, year_index, annual_labels, mask),
            "sparsity": zero,
            "smoothness": zero,
            "supervised": zero,
            "horizon": zero,
        }

        if cfg.sparsity_weight > 0:
            terms["sparsity"] = self._sparsity(onset_logits, mask)
        if cfg.smoothness_weight > 0 and spatial_edge_index is not None:
            terms["smoothness"] = self._smoothness(onset_logits, spatial_edge_index)
        if cfg.supervised_weight > 0 and state_labels is not None:
            terms["supervised"] = self._supervised(onset_logits, state_labels, mask)
        if (
            cfg.horizon_weight > 0
            and horizon_target is not None
            and output.horizon_logits is not None
        ):
            terms["horizon"] = self._masked_bce(
                output.horizon_logits, horizon_target, mask
            )

        terms["total"] = (
            terms["weak_annual"]
            + cfg.sparsity_weight * terms["sparsity"]
            + cfg.smoothness_weight * terms["smoothness"]
            + cfg.supervised_weight * terms["supervised"]
            + cfg.horizon_weight * terms["horizon"]
        )
        return terms

    def _weak_annual(
        self,
        onset_logits: Tensor,
        year_index: Tensor,
        annual_labels: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Multiple-instance BCE: one bag per (node, year), the year's composites."""
        cfg = self.config
        losses, weights = [], []
        for year in torch.unique(year_index):
            members = onset_logits[year_index == year]
            target = annual_labels[int(year)]
            valid = mask & (target != IGNORE_INDEX)
            if not bool(valid.any()):
                continue
            logit = aggregate_year_logit(
                members[:, valid], cfg.aggregation, cfg.logsumexp_temperature
            )
            losses.append(
                F.binary_cross_entropy_with_logits(
                    logit,
                    target[valid].float(),
                    pos_weight=logit.new_tensor(cfg.positive_weight),
                )
            )
            weights.append(float(valid.sum()))
        if not losses:
            return onset_logits.new_zeros(())
        stacked = torch.stack(losses)
        weight = stacked.new_tensor(weights)
        return (stacked * weight).sum() / weight.sum()

    @staticmethod
    def _sparsity(onset_logits: Tensor, mask: Tensor) -> Tensor:
        """Penalise expected onsets beyond one per node over the whole series."""
        expected = torch.sigmoid(onset_logits[:, mask]).sum(dim=0)
        return F.relu(expected - 1.0).mean()

    @staticmethod
    def _smoothness(onset_logits: Tensor, spatial_edge_index: Tensor) -> Tensor:
        """Dirichlet energy of the onset probability over the spatial edges."""
        if spatial_edge_index.numel() == 0:
            return onset_logits.new_zeros(())
        probability = torch.sigmoid(onset_logits)
        src, dst = spatial_edge_index[0].long(), spatial_edge_index[1].long()
        return (probability[:, src] - probability[:, dst]).pow(2).mean()

    @staticmethod
    def _supervised(onset_logits: Tensor, state_labels: Tensor, mask: Tensor) -> Tensor:
        """BCE on the cumulative sealed state where the exact date is known."""
        if state_labels.shape != onset_logits.shape:
            raise ValueError(
                f"state_labels must match onset_logits {tuple(onset_logits.shape)}, "
                f"got {tuple(state_labels.shape)}"
            )
        probability = sealed_probability(onset_logits)
        valid = (state_labels != IGNORE_INDEX) & mask.unsqueeze(0)
        if not bool(valid.any()):
            return onset_logits.new_zeros(())
        return F.binary_cross_entropy(
            probability[valid].clamp(_EPS, 1.0 - _EPS), state_labels[valid].float()
        )

    @staticmethod
    def _masked_bce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if target.shape != logits.shape:
            raise ValueError(
                f"target must match logits {tuple(logits.shape)}, got {tuple(target.shape)}"
            )
        valid = (target != IGNORE_INDEX) & mask.unsqueeze(0)
        if not bool(valid.any()):
            return logits.new_zeros(())
        return F.binary_cross_entropy_with_logits(
            logits[valid], target[valid].float()
        )

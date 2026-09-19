"""The five terms of the proposal's objective (§5.3), each on its own.

They were one class with five private methods; they are five modules because
the interesting experiments are *which* terms and *what* each one does — a
focal weak term against a plain one, smoothness on the sealed state rather than
on the onset, at most two onsets rather than one. None of those needed new
mathematics; they needed the terms to be separable.

Each reads the settings it shares with the others (the aggregation rule, the
class weight) from :class:`~gnn_sar.config.LossConfig` unless its own ``params``
override them, so the defaults reproduce the historical objective exactly.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gnn_sar.data.targets import IGNORE_INDEX
from gnn_sar.losses.aggregation import iter_year_bags
from gnn_sar.losses.registry import LossContext, register_term
from gnn_sar.models.stgnn import sealed_probability

__all__ = [
    "HorizonTerm",
    "SmoothnessTerm",
    "SparsityTerm",
    "SupervisedTerm",
    "WeakAnnualTerm",
]

_EPS = 1e-7


@register_term("weak_annual")
class WeakAnnualTerm(nn.Module):
    """Multiple-instance cross-entropy: one bag per (node, year).

    The annual label supervises the aggregate of the year's composites, and the
    sub-annual placement of the transition inside that year is never
    supervised. That is the whole point of the formulation, and the invariant
    ``test_the_sub_annual_placement_inside_the_labelled_year_is_not_penalised``
    states it.

    Args:
        aggregation: overrides :attr:`LossConfig.aggregation` for this term.
        temperature: overrides :attr:`LossConfig.logsumexp_temperature`.
        positive_weight: overrides :attr:`LossConfig.positive_weight`.
        objective: ``bce`` is the proposal's. ``focal`` down-weights the bags
            the model already gets right, which is worth trying precisely
            because land take is rare and most bags are easy negatives.
        focal_gamma: focusing exponent, used only by ``focal``.
    """

    def __init__(
        self,
        aggregation: str | None = None,
        temperature: float | None = None,
        positive_weight: float | None = None,
        objective: Literal["bce", "focal"] = "bce",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if objective not in ("bce", "focal"):
            raise ValueError(f"objective must be 'bce' or 'focal', got {objective!r}")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be >= 0")
        self.aggregation = aggregation
        self.temperature = temperature
        self.positive_weight = positive_weight
        self.objective = objective
        self.focal_gamma = focal_gamma

    def forward(self, ctx: LossContext) -> Tensor:
        cfg = ctx.config
        positive_weight = (
            cfg.positive_weight if self.positive_weight is None else self.positive_weight
        )
        losses, weights = [], []
        bags = iter_year_bags(
            ctx.onset_logits,
            ctx.year_index,
            ctx.annual_labels,
            ctx.node_mask,
            aggregation=self.aggregation or cfg.aggregation,
            temperature=(
                cfg.logsumexp_temperature if self.temperature is None else self.temperature
            ),
        )
        for logit, target in bags:
            losses.append(self._objective(logit, target.float(), positive_weight))
            weights.append(float(target.numel()))
        if not losses:
            return ctx.zero()
        # Each year counts for as many labelled nodes as it contributes, so a
        # year with sparse labels does not weigh as much as a full one.
        stacked = torch.stack(losses)
        weight = stacked.new_tensor(weights)
        return (stacked * weight).sum() / weight.sum()

    def _objective(self, logit: Tensor, target: Tensor, positive_weight: float) -> Tensor:
        pos_weight = logit.new_tensor(positive_weight)
        if self.objective == "bce":
            return F.binary_cross_entropy_with_logits(
                logit, target, pos_weight=pos_weight
            )
        entropy = F.binary_cross_entropy_with_logits(
            logit, target, pos_weight=pos_weight, reduction="none"
        )
        probability = torch.sigmoid(logit)
        confidence = probability * target + (1.0 - probability) * (1.0 - target)
        return ((1.0 - confidence).pow(self.focal_gamma) * entropy).mean()


@register_term("sparsity")
class SparsityTerm(nn.Module):
    """Penalise expected onsets beyond ``max_onsets`` per node over the series.

    Args:
        max_onsets: the budget. ``1.0`` is the proposal's "land take happens
            once"; raising it is the ablation of that assumption.
    """

    def __init__(self, max_onsets: float = 1.0) -> None:
        super().__init__()
        if max_onsets < 0:
            raise ValueError("max_onsets must be >= 0")
        self.max_onsets = max_onsets

    def forward(self, ctx: LossContext) -> Tensor:
        expected = torch.sigmoid(ctx.onset_logits[:, ctx.node_mask]).sum(dim=0)
        return F.relu(expected - self.max_onsets).mean()


@register_term("smoothness")
class SmoothnessTerm(nn.Module):
    """Dirichlet energy over the spatial edges — the diffusion term.

    Args:
        target: ``onset`` penalises disagreement on *when*, ``sealed`` on
            *whether*. The second is the weaker constraint, since two
            neighbours may seal in different composites of the same year.
        power: exponent of the per-edge difference. ``2`` is the Dirichlet
            energy; ``1`` is a total-variation penalty, which tolerates a sharp
            boundary between a built and an unbuilt block instead of blurring it.
    """

    def __init__(
        self, target: Literal["onset", "sealed"] = "onset", power: float = 2.0
    ) -> None:
        super().__init__()
        if target not in ("onset", "sealed"):
            raise ValueError(f"target must be 'onset' or 'sealed', got {target!r}")
        if power <= 0:
            raise ValueError("power must be > 0")
        self.target = target
        self.power = power

    def forward(self, ctx: LossContext) -> Tensor | None:
        edges = ctx.spatial_edge_index
        if edges is None:
            return None
        if edges.numel() == 0:
            return ctx.zero()
        probability = (
            torch.sigmoid(ctx.onset_logits)
            if self.target == "onset"
            else sealed_probability(ctx.onset_logits)
        )
        src, dst = edges[0].long(), edges[1].long()
        difference = probability[:, src] - probability[:, dst]
        # The squared case is spelled without ``abs`` so it is bit-for-bit the
        # Dirichlet energy this term has always computed.
        if self.power == 2.0:
            return difference.pow(2).mean()
        return difference.abs().pow(self.power).mean()


@register_term("supervised")
class SupervisedTerm(nn.Module):
    """BCE on the cumulative sealed state where the exact date is known.

    Inapplicable — and silently skipped — in the weak regime, which is the
    normal one: exact dates exist only for synthetic data and for the few
    validation points whose date was recorded.
    """

    def forward(self, ctx: LossContext) -> Tensor | None:
        state_labels = ctx.state_labels
        if state_labels is None:
            return None
        onset_logits = ctx.onset_logits
        if state_labels.shape != onset_logits.shape:
            raise ValueError(
                f"state_labels must match onset_logits {tuple(onset_logits.shape)}, "
                f"got {tuple(state_labels.shape)}"
            )
        probability = sealed_probability(onset_logits)
        valid = (state_labels != IGNORE_INDEX) & ctx.node_mask.unsqueeze(0)
        if not bool(valid.any()):
            return ctx.zero()
        return F.binary_cross_entropy(
            probability[valid].clamp(_EPS, 1.0 - _EPS), state_labels[valid].float()
        )


@register_term("horizon")
class HorizonTerm(nn.Module):
    """BCE for the early-warning head, on the labelled entries only.

    Args:
        positive_weight: class weight for the head. ``None`` is torch's own
            unweighted BCE, which is what the objective has always used here.
    """

    def __init__(self, positive_weight: float | None = None) -> None:
        super().__init__()
        if positive_weight is not None and positive_weight <= 0:
            raise ValueError("positive_weight must be > 0")
        self.positive_weight = positive_weight

    def forward(self, ctx: LossContext) -> Tensor | None:
        target = ctx.horizon_target
        logits = ctx.output.horizon_logits
        if target is None or logits is None:
            return None
        if target.shape != logits.shape:
            raise ValueError(
                f"target must match logits {tuple(logits.shape)}, got {tuple(target.shape)}"
            )
        valid = (target != IGNORE_INDEX) & ctx.node_mask.unsqueeze(0)
        if not bool(valid.any()):
            return ctx.zero()
        pos_weight = (
            None
            if self.positive_weight is None
            else logits.new_tensor(self.positive_weight)
        )
        return F.binary_cross_entropy_with_logits(
            logits[valid], target[valid].float(), pos_weight=pos_weight
        )

"""How a year's composites collapse into one land-take decision.

This is the hinge of the weak-supervision objective: the labels are annual, the
head is sub-annual, and the aggregation rule is what lets one supervise the
other. It lives in its own module because two callers need exactly the same
rule — the ``weak_annual`` loss term, and the annual metrics in
:mod:`gnn_sar.training.lightning_module`. Measuring a model through a different
rule than it was trained under is a legitimate ablation
(:attr:`~gnn_sar.config.LossConfig.metric_aggregation`), but it has to be asked
for rather than happen because two functions drifted apart.
"""

from __future__ import annotations

from typing import Callable, Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

from gnn_sar.data.targets import IGNORE_INDEX

__all__ = [
    "AGGREGATIONS",
    "aggregate_year_logit",
    "iter_year_bags",
    "register_aggregation",
]

_EPS = 1e-7

AggregationRule = Callable[[Tensor, float], Tensor]

AGGREGATIONS: dict[str, AggregationRule] = {}
"""Name -> ``(onset_logits, temperature) -> year logit``."""


def register_aggregation(name: str) -> Callable[[AggregationRule], AggregationRule]:
    """Add an aggregation rule to :data:`AGGREGATIONS` under ``name``."""

    def decorate(rule: AggregationRule) -> AggregationRule:
        AGGREGATIONS[name] = rule
        return rule

    return decorate


@register_aggregation("noisy_or")
def _noisy_or(onset_logits: Tensor, temperature: float) -> Tensor:
    """"At least one onset" under independence — the faithful reading."""
    del temperature
    # log P(no onset all year) = sum of log(1 - q_t); the logit of the
    # complement follows without ever forming a product of probabilities.
    log_survival = F.logsigmoid(-onset_logits).sum(dim=0)
    log_event = torch.log((-torch.expm1(log_survival)).clamp_min(_EPS))
    return log_event - log_survival


@register_aggregation("max")
def _max(onset_logits: Tensor, temperature: float) -> Tensor:
    """The hard multiple-instance rule: gradient to a single composite."""
    del temperature
    return onset_logits.max(dim=0).values


@register_aggregation("logsumexp")
def _logsumexp(onset_logits: Tensor, temperature: float) -> Tensor:
    """A temperature-controlled interpolation towards ``max``."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return temperature * torch.logsumexp(onset_logits / temperature, dim=0)


def aggregate_year_logit(
    onset_logits: Tensor,
    aggregation: str = "noisy_or",
    temperature: float = 1.0,
) -> Tensor:
    """Collapse a year's onset logits into one land-take logit per node.

    Args:
        onset_logits: ``(K, N)`` logits of the ``K`` composites of one year.
        aggregation: a key of :data:`AGGREGATIONS`. ``noisy_or`` is the
            probabilistically faithful choice — "at least one onset" under
            independence. ``max`` is the hard multiple-instance rule and passes
            gradient to a single composite. ``logsumexp`` interpolates between
            them.
        temperature: smoothing of ``logsumexp``; lower is closer to ``max``.

    Returns:
        ``(N,)`` logit that land take occurred in the year.
    """
    if onset_logits.dim() != 2:
        raise ValueError(f"onset_logits must be (K, N), got {tuple(onset_logits.shape)}")
    if onset_logits.size(0) == 0:
        raise ValueError("a year must contain at least one composite")
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"unknown aggregation {aggregation!r}, "
            f"expected one of {sorted(AGGREGATIONS)}"
        )
    return AGGREGATIONS[aggregation](onset_logits, temperature)


def iter_year_bags(
    onset_logits: Tensor,
    year_index: Tensor,
    annual_labels: Tensor,
    mask: Tensor,
    *,
    aggregation: str = "noisy_or",
    temperature: float = 1.0,
) -> Iterator[tuple[Tensor, Tensor]]:
    """One multiple-instance bag per year: its aggregated logit and its labels.

    A year with no labelled node inside the mask is skipped rather than
    contributing an empty tensor — ``-1`` means *unknown*, never *nothing
    happened*.

    Yields:
        ``(logit, target)`` over the nodes of that year that are both inside
        ``mask`` and labelled.
    """
    for year in torch.unique(year_index):
        members = onset_logits[year_index == year]
        target = annual_labels[int(year)]
        valid = mask & (target != IGNORE_INDEX)
        if not bool(valid.any()):
            continue
        yield aggregate_year_logit(members[:, valid], aggregation, temperature), target[
            valid
        ]

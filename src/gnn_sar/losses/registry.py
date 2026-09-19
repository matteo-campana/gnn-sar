"""The objective as a list of terms, assembled from a config.

:class:`CompositeLoss` owns only what every term shares: the shape checks, the
node mask, the dictionary that gets logged, and the weighted sum. Each term is
a small :class:`~torch.nn.Module` registered by name, handed one
:class:`LossContext` and returning one scalar — or ``None`` when the run
carries no data for it, which is how the optional terms switch themselves off.

That is what makes the objective an experiment rather than a fixed formula: a
new regulariser is a function plus a line in a JSON file.

    @register_term("my_penalty")
    class MyPenalty(nn.Module):
        def __init__(self, strength: float = 1.0) -> None:
            super().__init__()
            self.strength = strength

        def forward(self, ctx: LossContext) -> Tensor:
            return self.strength * ctx.output.onset_probability.mean()

and then ``{"terms": [{"name": "my_penalty", "weight": 0.1,
"params": {"strength": 2.0}}]}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from torch import Tensor, nn

from gnn_sar.config import LossConfig, LossTermConfig

__all__ = [
    "LOSS_TERMS",
    "CompositeLoss",
    "LossContext",
    "LossTerm",
    "build_loss",
    "register_term",
]


@dataclass
class LossContext:
    """Everything a term may read about one training step.

    Passing one object rather than eight arguments is deliberate: a term that
    needs something new does not change the signature every other term is
    written against.

    Attributes:
        output: the model's forward pass.
        year_index: ``(T,)`` year of each composite.
        annual_labels: ``(Y, N)`` labels in ``{0, 1}``, ``-1`` for unknown.
        node_mask: ``(N,)`` boolean mask of the nodes being trained on.
        spatial_edge_index: ``(2, E)`` spatial edges, or ``None``.
        state_labels: ``(T, N)`` exact sealed state where known, or ``None``.
        horizon_target: ``(T, N)`` early-warning target, or ``None``.
        config: the loss config, which terms read for the settings they share
            (the aggregation rule, the class weight) unless overridden.
    """

    output: Any
    year_index: Tensor
    annual_labels: Tensor
    node_mask: Tensor
    spatial_edge_index: Tensor | None
    state_labels: Tensor | None
    horizon_target: Tensor | None
    config: LossConfig

    @property
    def onset_logits(self) -> Tensor:
        return self.output.onset_logits

    def zero(self) -> Tensor:
        return self.output.onset_logits.new_zeros(())


class LossTerm(Protocol):
    """One term of the objective."""

    def __call__(self, ctx: LossContext) -> Tensor | None:
        """The term's scalar, or ``None`` if this run carries no data for it."""


LOSS_TERMS: dict[str, Callable[..., nn.Module]] = {}
"""Name -> factory taking the term's ``params`` and returning the module."""


def register_term(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Add a term to :data:`LOSS_TERMS` under ``name``."""

    def decorate(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        LOSS_TERMS[name] = factory
        return factory

    return decorate


class CompositeLoss(nn.Module):
    """A weighted sum of registered terms, returned decomposed so it can be logged.

    The returned dict always carries a key for **every configured term**, zero
    included, plus ``total``. That is what keeps two runs comparable: a term
    that was switched off still occupies its column in MLflow instead of
    leaving a gap.
    """

    def __init__(
        self,
        config: LossConfig,
        specs: tuple[LossTermConfig, ...],
        terms: dict[str, nn.Module],
    ) -> None:
        super().__init__()
        self.config = config
        self.specs = specs
        self.terms = nn.ModuleDict(terms)

    def forward(
        self,
        output: Any,
        *,
        year_index: Tensor,
        annual_labels: Tensor,
        node_mask: Tensor | None = None,
        spatial_edge_index: Tensor | None = None,
        state_labels: Tensor | None = None,
        horizon_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the objective.

        Returns:
            A dict with ``total`` and every configured term, all scalars.
        """
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

        ctx = LossContext(
            output=output,
            year_index=year_index,
            annual_labels=annual_labels,
            node_mask=(
                torch.ones(n_nodes, dtype=torch.bool, device=onset_logits.device)
                if node_mask is None
                else node_mask.bool()
            ),
            spatial_edge_index=spatial_edge_index,
            state_labels=state_labels,
            horizon_target=horizon_target,
            config=self.config,
        )

        zero = ctx.zero()
        values: dict[str, Tensor] = {}
        total = zero
        for spec in self.specs:
            if not spec.enabled or spec.weight == 0:
                # Skipping the computation, not just the contribution: a term at
                # weight zero should cost nothing to carry through a sweep.
                values[spec.name] = zero
                continue
            value = self.terms[spec.name](ctx)
            if value is None:
                values[spec.name] = zero
                continue
            values[spec.name] = value
            total = total + spec.weight * value
        values["total"] = total
        return values


def build_loss(config: LossConfig | None = None) -> CompositeLoss:
    """The objective :meth:`~gnn_sar.config.LossConfig.resolved_terms` describes.

    Raises:
        ValueError: if a term is not registered, or rejects its ``params`` —
            reported against the term's name, so a misspelled argument reads as
            a configuration mistake.
    """
    from gnn_sar.losses import terms as _terms  # noqa: F401  (registers the defaults)

    config = config or LossConfig()
    specs = config.resolved_terms()
    built: dict[str, nn.Module] = {}
    for spec in specs:
        if spec.name not in LOSS_TERMS:
            raise ValueError(
                f"unknown loss term {spec.name!r}, expected one of {sorted(LOSS_TERMS)}"
            )
        try:
            built[spec.name] = LOSS_TERMS[spec.name](**spec.params)
        except TypeError as error:
            raise ValueError(
                f"loss term {spec.name!r} does not accept "
                f"{sorted(spec.params)}: {error}"
            ) from error
    return CompositeLoss(config, specs, built)

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

Those five terms are now separate modules in :mod:`gnn_sar.losses.terms`, and
:class:`WeakTemporalSupervisionLoss` is the composition of them the proposal
describes — the default objective, and the one every result so far was produced
with. A run that wants a different one states it as a list of terms in
:attr:`~gnn_sar.config.LossConfig.terms` and builds it with
:func:`~gnn_sar.losses.registry.build_loss`; nothing else in the training loop
changes.
"""

from __future__ import annotations

from gnn_sar.config import LossConfig
from gnn_sar.losses.aggregation import aggregate_year_logit
from gnn_sar.losses.registry import CompositeLoss, build_loss

__all__ = ["WeakTemporalSupervisionLoss", "aggregate_year_logit"]


class WeakTemporalSupervisionLoss(CompositeLoss):
    """The proposal's objective, decomposed so every term can be logged.

    It is the composition :meth:`~gnn_sar.config.LossConfig.resolved_terms`
    describes, which for a config that names no terms explicitly is exactly the
    five of §5.3 at the weights it gives them — the objective every result so
    far was produced with. ``forward`` takes the same arguments and returns the
    same dict it always has.

    Args:
        config: the weights and the aggregation rule. ``None`` is the
            proposal's default configuration.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        config = config or LossConfig()
        built = build_loss(config)
        super().__init__(config, built.specs, dict(built.terms.items()))

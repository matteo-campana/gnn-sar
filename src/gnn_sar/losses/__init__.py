"""Phase 3: the weak-supervision objective bridging annual labels to sub-annual output.

The objective is a **composition of registered terms**, not a fixed formula:
:mod:`gnn_sar.losses.terms` holds the five of the proposal,
:mod:`gnn_sar.losses.registry` assembles whichever ones a config names, and
:mod:`gnn_sar.losses.aggregation` owns the rule that turns a year's composites
into the one decision an annual label can supervise.
:class:`WeakTemporalSupervisionLoss` is the default composition.
"""

from gnn_sar.losses.aggregation import (
    AGGREGATIONS,
    aggregate_year_logit,
    iter_year_bags,
    register_aggregation,
)
from gnn_sar.losses.registry import (
    LOSS_TERMS,
    CompositeLoss,
    LossContext,
    build_loss,
    register_term,
)
from gnn_sar.losses.terms import (
    HorizonTerm,
    SmoothnessTerm,
    SparsityTerm,
    SupervisedTerm,
    WeakAnnualTerm,
)
from gnn_sar.losses.weak import WeakTemporalSupervisionLoss

__all__ = [
    "AGGREGATIONS",
    "LOSS_TERMS",
    "CompositeLoss",
    "HorizonTerm",
    "LossContext",
    "SmoothnessTerm",
    "SparsityTerm",
    "SupervisedTerm",
    "WeakAnnualTerm",
    "WeakTemporalSupervisionLoss",
    "aggregate_year_logit",
    "build_loss",
    "iter_year_bags",
    "register_aggregation",
    "register_term",
]

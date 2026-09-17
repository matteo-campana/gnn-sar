"""Phase 3: the weak-supervision objective bridging annual labels to sub-annual output."""

from gnn_sar.losses.weak import WeakTemporalSupervisionLoss, aggregate_year_logit

__all__ = ["WeakTemporalSupervisionLoss", "aggregate_year_logit"]

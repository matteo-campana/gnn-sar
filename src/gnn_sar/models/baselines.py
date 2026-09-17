"""Baselines for the comparative validation (proposal §5.4).

The point of these is an *ablation you can read*: :class:`TemporalBaseline` is
:class:`~gnn_sar.models.stgnn.SpatioTemporalGNN` with the spatial message
passing removed and nothing else changed, so the gap between the two is
attributable to the graph rather than to capacity, optimiser or supervision.
:class:`MLPBaseline` removes the temporal stage as well, giving the node-level
analogue of a per-pixel classifier. :class:`RandomForestBaseline` is the
classical, non-neural reference, and is annual by construction.

The siamese U-Net baseline named in the proposal works on rasters rather than
on the node graph; it is a separate pipeline and is deliberately not faked here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F

from gnn_sar.config import ModelConfig
from gnn_sar.models.layers import build_temporal_encoder
from gnn_sar.models.stgnn import STGNNOutput

__all__ = ["TemporalBaseline", "MLPBaseline", "RandomForestBaseline"]


class _NodeWiseBackbone(nn.Module):
    """Shared plumbing: same heads and output contract as the full model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.in_channels, config.hidden_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.hidden_channels),
        )
        self.detection_head = nn.Linear(config.hidden_channels, 1)
        self.horizon_head = (
            nn.Linear(config.hidden_channels, 1) if config.predictive_head else None
        )
        nn.init.constant_(self.detection_head.bias, -4.0)
        if self.horizon_head is not None:
            nn.init.constant_(self.horizon_head.bias, -4.0)

    def _heads(self, embeddings: Tensor) -> STGNNOutput:
        return STGNNOutput(
            onset_logits=self.detection_head(embeddings).squeeze(-1),
            horizon_logits=(
                self.horizon_head(embeddings).squeeze(-1)
                if self.horizon_head is not None
                else None
            ),
            embeddings=embeddings,
        )

    def _check(self, features: Tensor) -> None:
        if features.dim() != 3:
            raise ValueError(f"features must be (T, N, D), got {tuple(features.shape)}")
        if features.size(-1) != self.config.in_channels:
            raise ValueError(
                f"model expects {self.config.in_channels} input channels, "
                f"got {features.size(-1)}"
            )


class TemporalBaseline(_NodeWiseBackbone):
    """Temporal stage only — the ablation that isolates the graph's contribution."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig()
        super().__init__(config)
        self.temporal = build_temporal_encoder(
            config.temporal_encoder,
            config.hidden_channels,
            config.temporal_layers,
            config.gat_heads,
            config.dropout,
        )

    def forward(
        self, features: Tensor, spatial_edge_index: Tensor | None = None
    ) -> STGNNOutput:
        """``spatial_edge_index`` is accepted and ignored: that is the ablation."""
        del spatial_edge_index
        self._check(features)
        return self._heads(self.temporal(self.encoder(features)))


class MLPBaseline(_NodeWiseBackbone):
    """No graph, no temporal context: one composite in, one decision out."""

    def forward(
        self, features: Tensor, spatial_edge_index: Tensor | None = None
    ) -> STGNNOutput:
        del spatial_edge_index
        self._check(features)
        return self._heads(F.gelu(self.encoder(features)))


class RandomForestBaseline:
    """Random forest on per-year feature summaries (proposal §5.4, baseline *b*).

    It consumes the same annual labels as the neural models but cannot produce a
    sub-annual answer: one decision per node per year is all the representation
    supports. That limit is the baseline's message, not a defect.

    Requires ``scikit-learn`` (a development dependency of this project).
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: dict[str, Any] = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": 0,
            **kwargs,
        }
        self._forest: Any | None = None

    @staticmethod
    def year_features(features: np.ndarray, year_index: np.ndarray) -> np.ndarray:
        """``(T, N, D) -> (Y, N, 4 * D)``: mean, std, first and last per year.

        First and last are kept because the within-year *trajectory* is the only
        change information a per-year summary can carry.
        """
        features = np.asarray(features, dtype=np.float64)
        year_index = np.asarray(year_index, dtype=np.int64)
        if features.shape[0] != year_index.shape[0]:
            raise ValueError(
                f"{features.shape[0]} composites but {year_index.shape[0]} year indices"
            )
        n_years = int(year_index.max()) + 1
        summaries = []
        for year in range(n_years):
            members = features[year_index == year]
            if members.shape[0] == 0:
                raise ValueError(f"year {year} has no composite")
            summaries.append(
                np.concatenate(
                    [
                        members.mean(axis=0),
                        members.std(axis=0),
                        members[0],
                        members[-1],
                    ],
                    axis=-1,
                )
            )
        return np.stack(summaries)

    def fit(
        self,
        features: np.ndarray,
        year_index: np.ndarray,
        annual_labels: np.ndarray,
        node_mask: np.ndarray | None = None,
    ) -> "RandomForestBaseline":
        """Fit on ``(T, N, D)`` features against ``(Y, N)`` annual labels."""
        from sklearn.ensemble import RandomForestClassifier

        summaries = self.year_features(features, year_index)
        annual_labels = np.asarray(annual_labels)
        if summaries.shape[:2] != annual_labels.shape:
            raise ValueError(
                f"features imply {summaries.shape[:2]} (year, node) pairs but "
                f"annual_labels has shape {annual_labels.shape}"
            )
        if node_mask is not None:
            summaries = summaries[:, np.asarray(node_mask, bool)]
            annual_labels = annual_labels[:, np.asarray(node_mask, bool)]

        x = summaries.reshape(-1, summaries.shape[-1])
        y = annual_labels.reshape(-1)
        if np.unique(y).size < 2:
            raise ValueError(
                "training split contains a single class; widen it or rebalance "
                "the spatial split"
            )
        self._forest = RandomForestClassifier(**self.kwargs).fit(x, y)
        return self

    def predict_proba(self, features: np.ndarray, year_index: np.ndarray) -> np.ndarray:
        """``(Y, N)`` probability of land take in each year."""
        if self._forest is None:
            raise RuntimeError("RandomForestBaseline must be fitted before predicting")
        summaries = self.year_features(features, year_index)
        flat = summaries.reshape(-1, summaries.shape[-1])
        proba = self._forest.predict_proba(flat)[:, 1]
        return proba.reshape(summaries.shape[:2])

    def predict(
        self, features: np.ndarray, year_index: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        return (self.predict_proba(features, year_index) >= threshold).astype(np.int64)

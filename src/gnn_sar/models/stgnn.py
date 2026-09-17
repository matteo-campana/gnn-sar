"""The two-stage spatio-temporal GNN (proposal §5.3, Figure 2 blocks 5–6).

The detection head does **not** predict "is this node sealed at ``t``". It
predicts the per-period *onset* probability :math:`q_t`, and the sealed state
is its cumulative noisy-OR

.. math:: p_t = 1 - \\prod_{s \\le t} (1 - q_s).

That parameterisation is what makes the irreversibility of soil sealing a
property of the model rather than a penalty added to the loss: :math:`p_t` is
non-decreasing for any parameter values. It also gives the weak-supervision
objective its natural form — the probability that land take occurred *somewhere
inside a year* is the same noisy-OR restricted to that year's composites, so an
annual ISPRA/SNPA label supervises a sub-annual prediction directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, SAGEConv

from gnn_sar.config import ModelConfig
from gnn_sar.models.layers import build_temporal_encoder, replicate_edges

__all__ = ["STGNNOutput", "SpatioTemporalGNN", "sealed_log_survival", "sealed_probability"]


def sealed_log_survival(onset_logits: Tensor) -> Tensor:
    """``log P(node still unsealed at t)`` from per-period onset logits.

    Computed as a cumulative sum of log-sigmoids rather than a product of
    probabilities, so a long series does not underflow.
    """
    return torch.cumsum(F.logsigmoid(-onset_logits), dim=0)


def sealed_probability(onset_logits: Tensor) -> Tensor:
    """``(T, N)`` probability that a node is sealed at each composite."""
    return -torch.expm1(sealed_log_survival(onset_logits))


@dataclass
class STGNNOutput:
    """What one forward pass returns.

    Attributes:
        onset_logits: ``(T, N)`` logit of a sealing *onset* in each composite.
        horizon_logits: ``(T, N)`` logit that a node seals within the next
            ``prediction_horizon`` composites, or ``None`` if the predictive
            head is disabled.
        embeddings: ``(T, N, C)`` node representations after the temporal stage.
    """

    onset_logits: Tensor
    horizon_logits: Tensor | None
    embeddings: Tensor

    @property
    def sealed_probability(self) -> Tensor:
        """``(T, N)`` monotone sealed-state probability."""
        return sealed_probability(self.onset_logits)

    @property
    def onset_probability(self) -> Tensor:
        """``(T, N)`` per-period onset probability."""
        return torch.sigmoid(self.onset_logits)


class SpatioTemporalGNN(nn.Module):
    """Spatial GraphSAGE/GAT stage followed by a causal temporal stage.

    The spatial stage shares its weights across composites and runs on the
    block-diagonal spatial graph; the temporal stage carries information
    forward in time along the super-graph's temporal edges. Splitting the two
    keeps the model inductive — the same weights apply to a study area with a
    different node set, which is what the transferability objective (OS4)
    between Lazio and the Pianura Padana requires.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.spatial = nn.ModuleList()
        in_channels = cfg.in_channels
        for _ in range(cfg.spatial_layers):
            if cfg.spatial_conv == "sage":
                conv: nn.Module = SAGEConv(in_channels, cfg.hidden_channels)
            else:
                conv = GATConv(
                    in_channels,
                    cfg.hidden_channels // cfg.gat_heads,
                    heads=cfg.gat_heads,
                    dropout=cfg.dropout,
                )
            self.spatial.append(conv)
            in_channels = cfg.hidden_channels

        self.spatial_norm = nn.LayerNorm(cfg.hidden_channels)
        self.dropout = nn.Dropout(cfg.dropout)
        self.temporal = build_temporal_encoder(
            cfg.temporal_encoder,
            cfg.hidden_channels,
            cfg.temporal_layers,
            cfg.gat_heads,
            cfg.dropout,
        )
        self.detection_head = nn.Linear(cfg.hidden_channels, 1)
        self.horizon_head = (
            nn.Linear(cfg.hidden_channels, 1) if cfg.predictive_head else None
        )
        # Land take is rare: start from a low onset probability so training does
        # not begin by flooding the map with false positives.
        nn.init.constant_(self.detection_head.bias, -4.0)
        if self.horizon_head is not None:
            nn.init.constant_(self.horizon_head.bias, -4.0)

    def forward(self, features: Tensor, spatial_edge_index: Tensor) -> STGNNOutput:
        """Run the backbone.

        Args:
            features: ``(T, N, D)`` node features per composite.
            spatial_edge_index: ``(2, E)`` single-frame edges, shared by all
                composites because the node geometries are fixed in time.

        Returns:
            An :class:`STGNNOutput`.
        """
        if features.dim() != 3:
            raise ValueError(f"features must be (T, N, D), got {tuple(features.shape)}")
        if features.size(-1) != self.config.in_channels:
            raise ValueError(
                f"model expects {self.config.in_channels} input channels, "
                f"got {features.size(-1)}"
            )
        n_time, n_nodes, _ = features.shape
        if spatial_edge_index.numel() and int(spatial_edge_index.max()) >= n_nodes:
            raise ValueError("spatial_edge_index refers to nodes outside the graph")

        edge_index = replicate_edges(spatial_edge_index.long(), n_nodes, n_time)
        # Features arrive already standardised per channel by
        # SpatioTemporalGraphDataModule. Normalising *across* channels here
        # instead would erase the absolute backscatter level, which is a large
        # part of what separates sealed from natural ground.
        hidden = features.reshape(n_time * n_nodes, -1)
        for layer in self.spatial:
            hidden = self.dropout(F.gelu(layer(hidden, edge_index)))
        hidden = self.spatial_norm(hidden).view(n_time, n_nodes, -1)

        embeddings = self.temporal(hidden)
        onset_logits = self.detection_head(embeddings).squeeze(-1)
        horizon_logits = (
            self.horizon_head(embeddings).squeeze(-1)
            if self.horizon_head is not None
            else None
        )
        return STGNNOutput(
            onset_logits=onset_logits,
            horizon_logits=horizon_logits,
            embeddings=embeddings,
        )

    @torch.no_grad()
    def predict_onset(
        self, features: Tensor, spatial_edge_index: Tensor, threshold: float = 0.5
    ) -> Tensor:
        """``(N,)`` first composite at which each node is declared sealed.

        Nodes never crossing ``threshold`` get ``-1``. This is the quantity the
        detection-delay metric is computed on.
        """
        self.eval()
        probability = self(features, spatial_edge_index).sealed_probability
        detected = probability >= threshold
        any_detection = detected.any(dim=0)
        first = detected.float().argmax(dim=0)
        return torch.where(any_detection, first, torch.full_like(first, -1))

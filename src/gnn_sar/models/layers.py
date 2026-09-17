"""Building blocks of the spatio-temporal backbone (proposal Figure 2, block 5)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "replicate_edges",
    "LSTMTemporalEncoder",
    "CausalAttentionTemporalEncoder",
    "build_temporal_encoder",
]


def replicate_edges(edge_index: Tensor, n_nodes: int, n_time: int) -> Tensor:
    """Tile a single-frame edge set into the block-diagonal spatial graph.

    The torch counterpart of :func:`gnn_sar.data.graph.block_diagonal`. Running
    one convolution over the ``T`` blocks is mathematically identical to running
    the same shared-weight convolution on each frame, and is what lets the
    spatial stage process the whole window in a single kernel call.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be (2, E), got {tuple(edge_index.shape)}")
    offsets = torch.arange(n_time, device=edge_index.device) * n_nodes
    return (edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)


class LSTMTemporalEncoder(nn.Module):
    """Unidirectional LSTM over the composites of each node.

    Unidirectional on purpose: a sub-annual product has to be producible at time
    ``t`` from ``t`` and its past, otherwise the "detection delay" it reports is
    not a delay the operational chain could ever achieve.
    """

    def __init__(self, channels: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        """``(T, N, C) -> (T, N, C)``."""
        out, _ = self.lstm(x)
        return self.norm(out + x)


class CausalAttentionTemporalEncoder(nn.Module):
    """Masked self-attention over time, one independent sequence per node.

    The mask is strictly causal (``t`` attends to ``<= t``), which is the same
    constraint the temporal edges of the super-graph encode.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.position = nn.Parameter(torch.zeros(max_length, 1, channels))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=4 * channels,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        # Nested tensors are unavailable with pre-norm blocks; asking for them
        # only produces a warning at construction time.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.max_length = max_length

    def forward(self, x: Tensor) -> Tensor:
        """``(T, N, C) -> (T, N, C)``."""
        n_time = x.size(0)
        if n_time > self.max_length:
            raise ValueError(
                f"sequence of {n_time} composites exceeds max_length={self.max_length}"
            )
        mask = nn.Transformer.generate_square_subsequent_mask(n_time, device=x.device)
        return self.encoder(x + self.position[:n_time], mask=mask, is_causal=True)


def build_temporal_encoder(
    kind: str, channels: int, num_layers: int, num_heads: int, dropout: float
) -> nn.Module:
    """Factory used by :class:`gnn_sar.models.stgnn.SpatioTemporalGNN`."""
    if kind == "lstm":
        return LSTMTemporalEncoder(channels, num_layers=num_layers, dropout=dropout)
    if kind == "attention":
        return CausalAttentionTemporalEncoder(
            channels, num_layers=num_layers, num_heads=num_heads, dropout=dropout
        )
    raise ValueError(f"unknown temporal encoder {kind!r}, expected 'lstm' or 'attention'")

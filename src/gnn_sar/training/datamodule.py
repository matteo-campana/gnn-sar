"""Data plumbing for Lightning.

A study area is one graph, so there is no minibatching over samples here: every
step sees the whole spatio-temporal graph and the splits are **node masks** cut
along a geographic axis. Sampling nodes at random instead would leak the test
set into training through the very edges the model is built to exploit — the
leakage the proposal's protocol (§5.4) explicitly rules out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from gnn_sar.config import GraphConfig
from gnn_sar.data.features import FeatureStandardizer
from gnn_sar.data.graph import SpatioTemporalGraph
from gnn_sar.data.targets import (
    annual_to_horizon_target,
    state_to_horizon_target,
)

__all__ = ["GraphBatch", "SpatioTemporalGraphDataModule"]

GraphBatch = dict[str, Any]


class _SingleGraphDataset(Dataset):
    """A dataset of exactly one item: the graph, plus the split being served."""

    def __init__(self, batch: GraphBatch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> GraphBatch:
        if index != 0:
            raise IndexError(index)
        return self.batch


class SpatioTemporalGraphDataModule(pl.LightningDataModule):
    """Turns a :class:`SpatioTemporalGraph` into train/val/test batches.

    Args:
        graph: the study area.
        graph_config: edge settings; only ``temporal_order`` is read here, to
            record the causal structure in the batch.
        split_fractions: train/val/test share of the nodes.
        split_axis: geographic axis the split cuts along.
        prediction_horizon: horizon of the early-warning target.
        standardize: fit a :class:`FeatureStandardizer` on the training nodes
            only and apply it to all of them.
        state_labels: optional ``(T, N)`` exact sub-annual labels. Supply them
            only where they genuinely exist — the default weak regime is the
            realistic one.
    """

    def __init__(
        self,
        graph: SpatioTemporalGraph,
        graph_config: GraphConfig | None = None,
        split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        split_axis: str = "col",
        prediction_horizon: int = 2,
        standardize: bool = True,
        state_labels: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if graph.annual_labels is None or graph.year_index is None:
            raise ValueError(
                "the graph must carry annual_labels and year_index: they are the "
                "supervision signal"
            )
        self.graph = graph
        self.graph_config = graph_config or GraphConfig()
        self.split_fractions = split_fractions
        self.split_axis = split_axis
        self.prediction_horizon = prediction_horizon
        self.standardize = standardize
        self.state_labels = state_labels if state_labels is not None else graph.node_labels
        self.standardizer: FeatureStandardizer | None = None
        self.masks: dict[str, np.ndarray] = {}
        self._batches: dict[str, GraphBatch] = {}

    @property
    def n_features(self) -> int:
        return self.graph.n_features

    def setup(self, stage: str | None = None) -> None:
        if self._batches:
            return
        masks = self.graph.split_nodes(self.split_fractions, self.split_axis)  # type: ignore[arg-type]
        self.masks = masks

        features = self.graph.features
        if self.standardize:
            self.standardizer = FeatureStandardizer().fit(features, masks["train"])
            features = self.standardizer.transform(features)

        if self.state_labels is not None:
            horizon_target = state_to_horizon_target(
                self.state_labels, self.prediction_horizon
            )
        else:
            horizon_target = annual_to_horizon_target(
                self.graph.annual_labels, self.graph.year_index, self.prediction_horizon
            )

        shared = {
            "features": torch.as_tensor(features, dtype=torch.float32),
            "spatial_edge_index": torch.as_tensor(
                self.graph.spatial_edge_index, dtype=torch.long
            ),
            "spatial_edge_type": torch.as_tensor(
                self.graph.spatial_edge_type, dtype=torch.long
            ),
            "year_index": torch.as_tensor(self.graph.year_index, dtype=torch.long),
            "annual_labels": torch.as_tensor(self.graph.annual_labels, dtype=torch.long),
            "horizon_target": torch.as_tensor(horizon_target, dtype=torch.long),
            "temporal_order": self.graph_config.temporal_order,
            "state_labels": (
                None
                if self.state_labels is None
                else torch.as_tensor(self.state_labels, dtype=torch.long)
            ),
        }
        for split, mask in masks.items():
            self._batches[split] = {
                **shared,
                "split": split,
                "node_mask": torch.as_tensor(mask, dtype=torch.bool),
            }

    def _loader(self, split: str) -> DataLoader:
        if not self._batches:
            self.setup()
        return DataLoader(
            _SingleGraphDataset(self._batches[split]),
            batch_size=1,
            shuffle=False,
            collate_fn=_identity_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader:
        return self._loader("test")

    def predict_dataloader(self) -> DataLoader:
        return self._loader("test")

    def batch(self, split: str) -> GraphBatch:
        """The prepared batch of a split, e.g. to evaluate outside a Trainer."""
        if not self._batches:
            self.setup()
        return self._batches[split]


def _identity_collate(items: list[GraphBatch]) -> GraphBatch:
    """Keep the single graph as-is; the default collate would add a batch axis."""
    return items[0]

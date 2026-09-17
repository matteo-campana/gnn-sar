"""Graph construction (proposal §5.2 and Figure 2, block 4).

Two edge families carry the hypothesis that land take is *topologically
structured*:

``adjacency``
    contiguity between neighbouring segments — the urban fringe thickens where
    it already touches built-up land;
``connectivity``
    longer-range links between nearby centroids — a proxy for propagation along
    roads and other infrastructure, which plain contiguity cannot express.

Time is added as a **causal** block matrix: node ``i`` at composite ``t`` is
linked to itself at ``t-1 ... t-order``, never to the future, so nothing in the
graph lets a prediction for month ``t`` see month ``t+1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gnn_sar.config import GraphConfig
from gnn_sar.data.segmentation import NodeGeometries

__all__ = [
    "EDGE_ADJACENCY",
    "EDGE_CONNECTIVITY",
    "EDGE_SELF_LOOP",
    "EDGE_TEMPORAL",
    "SpatioTemporalGraph",
    "adjacency_edges",
    "connectivity_edges",
    "build_spatial_edges",
    "temporal_edges",
    "block_diagonal",
]

EDGE_ADJACENCY = 0
EDGE_CONNECTIVITY = 1
EDGE_SELF_LOOP = 2
EDGE_TEMPORAL = 3


def adjacency_edges(geometries: NodeGeometries) -> np.ndarray:
    """Region-adjacency edges: segments sharing a 4-connected pixel border.

    Returns:
        ``(2, E)`` array of directed edges with ``src < dst`` (one direction
        only); symmetrise with :func:`build_spatial_edges`.
    """
    labels = geometries.labels
    pairs = []
    if labels.shape[1] > 1:
        pairs.append(np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()]))
    if labels.shape[0] > 1:
        pairs.append(np.stack([labels[:-1, :].ravel(), labels[1:, :].ravel()]))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)

    stacked = np.concatenate(pairs, axis=1)
    stacked = stacked[:, stacked[0] != stacked[1]]
    lo = np.minimum(stacked[0], stacked[1])
    hi = np.maximum(stacked[0], stacked[1])
    unique = np.unique(np.stack([lo, hi]), axis=1)
    return unique.astype(np.int64)


def connectivity_edges(
    geometries: NodeGeometries,
    n_neighbors: int,
    radius: float | None = None,
    corridor: np.ndarray | None = None,
) -> np.ndarray:
    """K-nearest-centroid edges standing in for infrastructural connectivity.

    Args:
        geometries: the fixed node partition.
        n_neighbors: neighbours per node, excluding the node itself.
        radius: optional maximum centroid distance.
        corridor: optional boolean ``(n_nodes,)`` mask. When given, an edge is
            kept only if **both** endpoints lie on the corridor (e.g. nodes
            intersecting the OSM road network), which is the intended semantics
            of "propagation along infrastructure" rather than mere proximity.

    Returns:
        ``(2, E)`` array of deduplicated undirected edges with ``src < dst``.
    """
    if n_neighbors <= 0 or geometries.n_nodes < 2:
        return np.zeros((2, 0), dtype=np.int64)

    from scipy.spatial import cKDTree

    centroids = geometries.centroids
    candidates = np.arange(geometries.n_nodes)
    if corridor is not None:
        corridor = np.asarray(corridor, dtype=bool)
        if corridor.shape != (geometries.n_nodes,):
            raise ValueError(
                f"corridor must be ({geometries.n_nodes},), got {corridor.shape}"
            )
        candidates = candidates[corridor]
        if candidates.size < 2:
            return np.zeros((2, 0), dtype=np.int64)

    points = centroids[candidates]
    k = min(n_neighbors + 1, points.shape[0])
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k)
    if k == 1:  # cKDTree squeezes the last axis when k == 1
        distances, indices = distances[:, None], indices[:, None]

    src = np.repeat(candidates, k)
    dst = candidates[indices.ravel()]
    keep = src != dst
    if radius is not None:
        keep &= distances.ravel() <= radius
    src, dst = src[keep], dst[keep]
    if src.size == 0:
        return np.zeros((2, 0), dtype=np.int64)

    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    return np.unique(np.stack([lo, hi]), axis=1).astype(np.int64)


def build_spatial_edges(
    geometries: NodeGeometries,
    config: GraphConfig | None = None,
    corridor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the single-frame spatial graph.

    Returns:
        ``(edge_index, edge_type)`` where ``edge_index`` is ``(2, E)`` and
        ``edge_type`` labels every edge with one of ``EDGE_ADJACENCY``,
        ``EDGE_CONNECTIVITY`` or ``EDGE_SELF_LOOP``. Adjacency wins when an
        edge would be produced by both families, so the types stay a partition.
    """
    config = config or GraphConfig()
    adjacency = adjacency_edges(geometries)
    connectivity = connectivity_edges(
        geometries, config.n_neighbors, config.connectivity_radius, corridor
    )
    connectivity = _difference(connectivity, adjacency)

    edges = [adjacency, connectivity]
    types = [
        np.full(adjacency.shape[1], EDGE_ADJACENCY, dtype=np.int64),
        np.full(connectivity.shape[1], EDGE_CONNECTIVITY, dtype=np.int64),
    ]
    edge_index = np.concatenate(edges, axis=1)
    edge_type = np.concatenate(types)

    if config.undirected_spatial:
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        edge_type = np.concatenate([edge_type, edge_type])

    if config.self_loops:
        loops = np.arange(geometries.n_nodes, dtype=np.int64)
        edge_index = np.concatenate([edge_index, np.stack([loops, loops])], axis=1)
        edge_type = np.concatenate(
            [edge_type, np.full(loops.size, EDGE_SELF_LOOP, dtype=np.int64)]
        )
    return edge_index, edge_type


def temporal_edges(n_nodes: int, n_time: int, order: int = 1) -> np.ndarray:
    """Causal ``(t-k, i) -> (t, i)`` edges over the flattened index ``t*N + i``.

    Only backward edges are emitted, so the super-graph's adjacency matrix is
    block lower-triangular and message passing cannot leak information from the
    future into the present.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    nodes = np.arange(n_nodes, dtype=np.int64)
    src_blocks, dst_blocks = [], []
    for lag in range(1, order + 1):
        for t in range(lag, n_time):
            src_blocks.append((t - lag) * n_nodes + nodes)
            dst_blocks.append(t * n_nodes + nodes)
    if not src_blocks:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(src_blocks), np.concatenate(dst_blocks)])


def block_diagonal(edge_index: np.ndarray, n_nodes: int, n_time: int) -> np.ndarray:
    """Replicate a single-frame edge set once per composite.

    The result is the block-diagonal spatial part of the super-graph: frame
    ``t`` occupies indices ``[t*N, (t+1)*N)`` and never links to another frame.
    """
    offsets = (np.arange(n_time, dtype=np.int64) * n_nodes)[:, None, None]
    return (edge_index[None, ...] + offsets).transpose(1, 0, 2).reshape(2, -1)


def _difference(edges: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """Drop from ``edges`` every column that also appears in ``remove``."""
    if edges.shape[1] == 0 or remove.shape[1] == 0:
        return edges
    scale = max(int(edges.max()), int(remove.max())) + 1
    keys = edges[0] * scale + edges[1]
    drop = remove[0] * scale + remove[1]
    return edges[:, ~np.isin(keys, drop)]


@dataclass(frozen=True)
class SpatioTemporalGraph:
    """A study area as the model sees it.

    Attributes:
        features: ``(T, N, D)`` node features per composite.
        spatial_edge_index: ``(2, E)`` single-frame edges, shared by all ``T``.
        spatial_edge_type: ``(E,)`` edge family, see the ``EDGE_*`` constants.
        node_labels: optional ``(T, N)`` sub-annual ground truth in ``{0, 1}``
            with ``-1`` for unknown. Available only on synthetic data and on
            the few validation points where the exact date is known.
        annual_labels: optional ``(Y, N)`` ISPRA/SNPA annual labels — the
            supervision actually available at scale.
        year_index: optional ``(T,)`` mapping each composite to its year row in
            ``annual_labels``.
        geometries: optional node partition, kept so predictions can be
            rasterised back to a map.
        feature_names: optional name per channel. Feature tensors here are
            assembled from several blocks — radar statistics, texture, static
            attributes — and without names a 14-channel tensor is unreadable
            the moment anyone wants to know what the model leaned on.
    """

    features: np.ndarray
    spatial_edge_index: np.ndarray
    spatial_edge_type: np.ndarray
    node_labels: np.ndarray | None = None
    annual_labels: np.ndarray | None = None
    year_index: np.ndarray | None = None
    geometries: NodeGeometries | None = None
    feature_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError(
                f"features must be (T, N, D), got shape {self.features.shape}"
            )
        if self.spatial_edge_index.shape[0] != 2:
            raise ValueError("spatial_edge_index must have shape (2, E)")
        if self.spatial_edge_type.shape != (self.spatial_edge_index.shape[1],):
            raise ValueError("spatial_edge_type must have one entry per edge")
        if self.spatial_edge_index.size and (
            self.spatial_edge_index.max() >= self.n_nodes
            or self.spatial_edge_index.min() < 0
        ):
            raise ValueError("spatial_edge_index refers to nodes outside the graph")
        if self.node_labels is not None and self.node_labels.shape != (
            self.n_time,
            self.n_nodes,
        ):
            raise ValueError(
                f"node_labels must be ({self.n_time}, {self.n_nodes}), "
                f"got {self.node_labels.shape}"
            )
        if self.annual_labels is not None:
            if self.annual_labels.ndim != 2 or self.annual_labels.shape[1] != self.n_nodes:
                raise ValueError(
                    f"annual_labels must be (Y, {self.n_nodes}), "
                    f"got {self.annual_labels.shape}"
                )
            if self.year_index is None:
                raise ValueError("annual_labels require year_index")
            if self.year_index.shape != (self.n_time,):
                raise ValueError(
                    f"year_index must be ({self.n_time},), got {self.year_index.shape}"
                )
            if self.year_index.size and (
                self.year_index.max() >= self.annual_labels.shape[0]
                or self.year_index.min() < 0
            ):
                raise ValueError("year_index refers to a year outside annual_labels")
        if self.feature_names is not None and len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names must have {self.n_features} entries, "
                f"got {len(self.feature_names)}"
            )

    @property
    def n_time(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[2])

    @property
    def n_years(self) -> int:
        return 0 if self.annual_labels is None else int(self.annual_labels.shape[0])

    def super_edge_index(self, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """The full causal spatio-temporal graph over ``T * N`` nodes.

        Provided for models that message-pass over space and time jointly, and
        for inspection. :class:`gnn_sar.models.stgnn.SpatioTemporalGNN` uses
        only the block-diagonal spatial half, and handles the temporal half
        with its recurrent stage.
        """
        spatial = block_diagonal(self.spatial_edge_index, self.n_nodes, self.n_time)
        spatial_type = np.tile(self.spatial_edge_type, self.n_time)
        temporal = temporal_edges(self.n_nodes, self.n_time, order)
        temporal_type = np.full(temporal.shape[1], EDGE_TEMPORAL, dtype=np.int64)
        return (
            np.concatenate([spatial, temporal], axis=1),
            np.concatenate([spatial_type, temporal_type]),
        )

    def window(self, start: int, length: int) -> "SpatioTemporalGraph":
        """A contiguous slice of composites, keeping the topology intact."""
        if start < 0 or length < 1 or start + length > self.n_time:
            raise ValueError(
                f"window [{start}, {start + length}) out of range for "
                f"{self.n_time} composites"
            )
        stop = start + length
        return SpatioTemporalGraph(
            features=self.features[start:stop],
            spatial_edge_index=self.spatial_edge_index,
            spatial_edge_type=self.spatial_edge_type,
            node_labels=None if self.node_labels is None else self.node_labels[start:stop],
            annual_labels=self.annual_labels,
            year_index=None if self.year_index is None else self.year_index[start:stop],
            geometries=self.geometries,
            feature_names=self.feature_names,
        )

    def split_nodes(
        self,
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        axis: Literal["row", "col"] = "col",
    ) -> dict[str, np.ndarray]:
        """Spatially disjoint train/val/test node masks (proposal §5.4).

        Nodes are cut along a geographic axis rather than sampled at random:
        neighbouring super-pixels share both signal and label, so a random
        split leaks the test set into training through the graph itself.
        """
        if self.geometries is None:
            raise ValueError("split_nodes requires geometries")
        if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
            raise ValueError(f"fractions must be three values summing to 1, got {fractions}")

        coordinate = self.geometries.centroids[:, 0 if axis == "row" else 1]
        order = np.argsort(coordinate, kind="stable")
        cuts = np.cumsum(np.asarray(fractions) * self.n_nodes).astype(int)[:2]
        chunks = np.split(order, cuts)
        masks = {}
        for name, chunk in zip(("train", "val", "test"), chunks):
            mask = np.zeros(self.n_nodes, dtype=bool)
            mask[chunk] = True
            masks[name] = mask
        return masks

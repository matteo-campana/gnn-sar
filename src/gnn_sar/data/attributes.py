"""Static node attributes (proposal §5.2).

Alongside the multitemporal radar statistics, the proposal lists node features
that do not vary in time: *attributi geometrici* and *prossimità a viabilità ed
edificato*. They matter for a different reason than the radar channels do. The
hypothesis under test is that land take propagates along infrastructure and
thickens at the urban fringe; encoding that only in the **edges** lets the model
use it when it passes messages, but never lets it read "I am 200 m from a road
and next to built-up land" as a property of the unit itself.

**Raw coordinates are deliberately not offered.** They would let a model
memorise where the positives are, and because the train/test split is a
geographic cut (§5.4), train and test would occupy disjoint coordinate ranges —
so a coordinate feature is both a leak and a guaranteed distribution shift. What
is offered is shape (size, compactness, elongation), which is translation
invariant, and distance *to a reference layer*, which is not a position.
"""

from __future__ import annotations

import numpy as np

from gnn_sar.data.segmentation import NodeGeometries

__all__ = [
    "geometric_attributes",
    "proximity_attributes",
    "broadcast_static",
    "concatenate_features",
]

_EPS = 1e-10


def geometric_attributes(geometries: NodeGeometries) -> np.ndarray:
    """``(N, 3)`` shape descriptors: log-area, compactness, elongation.

    * **log-area** — ``log1p`` of the pixel count. Large agricultural parcels
      and small peri-urban lots behave differently, and the log keeps the
      dynamic range usable.
    * **compactness** — ``4 pi A / P^2`` from the 4-connected perimeter, 1 for a
      disc and lower for a ragged or elongated shape. Cadastral parcels and
      SLIC super-pixels differ sharply on it.
    * **elongation** — ``1 - min(side) / max(side)`` of the bounding box, 0 for
      a square and approaching 1 for a strip. Linear units are often
      infrastructure.

    All three are invariant to where the unit sits, by design.
    """
    labels = geometries.labels
    n_nodes = geometries.n_nodes
    areas = geometries.areas

    perimeter = _perimeter(labels, n_nodes)
    compactness = 4.0 * np.pi * areas / np.maximum(perimeter**2, _EPS)

    rows, cols = np.indices(labels.shape)
    flat = labels.ravel()
    height = _extent(rows.ravel(), flat, n_nodes)
    width = _extent(cols.ravel(), flat, n_nodes)
    longer = np.maximum(height, width)
    shorter = np.minimum(height, width)
    elongation = 1.0 - shorter / np.maximum(longer, _EPS)

    return np.stack([np.log1p(areas), compactness, elongation], axis=1)


def proximity_attributes(
    geometries: NodeGeometries,
    references: dict[str, np.ndarray],
    decay: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Distance from each node to each reference layer, as a decaying feature.

    Args:
        geometries: the fixed node partition.
        references: named boolean masks over the nodes — e.g.
            ``{"road": on_road_network, "built": already_sealed_at_t0}``. A node
            belonging to the layer gets distance 0.
        decay: length scale of ``exp(-d / decay)``, in the units of the
            centroids. ``None`` picks one tenth of the domain's diagonal, which
            keeps the feature informative at the scale of a peri-urban fringe
            rather than saturating across the whole study area.

    Returns:
        ``(features, names)`` with features of shape ``(N, len(references))``,
        each in ``(0, 1]``: 1 on the layer, decaying away from it. A reference
        layer that is empty yields an all-zero channel rather than an error, so
        an area with no mapped roads still runs.
    """
    from scipy.spatial import cKDTree

    centroids = geometries.centroids
    if decay is None:
        height, width = geometries.shape
        decay = float(np.hypot(height, width)) / 10.0
    if decay <= 0:
        raise ValueError("decay must be > 0")

    names = sorted(references)
    out = np.zeros((geometries.n_nodes, len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        mask = np.asarray(references[name], dtype=bool)
        if mask.shape != (geometries.n_nodes,):
            raise ValueError(
                f"reference {name!r} must be ({geometries.n_nodes},), got {mask.shape}"
            )
        if not mask.any():
            continue  # nothing mapped: a zero channel, not a crash
        distances, _ = cKDTree(centroids[mask]).query(centroids, k=1)
        out[:, column] = np.exp(-distances / decay)
    return out, names


def broadcast_static(static: np.ndarray, n_time: int) -> np.ndarray:
    """``(N, D) -> (T, N, D)``: repeat time-invariant attributes over the series."""
    static = np.asarray(static, dtype=np.float64)
    if static.ndim != 2:
        raise ValueError(f"static must be (N, D), got {static.shape}")
    if n_time < 1:
        raise ValueError("n_time must be >= 1")
    return np.broadcast_to(static[None], (n_time, *static.shape)).copy()


def concatenate_features(*blocks: np.ndarray) -> np.ndarray:
    """Join ``(T, N, D_i)`` feature blocks along the channel axis."""
    if not blocks:
        raise ValueError("at least one feature block is required")
    arrays = [np.asarray(block, dtype=np.float64) for block in blocks]
    for block in arrays:
        if block.ndim != 3:
            raise ValueError(f"every block must be (T, N, D), got {block.shape}")
    if len({block.shape[:2] for block in arrays}) != 1:
        raise ValueError(
            "blocks disagree on (T, N): "
            f"{[block.shape[:2] for block in arrays]}"
        )
    return np.concatenate(arrays, axis=2)


def _perimeter(labels: np.ndarray, n_nodes: int) -> np.ndarray:
    """4-connected boundary length of each segment, counting the raster border."""
    counts = np.zeros(n_nodes, dtype=np.float64)
    for axis in (0, 1):
        moved = np.moveaxis(labels, axis, 0)
        # Interior boundaries: adjacent pixels carrying different labels.
        differing = moved[:-1] != moved[1:]
        for side in (moved[:-1][differing], moved[1:][differing]):
            counts += np.bincount(side, minlength=n_nodes)
        # The edge of the raster bounds whatever touches it.
        for border in (moved[0], moved[-1]):
            counts += np.bincount(border.ravel(), minlength=n_nodes)
    return counts


def _extent(coordinate: np.ndarray, labels: np.ndarray, n_nodes: int) -> np.ndarray:
    """Bounding-box side of each segment along one axis, in pixels."""
    lowest = np.full(n_nodes, np.inf)
    highest = np.full(n_nodes, -np.inf)
    np.minimum.at(lowest, labels, coordinate)
    np.maximum.at(highest, labels, coordinate)
    return highest - lowest + 1.0

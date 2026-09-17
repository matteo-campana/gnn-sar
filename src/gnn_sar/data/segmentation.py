"""Stable node geometries (proposal §5.2).

Speckle makes a segmentation recomputed at every acquisition unstable, so the
node set would change from one composite to the next and the temporal edges
would be meaningless. The mitigation adopted here is the one stated in the
proposal: *segment once* — on a multitemporal composite or on cadastral
parcels — and let only the **features** vary in time on a fixed node set. That
decouples the topology from the dynamics of the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["NodeGeometries", "aggregate_stack", "slic_geometries"]

_STATISTICS = ("mean", "std", "min", "max")


@dataclass(frozen=True)
class NodeGeometries:
    """A fixed partition of the raster domain into graph nodes.

    Attributes:
        labels: ``(H, W)`` integer array mapping each pixel to a node id in
            ``[0, n_nodes)``. Ids are contiguous by construction.
        centroids: ``(n_nodes, 2)`` array of ``(row, col)`` centroids.
        areas: ``(n_nodes,)`` pixel counts.
    """

    labels: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray

    def __post_init__(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError(f"labels must be 2-D, got shape {self.labels.shape}")
        if self.centroids.shape != (self.n_nodes, 2):
            raise ValueError(
                f"centroids must be ({self.n_nodes}, 2), got {self.centroids.shape}"
            )
        if self.areas.shape != (self.n_nodes,):
            raise ValueError(
                f"areas must be ({self.n_nodes},), got {self.areas.shape}"
            )

    @property
    def n_nodes(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.labels.shape[0]), int(self.labels.shape[1]))

    @classmethod
    def from_label_image(cls, labels: np.ndarray) -> "NodeGeometries":
        """Build geometries from an arbitrary label image.

        Labels are relabelled to a contiguous ``0..N-1`` range, so the output is
        safe to use as an index even when the input has gaps (a common result
        of clipping a cadastral layer to an area of interest).
        """
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
        _, flat = np.unique(labels, return_inverse=True)
        contiguous = flat.reshape(labels.shape).astype(np.int64)
        n = int(contiguous.max()) + 1 if contiguous.size else 0

        areas = np.bincount(contiguous.ravel(), minlength=n).astype(np.float64)
        rows, cols = np.indices(contiguous.shape, dtype=np.float64)
        centroid_r = np.bincount(contiguous.ravel(), weights=rows.ravel(), minlength=n)
        centroid_c = np.bincount(contiguous.ravel(), weights=cols.ravel(), minlength=n)
        centroids = np.stack([centroid_r / areas, centroid_c / areas], axis=1)
        return cls(labels=contiguous, centroids=centroids, areas=areas)

    @classmethod
    def regular_grid(cls, height: int, width: int, block: int = 4) -> "NodeGeometries":
        """Tile the domain into ``block x block`` cells.

        A deterministic stand-in for SLIC: it needs no optional dependency, so
        tests and the synthetic demo exercise the same code path as real data.
        """
        if block < 1:
            raise ValueError("block must be >= 1")
        rows = np.arange(height) // block
        cols = np.arange(width) // block
        labels = rows[:, None] * (int(np.ceil(width / block))) + cols[None, :]
        return cls.from_label_image(labels)

    def pixel_mask(self, node: int) -> np.ndarray:
        """Boolean ``(H, W)`` mask of the pixels belonging to ``node``."""
        return self.labels == node

    def rasterise(self, node_values: np.ndarray) -> np.ndarray:
        """Paint per-node values back onto the raster grid.

        This is how a node-level prediction becomes the sub-annual soil-sealing
        map that the proposal targets as final output.
        """
        node_values = np.asarray(node_values)
        if node_values.shape[0] != self.n_nodes:
            raise ValueError(
                f"expected leading dimension {self.n_nodes}, got {node_values.shape[0]}"
            )
        return node_values[self.labels]


def aggregate_stack(
    stack: np.ndarray,
    geometries: NodeGeometries,
    statistics: Sequence[str] = ("mean", "std"),
) -> np.ndarray:
    """Pool a pixel-level raster stack onto the node set.

    Args:
        stack: ``(T, B, H, W)`` array of ``B`` bands over ``T`` composites.
        geometries: the fixed node partition.
        statistics: any of ``mean``, ``std``, ``min``, ``max``.

    Returns:
        ``(T, N, B * len(statistics))`` feature tensor, with the statistics of
        a band kept adjacent so that channel groups stay interpretable.
    """
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 4:
        raise ValueError(f"stack must be (T, B, H, W), got shape {stack.shape}")
    unknown = [s for s in statistics if s not in _STATISTICS]
    if unknown:
        raise ValueError(f"unknown statistics {unknown}, expected any of {_STATISTICS}")
    if stack.shape[-2:] != geometries.shape:
        raise ValueError(
            f"stack raster {stack.shape[-2:]} does not match geometries "
            f"{geometries.shape}"
        )

    n_time, n_bands = stack.shape[:2]
    n_nodes = geometries.n_nodes
    flat_labels = geometries.labels.ravel()
    counts = geometries.areas
    out = np.empty((n_time, n_nodes, n_bands * len(statistics)), dtype=np.float64)

    for t in range(n_time):
        for b in range(n_bands):
            values = stack[t, b].ravel()
            column = b * len(statistics)
            cache: dict[str, np.ndarray] = {}
            for offset, stat in enumerate(statistics):
                out[t, :, column + offset] = _reduce(
                    values, flat_labels, counts, n_nodes, stat, cache
                )
    return out


def _reduce(
    values: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    n_nodes: int,
    statistic: str,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    if statistic in cache:
        return cache[statistic]
    if statistic in ("mean", "std"):
        mean = np.bincount(labels, weights=values, minlength=n_nodes) / counts
        cache["mean"] = mean
        if statistic == "std":
            sq = np.bincount(labels, weights=values**2, minlength=n_nodes) / counts
            cache["std"] = np.sqrt(np.maximum(sq - mean**2, 0.0))
    elif statistic == "min":
        reduced = np.full(n_nodes, np.inf)
        np.minimum.at(reduced, labels, values)
        cache["min"] = reduced
    elif statistic == "max":
        reduced = np.full(n_nodes, -np.inf)
        np.maximum.at(reduced, labels, values)
        cache["max"] = reduced
    return cache[statistic]


def slic_geometries(
    composite: np.ndarray,
    n_segments: int = 2000,
    compactness: float = 10.0,
    **kwargs: object,
) -> NodeGeometries:
    """SLIC super-pixels (Achanta et al., 2012) over a multitemporal composite.

    ``composite`` is ``(B, H, W)`` — typically the temporal median of VV/VH
    backscatter plus coherence, i.e. a representation stable enough that the
    resulting node set can be held fixed for the whole study period.

    Requires the optional ``scikit-image`` dependency
    (``poetry install --extras segmentation``).
    """
    try:
        from skimage.segmentation import slic
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "slic_geometries requires scikit-image; install it with "
            "`poetry install --extras segmentation` or use "
            "NodeGeometries.regular_grid / NodeGeometries.from_label_image "
            "(e.g. from cadastral parcels) instead."
        ) from exc

    composite = np.asarray(composite, dtype=np.float64)
    if composite.ndim != 3:
        raise ValueError(f"composite must be (B, H, W), got shape {composite.shape}")
    labels = slic(
        np.moveaxis(composite, 0, -1),
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=-1,
        **kwargs,
    )
    return NodeGeometries.from_label_image(labels)

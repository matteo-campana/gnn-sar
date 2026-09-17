"""Per-segment GLCM texture (proposal §5.2).

The proposal lists *tessitura (GLCM)* among the node features. A grey-level
co-occurrence matrix is usually computed in a sliding window; here it is
computed **per segment**, which is both cheaper and better matched to the graph:
the node is the unit of analysis, so its texture should be a property of the
whole unit rather than of an arbitrary window that straddles its border.

Only pixel pairs whose two members belong to the *same* segment are counted, so
a node's texture never picks up the contrast of the boundary it shares with a
neighbour — that information belongs to the edges, not to the node.

This matters for soil sealing specifically: a construction site is rough and
heterogeneous where the field it replaced was smooth, and that shows up as
texture well before the mean backscatter has finished shifting.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gnn_sar.data.segmentation import NodeGeometries

__all__ = ["glcm_texture", "glcm_texture_stack", "GLCM_PROPERTIES"]

GLCM_PROPERTIES = ("contrast", "homogeneity", "energy")

# (row, column) displacements: horizontal, vertical and the two diagonals. The
# four together make the descriptor rotation-invariant enough for a landscape
# with no privileged orientation.
_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def glcm_texture(
    band: np.ndarray,
    geometries: NodeGeometries,
    levels: int = 16,
    properties: Sequence[str] = ("contrast", "homogeneity"),
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """``(N, len(properties))`` Haralick descriptors, one row per node.

    Args:
        band: ``(H, W)`` single band, typically VV or VH in dB.
        geometries: the fixed node partition.
        levels: grey levels the band is quantised to. Cost and memory are
            ``O(N * levels^2)``, so 16 is a sensible default and 32 is about as
            far as a national-scale node set will go.
        properties: any of ``contrast``, ``homogeneity``, ``energy``.
        value_range: ``(low, high)`` used for quantisation. **Pass the same
            range for every composite of a series**, otherwise each date is
            quantised against its own extremes and the texture becomes
            incomparable over time — which is exactly what a change-detection
            feature must not be. Defaults to the band's own min and max.

    Returns:
        ``(n_nodes, len(properties))``. A node too small to contain a single
        valid pixel pair gets zeros.
    """
    band = np.asarray(band, dtype=np.float64)
    if band.ndim != 2:
        raise ValueError(f"band must be (H, W), got shape {band.shape}")
    if band.shape != geometries.shape:
        raise ValueError(
            f"band raster {band.shape} does not match geometries {geometries.shape}"
        )
    if levels < 2:
        raise ValueError("levels must be >= 2")
    unknown = [name for name in properties if name not in GLCM_PROPERTIES]
    if unknown:
        raise ValueError(f"unknown properties {unknown}, expected any of {GLCM_PROPERTIES}")

    quantised = _quantise(band, levels, value_range)
    matrices = _cooccurrence(quantised, geometries.labels, geometries.n_nodes, levels)

    totals = matrices.sum(axis=(1, 2), keepdims=True)
    probability = np.divide(
        matrices, totals, out=np.zeros_like(matrices), where=totals > 0
    )

    grid = np.arange(levels)
    difference = (grid[:, None] - grid[None, :]).astype(np.float64)
    kernels = {
        "contrast": difference**2,
        "homogeneity": 1.0 / (1.0 + difference**2),
    }

    columns = []
    for name in properties:
        if name == "energy":
            columns.append(np.sqrt((probability**2).sum(axis=(1, 2))))
        else:
            columns.append((probability * kernels[name]).sum(axis=(1, 2)))
    return np.stack(columns, axis=1)


def glcm_texture_stack(
    stack: np.ndarray,
    geometries: NodeGeometries,
    bands: Sequence[int] = (0,),
    levels: int = 16,
    properties: Sequence[str] = ("contrast", "homogeneity"),
) -> np.ndarray:
    """``(T, N, len(bands) * len(properties))`` texture over a whole series.

    The quantisation range is taken once over the **entire** stack for each
    band, so textures computed at different dates are on the same scale and a
    change between them means something.
    """
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 4:
        raise ValueError(f"stack must be (T, B, H, W), got shape {stack.shape}")

    n_time = stack.shape[0]
    blocks = []
    for band in bands:
        series = stack[:, band]
        value_range = (float(series.min()), float(series.max()))
        blocks.append(
            np.stack(
                [
                    glcm_texture(series[t], geometries, levels, properties, value_range)
                    for t in range(n_time)
                ]
            )
        )
    return np.concatenate(blocks, axis=2)


def _quantise(
    band: np.ndarray, levels: int, value_range: tuple[float, float] | None
) -> np.ndarray:
    low, high = value_range if value_range is not None else (band.min(), band.max())
    if high <= low:
        return np.zeros(band.shape, dtype=np.int64)
    scaled = (band - low) / (high - low) * (levels - 1)
    return np.clip(np.rint(scaled), 0, levels - 1).astype(np.int64)


def _cooccurrence(
    quantised: np.ndarray, labels: np.ndarray, n_nodes: int, levels: int
) -> np.ndarray:
    """``(N, levels, levels)`` symmetric co-occurrence counts, per segment."""
    matrices = np.zeros((n_nodes, levels, levels), dtype=np.float64)
    height, width = quantised.shape

    for row_step, column_step in _OFFSETS:
        rows = slice(0, height - row_step)
        shifted_rows = slice(row_step, height)
        if column_step >= 0:
            columns, shifted_columns = slice(0, width - column_step), slice(column_step, width)
        else:
            columns, shifted_columns = slice(-column_step, width), slice(0, width + column_step)

        here = quantised[rows, columns]
        there = quantised[shifted_rows, shifted_columns]
        label_here = labels[rows, columns]
        label_there = labels[shifted_rows, shifted_columns]

        # Only pairs inside one segment; a pair crossing a border is an edge
        # property, not a node property.
        inside = label_here == label_there
        if not inside.any():
            continue
        node = label_here[inside]
        index = (node * levels + here[inside]) * levels + there[inside]
        counts = np.bincount(index, minlength=n_nodes * levels * levels)
        matrices += counts.reshape(n_nodes, levels, levels)

    # Symmetrise: the pair (i, j) and (j, i) describe the same texture.
    return matrices + matrices.transpose(0, 2, 1)

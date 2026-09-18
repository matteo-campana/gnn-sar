"""Annual ground truth and auxiliary masks, from rasters to node arrays.

The ISPRA/SNPA cartography is a **state** map: one raster per year saying where
the soil is sealed. What supervises this model is a **transition**:
``annual_labels[y, n] == 1`` means node ``n`` was consumed *during* year ``y``,
and ``0`` means it was not — including the years after the transition, when the
node is sealed but no longer being consumed. :func:`annual_land_take_labels`
performs that conversion, which is where most of the accounting mistakes in a
land-take experiment live:

* a node already sealed before the series starts has no onset to detect. Its
  backscatter is high from the first composite to the last, so scoring the
  model on it measures nothing, and labelling it ``1`` in the first year would
  teach the model to fire on stable built-up land. It is excluded by default.
* a node whose state is unknown in some year cannot be called a negative for
  that year, and cannot be called a negative for any *later* year either until
  its state is known again — the transition may have happened in the gap.
  ``-1`` propagates accordingly.

Rasters come in at pixel level and leave at node level. A node is a super-pixel
or a cadastral parcel and rarely flips all at once, so
:func:`node_fraction` measures *how much* of it the mask covers and the caller
sets the threshold that counts as consumed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gnn_sar.data.segmentation import NodeGeometries
from gnn_sar.data.targets import IGNORE_INDEX

__all__ = [
    "node_fraction",
    "node_state_from_rasters",
    "annual_land_take_labels",
    "corridor_from_raster",
]


def node_fraction(
    mask: np.ndarray,
    geometries: NodeGeometries,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node share of the pixels where ``mask`` holds.

    Args:
        mask: ``(H, W)`` boolean or 0/1 raster — sealed soil, a road buffer, a
            protected area.
        geometries: the fixed node partition.
        valid: optional ``(H, W)`` mask of the pixels that carry information at
            all. Nodata pixels excluded here reduce a node's *coverage* instead
            of counting as zeros.

    Returns:
        ``(fraction, coverage)``, both ``(N,)``. ``fraction`` is the share of
        the **valid** pixels of the node where the mask holds, ``coverage`` the
        share of its pixels that were valid. A node with no valid pixel gets
        ``fraction = NaN`` and ``coverage = 0``.
    """
    mask = np.asarray(mask)
    if mask.shape != geometries.shape:
        raise ValueError(
            f"mask {mask.shape} does not match the geometries {geometries.shape}"
        )
    positive = np.nan_to_num(np.asarray(mask, dtype=np.float64), nan=0.0) > 0
    if valid is None:
        valid_mask = np.ones(geometries.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid, dtype=bool)
        if valid_mask.shape != geometries.shape:
            raise ValueError(
                f"valid {valid_mask.shape} does not match the geometries "
                f"{geometries.shape}"
            )

    flat = geometries.labels.ravel()
    n_nodes = geometries.n_nodes
    valid_count = np.bincount(flat, weights=valid_mask.ravel(), minlength=n_nodes)
    hit_count = np.bincount(
        flat, weights=(positive & valid_mask).ravel(), minlength=n_nodes
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(valid_count > 0, hit_count / valid_count, np.nan)
    return fraction, valid_count / np.maximum(geometries.areas, 1.0)


def node_state_from_rasters(
    yearly_masks: Sequence[np.ndarray],
    geometries: NodeGeometries,
    threshold: float = 0.5,
    min_coverage: float = 0.5,
    valid_masks: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """``(Y, N)`` sealed **state** from one sealing raster per year.

    Args:
        yearly_masks: ``Y`` rasters of ``(H, W)``, chronological, each true (or
            non-zero) where the soil is sealed that year.
        geometries: the fixed node partition.
        threshold: share of a node's valid pixels that has to be sealed for the
            node to count as sealed. ``0.5`` reads the node as a majority vote;
            lower it to catch narrow linear features such as a new road, and
            expect more false onsets in return.
        min_coverage: share of a node's pixels that has to be valid for its
            state to be decided at all. Below it the state is ``-1``.
        valid_masks: optional per-year nodata masks, see :func:`node_fraction`.

    Returns:
        ``(Y, N)`` array in ``{0, 1, -1}``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    if not yearly_masks:
        raise ValueError("at least one yearly mask is required")
    if valid_masks is not None and len(valid_masks) != len(yearly_masks):
        raise ValueError(
            f"got {len(yearly_masks)} yearly masks but {len(valid_masks)} valid masks"
        )

    state = np.full((len(yearly_masks), geometries.n_nodes), IGNORE_INDEX, dtype=np.int64)
    for year, mask in enumerate(yearly_masks):
        valid = None if valid_masks is None else valid_masks[year]
        fraction, coverage = node_fraction(mask, geometries, valid)
        decided = (coverage >= min_coverage) & ~np.isnan(fraction)
        state[year] = np.where(
            decided, (np.nan_to_num(fraction) >= threshold).astype(np.int64), IGNORE_INDEX
        )
    return state


def annual_land_take_labels(
    state: np.ndarray,
    pre_existing: str = "ignore",
) -> np.ndarray:
    """``(Y, N)`` land-take labels from a ``(Y, N)`` sealed state.

    ``1`` marks the year a node **became** sealed, ``0`` a year in which it did
    not, ``-1`` a year whose answer the state does not determine.

    Args:
        state: ``(Y, N)`` in ``{0, 1, -1}``, chronological, as produced by
            :func:`node_state_from_rasters`.
        pre_existing: what to do with a node already sealed in the first year —
            land that was consumed before the series begins.

            ``"ignore"`` (the default)
                label every year ``-1``. The node contributes nothing to the
                loss and nothing to the metrics, which is the honest treatment:
                there is no onset inside the window to find.
            ``"negative"``
                label every year ``0``. Use it when the first year is a true
                baseline and you want the model penalised for firing on stable
                built-up land. It makes the negative class much larger, so
                raise ``positive_weight`` in :class:`~gnn_sar.config.LossConfig`
                to compensate.

    Returns:
        ``(Y, N)`` array in ``{0, 1, -1}``, with at most one ``1`` per node.
    """
    if pre_existing not in ("ignore", "negative"):
        raise ValueError(
            f"pre_existing must be 'ignore' or 'negative', got {pre_existing!r}"
        )
    state = np.asarray(state, dtype=np.int64)
    if state.ndim != 2:
        raise ValueError(f"state must be (Y, N), got {state.shape}")
    n_years, n_nodes = state.shape

    labels = np.full((n_years, n_nodes), IGNORE_INDEX, dtype=np.int64)
    sealed_before = np.zeros(n_nodes, dtype=bool)
    # True once the node's history has a gap that could hide the transition;
    # cleared again as soon as an observed "still unsealed" rules it out.
    uncertain = np.zeros(n_nodes, dtype=bool)

    for year in range(n_years):
        observed = state[year]
        known = observed != IGNORE_INDEX
        becomes = known & (observed == 1) & ~sealed_before & ~uncertain
        stays = known & (observed == 0) & ~sealed_before

        labels[year] = np.where(becomes, 1, np.where(stays, 0, IGNORE_INDEX))
        # A node already sealed in an earlier year is not being consumed now.
        labels[year] = np.where(sealed_before, 0, labels[year])

        uncertain = np.where(known & (observed == 0), False, uncertain | ~known)
        sealed_before = sealed_before | (known & (observed == 1))

    first_year_sealed = state[0] == 1
    if pre_existing == "ignore":
        labels[:, first_year_sealed] = IGNORE_INDEX
    else:
        labels[:, first_year_sealed] = 0
    return labels


def corridor_from_raster(
    mask: np.ndarray,
    geometries: NodeGeometries,
    min_fraction: float = 0.0,
) -> np.ndarray:
    """``(N,)`` boolean corridor mask from a rasterised infrastructure layer.

    Pass the result to
    :func:`gnn_sar.data.graph.build_spatial_edges` as ``corridor``: connectivity
    edges are then kept only between nodes that both lie on the corridor, which
    is what "land take propagates along infrastructure" means as a graph. Without
    it the connectivity family degrades to plain proximity.

    Args:
        mask: ``(H, W)`` raster, non-zero on the roads (an OSM road network
            buffered by a few tens of metres and rasterised on the study grid).
        geometries: the fixed node partition.
        min_fraction: share of a node's pixels that must be on the corridor.
            ``0.0`` keeps any node the network touches at all, which is usually
            what a buffered road layer wants.
    """
    fraction, _ = node_fraction(mask, geometries)
    filled = np.nan_to_num(fraction, nan=0.0)
    if min_fraction == 0.0:
        return filled > 0.0
    return filled >= min_fraction

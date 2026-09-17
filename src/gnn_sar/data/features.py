"""Feature engineering (proposal §5.1 and Figure 2, blocks 2–3).

Sentinel-1 revisits an area every 6–12 days at irregular dates, while the model
consumes a regular sequence of composites. :func:`composite_by_period` performs
that regularisation; the rest of the module holds the small numerical helpers
the pipeline needs — decibel conversion and a leakage-safe standardiser.

Within-segment dispersion produced by
:func:`gnn_sar.data.segmentation.aggregate_stack` (``std``) is what plays the
role of the texture feature: it is the cheap, orbit-stable summary of the
heterogeneity that GLCM contrast would otherwise measure per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["to_decibel", "composite_by_period", "FeatureStandardizer"]

_EPS = 1e-10


def to_decibel(power: np.ndarray, floor: float = -35.0) -> np.ndarray:
    """Convert linear backscatter :math:`\\sigma^0` to dB.

    The floor keeps calibrated water and radar-shadow pixels (which reach
    values close to zero in linear power) from producing ``-inf`` and poisoning
    the segment statistics downstream.
    """
    power = np.asarray(power, dtype=np.float64)
    if np.any(power < 0):
        raise ValueError("backscatter power must be non-negative")
    return np.maximum(10.0 * np.log10(power + _EPS), floor)


def composite_by_period(
    acquisitions: np.ndarray,
    period_index: np.ndarray,
    n_periods: int | None = None,
    reducer: Literal["median", "mean", "min", "max"] = "median",
    fill: Literal["previous", "nan", "zero"] = "previous",
) -> np.ndarray:
    """Reduce irregular acquisitions into a regular sequence of composites.

    Args:
        acquisitions: ``(K, ...)`` stack of ``K`` acquisitions, any trailing shape.
        period_index: ``(K,)`` composite each acquisition falls into (e.g. the
            month number since the start of the study period).
        n_periods: length of the output sequence. Defaults to
            ``period_index.max() + 1``.
        reducer: how acquisitions inside one period are combined. ``median`` is
            the default because it suppresses residual speckle outliers.
        fill: what to emit for a period with no acquisition — carry the
            previous composite forward (the default, and what an operational
            chain does), or mark the gap explicitly.

    Returns:
        ``(n_periods, ...)`` array of composites.
    """
    acquisitions = np.asarray(acquisitions, dtype=np.float64)
    period_index = np.asarray(period_index, dtype=np.int64)
    if acquisitions.shape[0] != period_index.shape[0]:
        raise ValueError(
            f"got {acquisitions.shape[0]} acquisitions but "
            f"{period_index.shape[0]} period indices"
        )
    if period_index.size == 0:
        raise ValueError("at least one acquisition is required")
    if period_index.min() < 0:
        raise ValueError("period_index must be non-negative")

    n_periods = int(period_index.max()) + 1 if n_periods is None else int(n_periods)
    if n_periods <= int(period_index.max()):
        raise ValueError(
            f"n_periods={n_periods} cannot hold period index {int(period_index.max())}"
        )

    reduce = {
        "median": np.median,
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
    }[reducer]

    out = np.full((n_periods, *acquisitions.shape[1:]), np.nan, dtype=np.float64)
    for period in range(n_periods):
        members = acquisitions[period_index == period]
        if members.shape[0]:
            out[period] = reduce(members, axis=0)

    empty = np.isnan(out.reshape(n_periods, -1)).all(axis=1)
    if not empty.any():
        return out
    if fill == "nan":
        return out
    if fill == "zero":
        out[empty] = 0.0
        return out

    # "previous": carry forward, and back-fill any leading gap with the first
    # observed composite so the sequence never starts undefined.
    first = int(np.argmin(empty)) if not empty.all() else None
    if first is None:
        raise ValueError("no period contains an acquisition")
    for period in range(n_periods):
        if not empty[period]:
            continue
        out[period] = out[period - 1] if period > first else out[first]
    return out


@dataclass
class FeatureStandardizer:
    """Per-channel standardisation fitted on the training nodes only.

    Fitting on every node would leak the test areas' statistics into training,
    which is exactly what the spatially disjoint split of
    :meth:`gnn_sar.data.graph.SpatioTemporalGraph.split_nodes` is there to
    prevent.
    """

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, node_mask: np.ndarray | None = None
    ) -> "FeatureStandardizer":
        """Fit on ``(T, N, D)`` features, restricted to ``node_mask`` if given."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 3:
            raise ValueError(f"features must be (T, N, D), got {features.shape}")
        subset = features if node_mask is None else features[:, np.asarray(node_mask, bool)]
        if subset.shape[1] == 0:
            raise ValueError("node_mask selects no nodes")
        flat = subset.reshape(-1, features.shape[2])
        self.mean = flat.mean(axis=0)
        scale = flat.std(axis=0)
        # A constant channel carries no information; leaving its scale at 1
        # keeps it at zero after centring instead of blowing up.
        self.scale = np.where(scale > _EPS, scale, 1.0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureStandardizer must be fitted before transform")
        features = np.asarray(features, dtype=np.float64)
        if features.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"expected {self.mean.shape[0]} channels, got {features.shape[-1]}"
            )
        return (features - self.mean) / self.scale

    def fit_transform(
        self, features: np.ndarray, node_mask: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(features, node_mask).transform(features)

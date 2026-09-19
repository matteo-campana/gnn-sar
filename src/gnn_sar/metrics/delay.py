"""Timeliness metrics (proposal §5.4).

Overall accuracy, F1, IoU and kappa say whether a map is right. They say
nothing about *when* it became right, which is the whole point of moving from
an annual to a sub-annual product — so the quantities below are the ones that
actually measure this project's claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gnn_sar.metrics._convert import to_numpy

__all__ = ["onset_from_probability", "detection_delay", "early_warning_lead_time"]

NEVER = -1


def onset_from_probability(
    probability: Any, threshold: float = 0.5, min_persistence: int = 1
) -> np.ndarray:
    """First composite at which a node is declared sealed, ``-1`` if never.

    Args:
        probability: ``(T, N)`` sealed-state probabilities.
        threshold: decision threshold.
        min_persistence: number of *consecutive* composites above the threshold
            required before a detection is emitted. Raising it trades delay for
            robustness to a single noisy composite, which is the trade-off an
            operational chain has to make explicitly.
    """
    probability = to_numpy(probability)
    if probability.ndim != 2:
        raise ValueError(f"probability must be (T, N), got {probability.shape}")
    if min_persistence < 1:
        raise ValueError("min_persistence must be >= 1")

    above = probability >= threshold
    n_time, n_nodes = above.shape
    if min_persistence > 1:
        if n_time < min_persistence:
            return np.full(n_nodes, NEVER, dtype=np.int64)
        windows = np.stack(
            [above[i : n_time - min_persistence + 1 + i] for i in range(min_persistence)]
        )
        # A detection at t requires t..t+min_persistence-1 all above threshold;
        # the reported date stays t, the first evidence, not the confirmation.
        above = windows.all(axis=0)

    detected = above.any(axis=0)
    first = above.argmax(axis=0).astype(np.int64)
    return np.where(detected, first, NEVER)


def detection_delay(
    predicted_onset: Any, true_onset: Any, n_time: int | None = None
) -> dict[str, float]:
    """Compare predicted and true onset dates.

    Args:
        predicted_onset: ``(N,)`` predicted onset index, ``-1`` for "never".
        true_onset: ``(N,)`` true onset index, ``-1`` for "never sealed".
        n_time: length of the series. When given, a node that truly sealed but
            was never detected is counted as a miss *and* charged a delay of
            ``n_time - true_onset``, so a model cannot look punctual simply by
            staying silent on the hard cases.

    Returns:
        ``mean_delay`` and ``median_delay`` in composites (positive = late,
        negative = early), plus ``detection_rate``, ``false_alarm_rate``,
        ``n_evaluated`` and ``n_missed``.
    """
    predicted = to_numpy(predicted_onset).astype(np.int64).ravel()
    truth = to_numpy(true_onset).astype(np.int64).ravel()
    if predicted.shape != truth.shape:
        raise ValueError(
            f"predicted_onset {predicted.shape} and true_onset {truth.shape} differ"
        )

    sealed = truth != NEVER
    detected = predicted != NEVER
    hit = sealed & detected
    missed = sealed & ~detected

    delays = (predicted[hit] - truth[hit]).astype(np.float64)
    if n_time is not None and missed.any():
        delays = np.concatenate([delays, (n_time - truth[missed]).astype(np.float64)])

    n_sealed = int(sealed.sum())
    n_unsealed = int((~sealed).sum())
    return {
        "mean_delay": float(delays.mean()) if delays.size else float("nan"),
        "median_delay": float(np.median(delays)) if delays.size else float("nan"),
        "detection_rate": float(hit.sum() / n_sealed) if n_sealed else float("nan"),
        "false_alarm_rate": (
            float((detected & ~sealed).sum() / n_unsealed) if n_unsealed else float("nan")
        ),
        "n_evaluated": float(n_sealed),
        "n_missed": float(missed.sum()),
    }


def early_warning_lead_time(
    horizon_probability: Any, true_onset: Any, threshold: float = 0.5
) -> dict[str, float]:
    """How far ahead the predictive head raises its first alarm.

    Only alarms raised *before* the transition count as lead time; an alarm at
    or after the onset is a detection, not an anticipation, and is excluded so
    the two capabilities are not conflated.

    Returns:
        ``mean_lead``/``median_lead`` in composites (higher is earlier),
        ``alarm_rate`` (share of sealed nodes warned in advance) and
        ``false_alarm_rate`` over nodes that never sealed.
    """
    probability = to_numpy(horizon_probability)
    truth = to_numpy(true_onset).astype(np.int64).ravel()
    if probability.ndim != 2:
        raise ValueError(f"horizon_probability must be (T, N), got {probability.shape}")
    if probability.shape[1] != truth.shape[0]:
        raise ValueError(
            f"{probability.shape[1]} nodes in probabilities but {truth.shape[0]} onsets"
        )

    above = probability >= threshold
    alarmed = above.any(axis=0)
    first_alarm = np.where(alarmed, above.argmax(axis=0), NEVER)

    sealed = truth != NEVER
    in_advance = sealed & alarmed & (first_alarm < truth)
    leads = (truth[in_advance] - first_alarm[in_advance]).astype(np.float64)

    n_sealed = int(sealed.sum())
    n_unsealed = int((~sealed).sum())
    return {
        "mean_lead": float(leads.mean()) if leads.size else float("nan"),
        "median_lead": float(np.median(leads)) if leads.size else float("nan"),
        "alarm_rate": float(in_advance.sum() / n_sealed) if n_sealed else float("nan"),
        "false_alarm_rate": (
            float((alarmed & ~sealed).sum() / n_unsealed) if n_unsealed else float("nan")
        ),
    }

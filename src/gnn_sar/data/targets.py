"""Supervision targets derived from labels of different strength.

Two regimes coexist in this project, and keeping them apart matters:

*weak*
    ISPRA/SNPA publishes one label per node per **year**. That is all a real run
    has at scale, and it is what :func:`annual_to_horizon_target` works from.
*strong*
    On the validation network — and on synthetic data — the exact composite of
    the transition is known, which :func:`state_to_horizon_target` exploits.

``-1`` marks "unknown" throughout, and losses must skip those entries rather
than treat them as negatives: a node whose year is unlabelled is not a node
where nothing happened.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "IGNORE_INDEX",
    "onset_to_state",
    "state_to_onset",
    "state_to_horizon_target",
    "annual_to_horizon_target",
]

IGNORE_INDEX = -1


def onset_to_state(onset: np.ndarray, n_time: int) -> np.ndarray:
    """``(N,)`` onset indices (``-1`` = never) to a ``(T, N)`` sealed state."""
    onset = np.asarray(onset, dtype=np.int64)
    times = np.arange(n_time, dtype=np.int64)[:, None]
    return ((onset[None, :] >= 0) & (times >= onset[None, :])).astype(np.int64)


def state_to_onset(state_labels: np.ndarray) -> np.ndarray:
    """``(T, N)`` sealed state to ``(N,)`` onset indices, ``-1`` if never sealed."""
    state = np.asarray(state_labels, dtype=np.int64)
    if state.ndim != 2:
        raise ValueError(f"state_labels must be (T, N), got {state.shape}")
    sealed = state == 1
    return np.where(sealed.any(axis=0), sealed.argmax(axis=0), -1).astype(np.int64)


def state_to_horizon_target(state_labels: np.ndarray, horizon: int) -> np.ndarray:
    """Early-warning target from exact sub-annual labels.

    ``target[t, n] = 1`` when node ``n`` is unsealed at ``t`` and sealed at some
    point in ``(t, t + horizon]``. Nodes already sealed at ``t`` are ignored —
    "will it be consumed" is not a question about land that already is — and so
    are the last ``horizon`` composites, whose future is outside the series.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    state = np.asarray(state_labels, dtype=np.int64)
    if state.ndim != 2:
        raise ValueError(f"state_labels must be (T, N), got {state.shape}")
    n_time = state.shape[0]

    target = np.full_like(state, IGNORE_INDEX)
    for t in range(n_time - horizon):
        known = state[t] != IGNORE_INDEX
        future = state[t + 1 : t + 1 + horizon]
        future_known = (future != IGNORE_INDEX).all(axis=0)
        valid = known & future_known & (state[t] == 0)
        target[t] = np.where(valid, (future.max(axis=0) > 0).astype(np.int64), IGNORE_INDEX)
    return target


def annual_to_horizon_target(
    annual_labels: np.ndarray, year_index: np.ndarray, horizon: int
) -> np.ndarray:
    """Early-warning target from annual labels only.

    ``target[t, n] = 1`` when land take is recorded for ``n`` in the year of any
    composite in ``(t, t + horizon]``. The target is therefore *coarse*: it
    stays 1 across the whole land-take year, including composites after the
    transition, because an annual label cannot say when inside the year the
    transition happened. Trained on this signal, the predictive head learns "a
    transition is near", not "a transition is exactly ``k`` composites away";
    use :func:`state_to_horizon_target` wherever exact dates exist.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    annual_labels = np.asarray(annual_labels, dtype=np.int64)
    year_index = np.asarray(year_index, dtype=np.int64)
    if annual_labels.ndim != 2:
        raise ValueError(f"annual_labels must be (Y, N), got {annual_labels.shape}")
    if year_index.ndim != 1:
        raise ValueError(f"year_index must be (T,), got {year_index.shape}")

    per_composite = annual_labels[year_index]  # (T, N)
    n_time = per_composite.shape[0]
    target = np.full_like(per_composite, IGNORE_INDEX)
    for t in range(n_time - horizon):
        future = per_composite[t + 1 : t + 1 + horizon]
        known = (future != IGNORE_INDEX).all(axis=0)
        target[t] = np.where(known, (future.max(axis=0) > 0).astype(np.int64), IGNORE_INDEX)
    return target

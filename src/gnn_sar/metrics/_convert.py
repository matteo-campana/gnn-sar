"""Turning model outputs into NumPy, whatever precision they were computed in.

Both metric modules score predictions with NumPy, so every tensor crosses this
boundary. Under mixed precision the heads emit ``float16`` or ``bfloat16``, and
``Tensor.numpy()`` refuses both — ``bfloat16`` has no NumPy dtype at all. So the
conversion upcasts reduced-precision floats first: the metrics are computed on
handfuls of numbers, where float32 costs nothing and a crash halfway through a
GPU run costs an epoch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["to_numpy"]

_UNSUPPORTED_FLOATS = ("float16", "bfloat16", "float8")


def to_numpy(values: Any) -> np.ndarray:
    """``values`` as a NumPy array, detached from the graph and off the device."""
    detach = getattr(values, "detach", None)
    if detach is None:  # already array-like
        return np.asarray(values)
    tensor = detach().cpu()
    if str(tensor.dtype).rsplit(".", 1)[-1].startswith(_UNSUPPORTED_FLOATS):
        tensor = tensor.float()
    return tensor.numpy()

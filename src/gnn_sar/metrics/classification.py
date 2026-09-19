"""Map-accuracy metrics computed from a confusion matrix.

The same four numbers the ISPRA/SNPA validation reports — overall accuracy, F1,
IoU and Cohen's kappa — in a dependency-light form usable outside a training
loop. :mod:`gnn_sar.training.lightning_module` uses the torchmetrics
equivalents during training; the two agree by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gnn_sar.data.targets import IGNORE_INDEX
from gnn_sar.metrics._convert import to_numpy

__all__ = ["confusion_matrix", "sealing_metrics"]


def confusion_matrix(prediction: Any, target: Any) -> np.ndarray:
    """``2x2`` matrix ``[[tn, fp], [fn, tp]]``, skipping ``-1`` targets."""
    prediction = to_numpy(prediction).astype(np.int64).ravel()
    target = to_numpy(target).astype(np.int64).ravel()
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {prediction.shape} and target {target.shape} differ"
        )
    keep = target != IGNORE_INDEX
    prediction, target = prediction[keep], target[keep]
    if np.any((prediction < 0) | (prediction > 1)) or np.any(target > 1):
        raise ValueError("prediction and target must be binary")
    return np.bincount(target * 2 + prediction, minlength=4).reshape(2, 2)


def sealing_metrics(prediction: Any, target: Any) -> dict[str, float]:
    """Overall accuracy, F1, IoU and kappa for the sealed class."""
    matrix = confusion_matrix(prediction, target)
    tn, fp, fn, tp = (float(v) for v in matrix.ravel())
    total = tn + fp + fn + tp
    if total == 0:
        return {name: float("nan") for name in ("accuracy", "f1", "iou", "kappa", "support")}

    accuracy = (tp + tn) / total
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else float("nan")
    union = tp + fp + fn
    iou = tp / union if union else float("nan")

    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / total**2
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else float("nan")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "iou": iou,
        "kappa": kappa,
        "support": tp + fn,
    }

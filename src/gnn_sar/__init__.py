"""Spatio-temporal GNN for sub-annual soil-sealing monitoring from Sentinel-1 SAR.

The package mirrors the five phases of the research proposal in ``docs/``:

============  ==========================================================
Phase         Module
============  ==========================================================
1 data        :mod:`gnn_sar.data.features`, :mod:`gnn_sar.data.synthetic`
2 graph       :mod:`gnn_sar.data.segmentation`, :mod:`gnn_sar.data.graph`
3 model       :mod:`gnn_sar.models`, :mod:`gnn_sar.losses`
4 validation  :mod:`gnn_sar.metrics`, :mod:`gnn_sar.models.baselines`
5 training    :mod:`gnn_sar.training`
============  ==========================================================

Top-level names are resolved lazily so that ``import gnn_sar`` stays cheap and
does not drag in torch/torch-geometric until a model is actually requested.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_LAZY: dict[str, str] = {
    # data
    "GraphConfig": "gnn_sar.config",
    "ModelConfig": "gnn_sar.config",
    "TrainingConfig": "gnn_sar.config",
    "ExperimentConfig": "gnn_sar.config",
    "SpatioTemporalGraph": "gnn_sar.data.graph",
    "NodeGeometries": "gnn_sar.data.segmentation",
    "SyntheticLandTakeScenario": "gnn_sar.data.synthetic",
    "make_synthetic_scenario": "gnn_sar.data.synthetic",
    # models
    "SpatioTemporalGNN": "gnn_sar.models.stgnn",
    # losses / metrics
    "WeakTemporalSupervisionLoss": "gnn_sar.losses.weak",
    "detection_delay": "gnn_sar.metrics.delay",
    # training
    "SoilSealingModule": "gnn_sar.training.lightning_module",
}

__all__ = ["__version__", *sorted(_LAZY)]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gnn_sar.config import (
        ExperimentConfig,
        GraphConfig,
        ModelConfig,
        TrainingConfig,
    )
    from gnn_sar.data.graph import SpatioTemporalGraph
    from gnn_sar.data.segmentation import NodeGeometries
    from gnn_sar.data.synthetic import (
        SyntheticLandTakeScenario,
        make_synthetic_scenario,
    )
    from gnn_sar.losses.weak import WeakTemporalSupervisionLoss
    from gnn_sar.metrics.delay import detection_delay
    from gnn_sar.models.stgnn import SpatioTemporalGNN
    from gnn_sar.training.lightning_module import SoilSealingModule


def __getattr__(name: str) -> Any:
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'gnn_sar' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return list(__all__)

"""Configuration dataclasses shared by the data, model and training code.

Configs are plain dataclasses so they can be built in Python, serialised to
JSON, or loaded from a file by :mod:`gnn_sar.training.cli` without pulling in a
configuration framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_type_hints

__all__ = [
    "GraphConfig",
    "ModelConfig",
    "LossConfig",
    "TrainingConfig",
    "TrackingConfig",
    "ExperimentConfig",
]

PRECISIONS = (
    "32-true",
    "64-true",
    "16-mixed",
    "bf16-mixed",
    "16-true",
    "bf16-true",
)
"""The Lightning precision settings this project supports.

The mixed ones are the reason a GPU run is worth it: ``bf16-mixed`` on any
Ampere or newer card roughly halves the memory a study area's feature tensor
occupies, which is what decides whether a whole province fits at once.
"""


@dataclass(frozen=True)
class GraphConfig:
    """Phase 2 — how pixels become a spatio-temporal graph.

    Attributes:
        n_neighbors: Number of nearest centroids linked as *connectivity* edges
            on top of plain adjacency. Models the propagation of land take
            along infrastructure, which pure contiguity cannot express.
        connectivity_radius: Maximum distance (in geometry units) for a
            connectivity edge. ``None`` disables the distance cut.
        temporal_order: How many previous composites a node is linked to by
            explicit temporal edges. ``1`` gives ``(t-1) -> t`` only. The length
            of the series itself is a property of the data, not a setting.
        undirected_spatial: Whether spatial edges are symmetrised.
        self_loops: Whether each node keeps a self-loop in the spatial graph.
    """

    n_neighbors: int = 4
    connectivity_radius: float | None = None
    temporal_order: int = 1
    undirected_spatial: bool = True
    self_loops: bool = False

    def __post_init__(self) -> None:
        if self.temporal_order < 1:
            raise ValueError("temporal_order must be >= 1")
        if self.n_neighbors < 0:
            raise ValueError("n_neighbors must be >= 0")
        if self.connectivity_radius is not None and self.connectivity_radius <= 0:
            raise ValueError("connectivity_radius must be > 0 or None")


@dataclass(frozen=True)
class ModelConfig:
    """Phase 3 — the two-stage spatio-temporal backbone."""

    in_channels: int = 8
    hidden_channels: int = 64
    spatial_layers: int = 2
    spatial_conv: Literal["sage", "gat"] = "sage"
    gat_heads: int = 4
    temporal_encoder: Literal["lstm", "attention"] = "lstm"
    temporal_layers: int = 1
    dropout: float = 0.1
    predictive_head: bool = True
    prediction_horizon: int = 2
    """Number of composites ahead the early-warning head looks."""

    def __post_init__(self) -> None:
        if self.in_channels < 1 or self.hidden_channels < 1:
            raise ValueError("in_channels and hidden_channels must be >= 1")
        if self.spatial_layers < 1 or self.temporal_layers < 1:
            raise ValueError("layer counts must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.predictive_head and self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be >= 1")
        if self.spatial_conv == "gat" and self.hidden_channels % self.gat_heads:
            raise ValueError("hidden_channels must be divisible by gat_heads")


@dataclass(frozen=True)
class LossConfig:
    """Weights of the weak-supervision objective (proposal §5.3)."""

    aggregation: Literal["noisy_or", "logsumexp", "max"] = "noisy_or"
    positive_weight: float = 1.0
    """Class weight for sealed nodes; land take is a rare event."""
    smoothness_weight: float = 0.05
    """Dirichlet energy on spatial edges — diffuses certain to uncertain areas."""
    sparsity_weight: float = 0.05
    """Penalises more than one onset per node inside the study period."""
    horizon_weight: float = 0.5
    """Weight of the early-warning head in the total loss."""
    supervised_weight: float = 1.0
    """Weight of the strong loss on nodes whose exact onset date is known."""
    logsumexp_temperature: float = 1.0

    # Monotonicity is not a penalty here: the detection head predicts per-period
    # *onset* probabilities and the sealed state is their cumulative noisy-OR,
    # so irreversibility holds by construction rather than by regularisation.

    def __post_init__(self) -> None:
        negative = {
            name: value
            for name, value in (
                ("positive_weight", self.positive_weight),
                ("smoothness_weight", self.smoothness_weight),
                ("sparsity_weight", self.sparsity_weight),
                ("horizon_weight", self.horizon_weight),
                ("supervised_weight", self.supervised_weight),
            )
            if value < 0
        }
        if negative:
            raise ValueError(f"loss weights must be >= 0, got {negative}")
        if self.logsumexp_temperature <= 0:
            raise ValueError("logsumexp_temperature must be > 0")


@dataclass(frozen=True)
class TrainingConfig:
    """Phase 4/5 — optimisation, hardware and evaluation protocol.

    The hardware settings are deliberately thin: ``accelerator="auto"`` with
    ``devices=1`` picks up a CUDA card, Apple's MPS or the CPU without the
    caller doing anything. A study area is **one** full-batch graph, so more
    than one device would replicate the same graph rather than split it; the
    lever that actually buys headroom is ``precision``.
    """

    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    accelerator: str = "auto"
    devices: int | str = 1
    precision: str = "32-true"
    """Lightning precision; see :data:`PRECISIONS`. Mixed precision is ignored
    on CPU by Lightning itself, so the default config stays portable."""
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Tensor-core policy for float32 matmuls (``torch.set_float32_matmul_precision``)."""
    gradient_clip_val: float = 1.0
    decision_threshold: float = 0.5
    log_every_n_steps: int = 1
    deterministic: bool = False
    """Ask cuDNN for reproducible kernels. Slower, and some scatter operations
    torch-geometric relies on have no deterministic implementation, so this is
    off by default and ``warn`` rather than ``True`` is used when enabled."""
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {list(PRECISIONS)}, got {self.precision!r}"
            )
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")


@dataclass(frozen=True)
class TrackingConfig:
    """Where a run is recorded — MLflow, for every experiment in the project.

    Attributes:
        enabled: Turn tracking off entirely (``--no-tracking``). A run then
            produces its metrics on stdout and nothing on disk.
        experiment_name: MLflow experiment the runs are grouped under.
        tracking_uri: ``None`` falls back to ``MLFLOW_TRACKING_URI`` and then to
            a local ``./mlruns`` file store, so ``mlflow ui`` works with no
            setup; point it at a server to share the runs.
        run_name: Name of the run. ``None`` lets MLflow generate one, except in
            the demo, where the model being compared is the natural name.
        log_checkpoints: Attach the trained weights to the run.
        log_maps: Attach the exported sub-annual GeoTIFFs to the run. They are
            the actual product, but also the bulkiest artefact.
    """

    enabled: bool = True
    experiment_name: str = "gnn-sar"
    tracking_uri: str | None = None
    run_name: str | None = None
    log_checkpoints: bool = True
    log_maps: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce a run."""

    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return _build(cls, payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _build(cls: type, payload: dict[str, Any]) -> Any:
    """Recursively instantiate nested dataclasses from a plain mapping."""
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    # ``from __future__ import annotations`` turns field types into strings, so
    # resolve them before testing for nested dataclasses.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        annotation = hints[name]
        if isinstance(value, dict) and is_dataclass(annotation):
            kwargs[name] = _build(annotation, value)  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)

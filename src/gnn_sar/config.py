"""Configuration dataclasses shared by the data, model and training code.

Configs are plain dataclasses so they can be built in Python, serialised to
JSON, or loaded from a file by :mod:`gnn_sar.training.cli` without pulling in a
configuration framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

__all__ = [
    "GraphConfig",
    "ModelConfig",
    "LossTermConfig",
    "LossConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "TrackingConfig",
    "ExperimentConfig",
    "apply_overrides",
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
class LossTermConfig:
    """One term of the objective, named after its entry in the term registry.

    Attributes:
        name: key in :data:`gnn_sar.losses.registry.LOSS_TERMS`.
        weight: multiplier of the term inside ``total``.
        enabled: ``False`` keeps the term in the log at zero without computing
            it — the difference from ``weight=0`` is only that the intent is
            explicit in the recorded config.
        params: keyword arguments handed to the term's factory. What each term
            accepts is documented on the term itself.
    """

    name: str
    weight: float = 1.0
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a loss term needs a name")
        if self.weight < 0:
            raise ValueError(f"loss weights must be >= 0, got {self.weight} for {self.name!r}")
        if not isinstance(self.params, dict):
            raise ValueError(f"params of {self.name!r} must be a mapping")


@dataclass(frozen=True)
class LossConfig:
    """The weak-supervision objective (proposal §5.3), term by term.

    There are two ways to describe it, and they are the same thing seen from
    two distances. The **weights** below name the five terms of the proposal
    directly and are what a run normally touches. :attr:`terms`, when it is not
    empty, replaces them with an explicit list — which is what lets a run add a
    term the proposal never mentioned, drop one it does, or give the same term
    different hyper-parameters. :meth:`resolved_terms` is the single answer to
    "what is actually being optimised", and everything downstream reads it
    rather than the weights.
    """

    aggregation: Literal["noisy_or", "logsumexp", "max"] = "noisy_or"
    positive_weight: float = 1.0
    """Class weight for sealed nodes; land take is a rare event."""
    weak_annual_weight: float = 1.0
    """Weight of the multiple-instance term the annual labels supervise."""
    smoothness_weight: float = 0.05
    """Dirichlet energy on spatial edges — diffuses certain to uncertain areas."""
    sparsity_weight: float = 0.05
    """Penalises more than one onset per node inside the study period."""
    horizon_weight: float = 0.5
    """Weight of the early-warning head in the total loss."""
    supervised_weight: float = 1.0
    """Weight of the strong loss on nodes whose exact onset date is known."""
    logsumexp_temperature: float = 1.0
    metric_aggregation: Literal["noisy_or", "logsumexp", "max"] | None = None
    """Aggregation the *annual metrics* use. ``None`` follows :attr:`aggregation`,
    which is the honest default: measuring a model through a rule it was not
    trained under is a deliberate choice, so it has to be made explicitly."""
    terms: tuple[LossTermConfig, ...] = ()
    """An explicit objective. Empty means "the five terms and the weights above"."""

    # Monotonicity is not a penalty here: the detection head predicts per-period
    # *onset* probabilities and the sealed state is their cumulative noisy-OR,
    # so irreversibility holds by construction rather than by regularisation.

    def __post_init__(self) -> None:
        negative = {
            name: value
            for name, value in (
                ("positive_weight", self.positive_weight),
                ("weak_annual_weight", self.weak_annual_weight),
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
        if self.terms:
            object.__setattr__(self, "terms", tuple(self.terms))
            names = [term.name for term in self.terms]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    f"a term may appear once in the objective, got {duplicates} twice"
                )

    @property
    def effective_metric_aggregation(self) -> str:
        """The rule the annual metrics aggregate a year's composites with."""
        return self.metric_aggregation or self.aggregation

    def resolved_terms(self) -> tuple[LossTermConfig, ...]:
        """The objective as a list of terms, explicit or derived from the weights.

        The derived form lists all five terms even at weight zero, so a run's
        log always carries the same keys and two runs stay comparable.
        """
        if self.terms:
            return self.terms
        return (
            LossTermConfig("weak_annual", self.weak_annual_weight),
            LossTermConfig("sparsity", self.sparsity_weight),
            LossTermConfig("smoothness", self.smoothness_weight),
            LossTermConfig("supervised", self.supervised_weight),
            LossTermConfig("horizon", self.horizon_weight),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    """Which optimiser, and the two settings every optimiser shares.

    Anything specific to one of them — ``betas``, ``momentum``, ``nesterov``,
    ``amsgrad`` — goes in :attr:`params` and is handed to its constructor
    untouched, so the registry can gain an optimiser without this class
    changing.
    """

    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    params: dict[str, Any] = field(default_factory=dict)
    decay_on_norm_and_bias: bool = True
    """``False`` splits the parameters in two groups and exempts biases and
    normalisation scales from the decay, which is the usual practice for
    transformers and costs nothing to try here."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an optimiser needs a name")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not isinstance(self.params, dict):
            raise ValueError("optimizer params must be a mapping")


@dataclass(frozen=True)
class SchedulerConfig:
    """The learning-rate schedule. ``name="none"`` is a constant rate.

    A study area is one full-batch graph, so there is exactly **one optimiser
    step per epoch**: ``interval`` is a formality here, the total number of
    steps is known in advance (``max_epochs``), and the schedule matters more
    than it would in a run with thousands of steps.
    """

    name: str = "none"
    params: dict[str, Any] = field(default_factory=dict)
    monitor: str = "val/total"
    """Metric a plateau schedule watches. It has to be one the module logs."""
    interval: Literal["epoch", "step"] = "epoch"
    frequency: int = 1
    warmup_epochs: int = 0
    """Epochs of linear warm-up prepended to the schedule. ``0`` disables it."""
    warmup_start_factor: float = 0.1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a scheduler needs a name ('none' for a constant rate)")
        if self.frequency < 1:
            raise ValueError("frequency must be >= 1")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if not 0.0 < self.warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0, 1]")
        if not isinstance(self.params, dict):
            raise ValueError("scheduler params must be a mapping")

    @property
    def enabled(self) -> bool:
        return self.name != "none"


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
    """Shorthand for ``optimizer.learning_rate``; the two are kept in step."""
    weight_decay: float = 1e-4
    """Shorthand for ``optimizer.weight_decay``; the two are kept in step."""
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
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self._reconcile_optimizer()
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {list(PRECISIONS)}, got {self.precision!r}"
            )
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")

    def _reconcile_optimizer(self) -> None:
        """Keep ``learning_rate``/``weight_decay`` and ``optimizer`` one setting.

        Both spellings exist because the short one reads better in a sweep and
        predates the registry, while the long one is where an optimiser's own
        arguments live. **The optimizer section is the canonical one**: the
        shorthand seeds it when the section says nothing, and the section wins
        when it does. Either way the two end up equal, so reading the shorthand
        always reports what the run will actually use.

        The asymmetry is what makes an override predictable. Once a shorthand
        has been set, it is no longer at its default, and a later
        ``training.optimizer.learning_rate=...`` would otherwise be fighting a
        value the user set an hour earlier in a config file.
        """
        defaults = OptimizerConfig()
        seeded = {
            name: getattr(self, name)
            for name in ("learning_rate", "weight_decay")
            if getattr(self, name) != getattr(defaults, name)
            and getattr(self.optimizer, name) == getattr(defaults, name)
        }
        if seeded:
            object.__setattr__(self, "optimizer", replace(self.optimizer, **seeded))
        object.__setattr__(self, "learning_rate", self.optimizer.learning_rate)
        object.__setattr__(self, "weight_decay", self.optimizer.weight_decay)


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
    return cls(**{name: _coerce(hints[name], value) for name, value in payload.items()})


def _coerce(annotation: Any, value: Any) -> Any:
    """One JSON value as the field's type, one level at a time.

    Three shapes matter: a nested dataclass (a mapping), a sequence of them
    (``LossConfig.terms``, which JSON round-trips as a list), and a free-form
    ``params`` mapping, which is handed through untouched precisely because its
    keys belong to the optimiser or term rather than to this schema.
    """
    if isinstance(value, dict) and is_dataclass(annotation):
        return _build(annotation, value)  # type: ignore[arg-type]
    origin = get_origin(annotation)
    if origin in (tuple, list) and isinstance(value, (list, tuple)):
        args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
        item = args[0] if args else Any
        coerced = [_coerce(item, entry) for entry in value]
        return tuple(coerced) if origin is tuple else coerced
    return value


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """A config with dotted keys replaced — ``training.optimizer.name=sgd``.

    This is what makes a sweep expressible without writing one JSON file per
    point. Keys address the dataclass tree by field name; the last segment may
    also address a key *inside* a free-form ``params`` mapping, which is the
    only place the tree stops being a schema. Every value goes through the
    dataclasses' own ``__post_init__``, so an impossible combination is
    rejected here rather than at epoch one.
    """
    result = config
    for dotted, value in overrides.items():
        result = _override(result, dotted.split("."), value, dotted)
    return result


def _override(node: Any, path: list[str], value: Any, dotted: str) -> Any:
    head, rest = path[0], path[1:]
    if isinstance(node, dict):
        # A free-form ``params`` mapping: its keys are not a schema, so the
        # remaining path is taken literally.
        return {**node, ".".join(path): value} if rest else {**node, head: value}
    if not is_dataclass(node):
        raise ValueError(f"{dotted!r} descends into {type(node).__name__}, which has no fields")
    known = {f.name for f in fields(node)}
    if head not in known:
        raise ValueError(
            f"unknown key {dotted!r}: {type(node).__name__} has no field {head!r} "
            f"(it has {sorted(known)})"
        )
    current = getattr(node, head)
    if not rest:
        return replace(node, **{head: _coerce(get_type_hints(type(node))[head], value)})
    return replace(node, **{head: _override(current, rest, value, dotted)})

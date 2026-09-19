"""Optimisers and learning-rate schedules, chosen by name.

The two registries here are the same idea the model comparison already uses in
:mod:`gnn_sar.training.cli`: a name in a config selects a factory, so an
experiment is a line in a JSON file rather than an edit to
:class:`~gnn_sar.training.lightning_module.SoilSealingModule`.

One property of this project decides how the schedules behave. A study area is
**one full-batch graph**, so the loop takes exactly one optimiser step per
epoch. ``interval="step"`` and ``interval="epoch"`` therefore describe the same
cadence, the total number of steps is known before the run starts
(``max_epochs``), and a schedule over fifty steps is a first-order
hyper-parameter rather than a refinement — which is why ``cosine`` and
``onecycle`` can be built here without the caller computing anything.

Registering a new one from outside the package::

    @register_optimizer("lion")
    def _lion(params, *, lr, weight_decay, **extra):
        return Lion(params, lr=lr, weight_decay=weight_decay, **extra)
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from gnn_sar.config import OptimizerConfig, SchedulerConfig

__all__ = [
    "OPTIMIZERS",
    "SCHEDULERS",
    "build_optimizer",
    "build_scheduler",
    "parameter_groups",
    "register_optimizer",
    "register_scheduler",
]

OptimizerFactory = Callable[..., Optimizer]
SchedulerFactory = Callable[..., Any]

OPTIMIZERS: dict[str, OptimizerFactory] = {}
"""Name -> factory ``(params, *, lr, weight_decay, **extra) -> Optimizer``."""

SCHEDULERS: dict[str, SchedulerFactory] = {}
"""Name -> factory ``(optimizer, *, max_epochs, **extra) -> scheduler``."""


def register_optimizer(name: str) -> Callable[[OptimizerFactory], OptimizerFactory]:
    """Add an optimiser to :data:`OPTIMIZERS` under ``name``."""

    def decorate(factory: OptimizerFactory) -> OptimizerFactory:
        OPTIMIZERS[name] = factory
        return factory

    return decorate


def register_scheduler(name: str) -> Callable[[SchedulerFactory], SchedulerFactory]:
    """Add a schedule to :data:`SCHEDULERS` under ``name``."""

    def decorate(factory: SchedulerFactory) -> SchedulerFactory:
        SCHEDULERS[name] = factory
        return factory

    return decorate


# --------------------------------------------------------------------------
# Optimisers
# --------------------------------------------------------------------------


@register_optimizer("adamw")
def _adamw(params: Any, *, lr: float, weight_decay: float, **extra: Any) -> Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **extra)


@register_optimizer("adam")
def _adam(params: Any, *, lr: float, weight_decay: float, **extra: Any) -> Optimizer:
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **extra)


@register_optimizer("sgd")
def _sgd(params: Any, *, lr: float, weight_decay: float, **extra: Any) -> Optimizer:
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **extra)


@register_optimizer("rmsprop")
def _rmsprop(params: Any, *, lr: float, weight_decay: float, **extra: Any) -> Optimizer:
    return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **extra)


@register_optimizer("adagrad")
def _adagrad(params: Any, *, lr: float, weight_decay: float, **extra: Any) -> Optimizer:
    return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay, **extra)


@register_optimizer("nadam")
def _nadam(params: Any, *, lr: float, weight_decay: float, **extra: Any) -> Optimizer:
    return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay, **extra)


def parameter_groups(module: nn.Module, config: OptimizerConfig) -> Any:
    """The parameters to optimise, in one group or two.

    One group — every parameter, decayed alike — is the historical behaviour and
    stays the default. Two groups exempt biases and normalisation scales from
    the weight decay: those are one number per channel rather than a direction
    in feature space, and shrinking them mostly costs calibration. It is a cheap
    thing to ablate, which is the only reason it is a setting.
    """
    if config.decay_on_norm_and_bias:
        return module.parameters()
    decayed, spared = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        (spared if parameter.ndim <= 1 or name.endswith(".bias") else decayed).append(
            parameter
        )
    return [
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": spared, "weight_decay": 0.0},
    ]


def build_optimizer(module: nn.Module, config: OptimizerConfig) -> Optimizer:
    """The optimiser :attr:`OptimizerConfig.name` selects.

    Raises:
        ValueError: if the name is not registered, or the optimiser rejects one
            of ``config.params`` — reported with the name of the optimiser, so
            ``momentum`` handed to AdamW reads as a configuration mistake
            rather than a ``TypeError`` from inside torch.
    """
    factory = _lookup(OPTIMIZERS, config.name, "optimizer")
    try:
        return factory(
            parameter_groups(module, config),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **config.params,
        )
    except TypeError as error:
        raise ValueError(
            f"optimizer {config.name!r} does not accept "
            f"{sorted(config.params)}: {error}"
        ) from error


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------


@register_scheduler("cosine")
def _cosine(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> LRScheduler:
    """One cosine arc over the whole run. Reaches ``eta_min`` (0) at the end."""
    extra.setdefault("T_max", max_epochs)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **extra)


@register_scheduler("cosine_warm_restarts")
def _cosine_restarts(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> LRScheduler:
    """Cosine arcs restarting four times over the run."""
    extra.setdefault("T_0", max(1, max_epochs // 4))
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **extra)


@register_scheduler("step")
def _step(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> LRScheduler:
    """Three drops over the run, at torch's ``gamma`` of 0.1 unless overridden.

    Three decades of decay is a lot; ``{"gamma": 0.5}`` is the gentler version,
    and is the reason ``gamma`` is not pinned here.
    """
    extra.setdefault("step_size", max(1, max_epochs // 3))
    return torch.optim.lr_scheduler.StepLR(optimizer, **extra)


@register_scheduler("multistep")
def _multistep(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> LRScheduler:
    """Drops at half and three quarters of the run."""
    extra.setdefault("milestones", [max_epochs // 2, (3 * max_epochs) // 4])
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, **extra)


@register_scheduler("exponential")
def _exponential(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> LRScheduler:
    """A fixed factor per epoch; 0.95 halves the rate roughly every 14."""
    del max_epochs
    extra.setdefault("gamma", 0.95)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, **extra)


@register_scheduler("linear")
def _linear(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> LRScheduler:
    """A straight ramp down to a hundredth of the rate over the run."""
    extra.setdefault("total_iters", max_epochs)
    extra.setdefault("end_factor", 0.01)
    return torch.optim.lr_scheduler.LinearLR(optimizer, **extra)


@register_scheduler("onecycle")
def _onecycle(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> LRScheduler:
    """Warm-up then anneal, peaking at the configured rate.

    ``max_lr`` defaults to each group's own rate, so the configured
    ``learning_rate`` is the peak rather than the start.
    """
    extra.setdefault("total_steps", max_epochs)
    extra.setdefault(
        "max_lr", [group["lr"] for group in optimizer.param_groups]
    )
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, **extra)


@register_scheduler("plateau")
def _plateau(optimizer: Optimizer, *, max_epochs: int, **extra: Any) -> ReduceLROnPlateau:
    """Halve the rate after five epochs without an improvement in ``monitor``."""
    del max_epochs
    extra.setdefault("mode", "min")
    extra.setdefault("factor", 0.5)
    extra.setdefault("patience", 5)
    return ReduceLROnPlateau(optimizer, **extra)


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    max_epochs: int,
    available_metrics: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """The ``lr_scheduler`` entry Lightning expects, or ``None`` for a flat rate.

    Args:
        optimizer: the optimiser the schedule drives.
        config: which schedule, and its arguments.
        max_epochs: the run's length — with one step per epoch this is also the
            total number of steps, which is what ``cosine`` and ``onecycle``
            need and what the caller would otherwise have to work out.
        available_metrics: the metric names the module logs. When given, a
            plateau schedule's ``monitor`` is checked against it, so a typo
            fails before the fit rather than at the first validation epoch.

    Returns:
        ``None`` when the schedule is ``"none"``, otherwise the dict
        ``configure_optimizers`` returns under ``"lr_scheduler"``.
    """
    if not config.enabled:
        return None
    factory = _lookup(SCHEDULERS, config.name, "scheduler")
    try:
        scheduler = factory(optimizer, max_epochs=max_epochs, **config.params)
    except TypeError as error:
        raise ValueError(
            f"scheduler {config.name!r} does not accept {sorted(config.params)}: {error}"
        ) from error

    plateau = isinstance(scheduler, ReduceLROnPlateau)
    if config.warmup_epochs:
        if plateau:
            raise ValueError(
                "warmup_epochs cannot precede the 'plateau' schedule: it reacts "
                "to a metric rather than to the epoch count, so the two cannot "
                "be chained. Use 'cosine' or 'step' for a warm-up run"
            )
        scheduler = _with_warmup(optimizer, scheduler, config)

    entry: dict[str, Any] = {
        "scheduler": scheduler,
        "interval": config.interval,
        "frequency": config.frequency,
    }
    if plateau:
        if available_metrics is not None and config.monitor not in set(available_metrics):
            raise ValueError(
                f"the 'plateau' schedule monitors {config.monitor!r}, which this "
                f"module never logs; it logs {sorted(available_metrics)}"
            )
        entry["monitor"] = config.monitor
    return entry


def _with_warmup(
    optimizer: Optimizer, scheduler: LRScheduler, config: SchedulerConfig
) -> LRScheduler:
    """Linear warm-up chained in front of the real schedule."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, scheduler], milestones=[config.warmup_epochs]
    )


def _lookup(registry: dict[str, Any], name: str, kind: str) -> Any:
    if name not in registry:
        raise ValueError(
            f"unknown {kind} {name!r}, expected one of {sorted(registry)}"
        )
    return registry[name]

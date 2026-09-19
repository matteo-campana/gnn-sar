"""Parameter sweeps: one config, many runs, one MLflow experiment.

A sweep file is a base :class:`~gnn_sar.config.ExperimentConfig` plus a grid of
dotted keys to vary over it — the same keys ``--set`` takes::

    {
      "command": "demo",
      "argv": ["--models", "stgnn", "--max-epochs", "20"],
      "base": {"training": {"max_epochs": 20}},
      "grid": {
        "training.optimizer.learning_rate": [1e-3, 3e-4],
        "training.scheduler.name": ["none", "cosine"],
        "training.loss.smoothness_weight": [0.0, 0.05, 0.2]
      },
      "seeds": [0, 1, 2],
      "strategy": "grid"
    }

Each point becomes an ordinary ``demo`` or ``train`` invocation with its own
MLflow run, so a sweep is filterable in the UI by the very parameters it varied
— they are part of the config every run already logs. Nothing here fits or
evaluates anything itself; it builds configs and calls the same two entry points
a person would.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from gnn_sar.config import ExperimentConfig, apply_overrides

__all__ = ["SweepConfig", "SweepPoint", "expand_sweep", "load_sweep"]


@dataclass(frozen=True)
class SweepPoint:
    """One run of a sweep: the config it uses and what made it different."""

    index: int
    overrides: dict[str, Any]
    seed: int | None
    config: ExperimentConfig

    @property
    def label(self) -> str:
        """A short, stable name for the MLflow run.

        Built from the varied values only — the constant part of the sweep is
        the same in every run and would just make the names longer. Keys keep
        their last two segments, because the last alone is often the same word
        in two different sections (``scheduler.name`` and ``optimizer.name``).
        """
        parts = [
            f"{'.'.join(key.split('.')[-2:])}={_render(value)}"
            for key, value in self.overrides.items()
        ]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return "-".join(parts) if parts else f"point{self.index}"


@dataclass(frozen=True)
class SweepConfig:
    """A sweep as read from its JSON file.

    Attributes:
        command: ``demo`` or ``train`` — which entry point each point runs.
        argv: the command's own arguments, exactly as they would be typed,
            positionals included — a ``train`` sweep starts with the path of
            the ingested study area. They are parsed by the real parser, so a
            mistake here is the same error it would be on the command line.
        base: the config every point starts from.
        grid: dotted key -> the values to try.
        seeds: repeat every point once per seed. A single-seed sweep measures
            the settings and the initialisation together, which is the usual
            way to mistake noise for an effect.
        strategy: ``grid`` is the full product; ``random`` draws ``n_samples``
            combinations from it.
        n_samples: how many, for ``random``.
        max_runs: a hard cap, applied after expansion. A grid grows faster than
            it looks.
    """

    command: str = "demo"
    argv: tuple[str, ...] = ()
    base: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()
    strategy: str = "grid"
    n_samples: int = 10
    max_runs: int | None = None
    seed: int = 0
    """Seed of the *sampler*, so a random sweep is itself reproducible."""

    def __post_init__(self) -> None:
        if self.command not in ("demo", "train"):
            raise ValueError(f"a sweep runs 'demo' or 'train', got {self.command!r}")
        if self.strategy not in ("grid", "random"):
            raise ValueError(f"strategy must be 'grid' or 'random', got {self.strategy!r}")
        empty = sorted(key for key, values in self.grid.items() if not values)
        if empty:
            raise ValueError(f"these grid axes have no values to try: {empty}")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("max_runs must be >= 1 or absent")


def load_sweep(path: str | Path) -> SweepConfig:
    """Read a sweep file, reporting an unknown key rather than ignoring it."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {
        "command",
        "argv",
        "base",
        "grid",
        "seeds",
        "strategy",
        "n_samples",
        "max_runs",
        "seed",
    }
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown keys in the sweep file: {sorted(unknown)}")
    payload["argv"] = tuple(payload.get("argv", ()))
    payload["seeds"] = tuple(payload.get("seeds", ()))
    return SweepConfig(**payload)


def expand_sweep(sweep: SweepConfig) -> list[SweepPoint]:
    """The runs a sweep describes, in the order they will be executed.

    Raises:
        ValueError: if a combination is not a valid config. The whole sweep is
            expanded — and therefore validated — before the first run starts,
            so an impossible corner of the grid is found in a second rather
            than after an hour of fitting.
    """
    base = ExperimentConfig.from_dict(sweep.base) if sweep.base else ExperimentConfig()
    points: list[SweepPoint] = []
    for index, (overrides, seed) in enumerate(_combinations(sweep)):
        try:
            config = apply_overrides(base, overrides) if overrides else base
        except ValueError as error:
            raise ValueError(f"sweep point {index} ({overrides}) is not valid: {error}") from error
        points.append(SweepPoint(index=index, overrides=overrides, seed=seed, config=config))
        if sweep.max_runs is not None and len(points) >= sweep.max_runs:
            break
    return points


def _combinations(sweep: SweepConfig) -> Iterator[tuple[dict[str, Any], int | None]]:
    keys = list(sweep.grid)
    if sweep.strategy == "grid":
        values = [sweep.grid[key] for key in keys]
        assignments = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    else:
        rng = random.Random(sweep.seed)
        assignments = [
            {key: rng.choice(sweep.grid[key]) for key in keys} for _ in range(sweep.n_samples)
        ]
    seeds: tuple[int | None, ...] = sweep.seeds or (None,)
    for assignment in assignments:
        for seed in seeds:
            yield assignment, seed


def _render(value: Any) -> str:
    """A grid value as a filename-safe fragment of a run name."""
    text = value if isinstance(value, str) else json.dumps(value)
    return "".join(character if character.isalnum() or character in "._-" else "" for character in text)

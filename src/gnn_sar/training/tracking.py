"""MLflow tracking for every run the project produces.

One run records a whole experiment: the resolved :class:`ExperimentConfig` as
parameters, the Lightning metrics as they are logged, and the artefacts that
make the run citable afterwards — the config itself, ``metrics.json``, the
checkpoint and the sub-annual GeoTIFFs.

The tracking URI resolves in three steps, from the most explicit to the most
convenient:

1. whatever the caller passed (``--tracking-uri``, or ``TrackingConfig``),
2. ``MLFLOW_TRACKING_URI``, which is how a remote server is normally wired in,
3. a local ``./mlruns`` file store, so ``mlflow ui`` works with no setup at all.

MLflow 3 puts that file store in maintenance mode and refuses it unless
``MLFLOW_ALLOW_FILE_STORE`` is set, steering users to a database backend. For a
research project the file store is still the right default — a run is a directory
you can archive next to the thesis chapter that cites it — so
:func:`allow_local_file_store` sets that opt-out for the current process when the
resolved URI is a local path. Passing ``--tracking-uri sqlite:///mlflow.db`` is
the backend MLflow itself recommends, and needs nothing special.

Everything goes through :class:`RunTracker`, which is also usable in its
disabled form. That keeps the call sites free of ``if tracking:`` branches: a
disabled tracker accepts every call and does nothing, and hands the Trainer
``False`` instead of a logger.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from pytorch_lightning.loggers import MLFlowLogger

from gnn_sar.config import TrackingConfig

__all__ = [
    "DEFAULT_LOCAL_STORE",
    "allow_local_file_store",
    "RunTracker",
    "describe_destination",
    "flatten_params",
    "open_run",
    "resolve_tracking_uri",
    "tracked_run",
]

DEFAULT_LOCAL_STORE = "mlruns"
"""Directory of the fallback file store, relative to the working directory."""

_FILE_STORE_OPT_OUT = "MLFLOW_ALLOW_FILE_STORE"


def allow_local_file_store() -> None:
    """Opt this process out of MLflow 3's refusal to use a file-based store.

    Only ever called for a ``file:`` URI, i.e. one this project defaulted to or
    the user asked for by giving a path. An existing value is left alone, so
    someone who set the variable to ``false`` on purpose keeps that decision.
    """
    os.environ.setdefault(_FILE_STORE_OPT_OUT, "true")


def resolve_tracking_uri(uri: str | None = None) -> str:
    """The URI a run will be written to, as an absolute ``file:`` URI or a server.

    A relative path is made absolute on purpose: Lightning and MLflow resolve
    relative stores against the *current* working directory, so a run started
    from a subdirectory would otherwise silently open a second store.
    """
    candidate = uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_LOCAL_STORE
    if "://" in candidate:
        return candidate
    return Path(candidate).expanduser().resolve().as_uri()


def flatten_params(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config into the dotted keys MLflow stores as parameters.

    ``{"training": {"loss": {"positive_weight": 1.0}}}`` becomes
    ``{"training.loss.positive_weight": 1.0}``. MLflow keeps parameters flat, so
    without this the nested sections would be logged as one unsearchable blob.

    A list of sections — a configured objective's ``terms`` — is indexed by
    position (``training.loss.terms.0.weight``), so two runs that composed the
    objective differently can be compared term by term rather than as two
    opaque strings.
    """
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    if not isinstance(payload, dict):
        return {prefix or "value": payload}
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        flat.update(_flatten_value(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _flatten_value(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, dict) or (is_dataclass(value) and not isinstance(value, type)):
        return flatten_params(value, name)
    if isinstance(value, (list, tuple)) and any(
        isinstance(entry, dict) or (is_dataclass(entry) and not isinstance(entry, type))
        for entry in value
    ):
        flat: dict[str, Any] = {}
        for index, entry in enumerate(value):
            flat.update(_flatten_value(entry, f"{name}.{index}"))
        return flat
    return {name: value}


class RunTracker:
    """An MLflow run, or a no-op standing in for one.

    Args:
        logger: the Lightning logger bound to the run, or ``None`` when tracking
            is disabled.
    """

    def __init__(self, logger: MLFlowLogger | None) -> None:
        self._logger = logger

    @property
    def enabled(self) -> bool:
        return self._logger is not None

    @property
    def logger(self) -> MLFlowLogger | None:
        return self._logger

    @property
    def trainer_logger(self) -> MLFlowLogger | bool:
        """What to hand :class:`pytorch_lightning.Trainer`.

        Lightning wants ``False``, not ``None``, to mean "no logging at all";
        ``None`` would fall back to the default TensorBoard/CSV logger.
        """
        return self._logger if self._logger is not None else False

    @property
    def run_id(self) -> str | None:
        return self._logger.run_id if self._logger is not None else None

    def log_params(self, params: Any) -> None:
        """Record a config (dataclass or mapping) as flat MLflow parameters."""
        if self._logger is None:
            return
        self._logger.log_hyperparams(flatten_params(params))

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Record metrics computed outside a Trainer, e.g. the random forest's."""
        if self._logger is None:
            return
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and float(value) == float(value)
        }
        if numeric:
            self._logger.log_metrics(numeric, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._logger is None:
            return
        for key, value in tags.items():
            self._logger.experiment.set_tag(self._logger.run_id, key, str(value))

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Attach a file already on disk to the run."""
        if self._logger is None:
            return
        local = Path(path)
        if not local.exists():
            return
        self._logger.experiment.log_artifact(
            self._logger.run_id, str(local), artifact_path=artifact_path
        )

    def log_artifacts(
        self, paths: list[str | Path] | tuple[str | Path, ...], artifact_path: str | None = None
    ) -> None:
        for path in paths:
            self.log_artifact(path, artifact_path=artifact_path)

    def log_json(self, payload: Any, name: str, artifact_path: str | None = None) -> None:
        """Attach a dict as a JSON artefact without the caller touching a temp file."""
        if self._logger is None:
            return
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / name
            local.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            self.log_artifact(local, artifact_path=artifact_path)

    def finish(self, status: str = "FINISHED") -> None:
        """Close the run.

        Lightning already terminates the run when ``fit`` and ``test`` tear
        down, so this is called last, after the artefacts, to leave the run with
        the status that reflects the whole command rather than its first stage.
        """
        if self._logger is None:
            return
        self._logger.experiment.set_terminated(self._logger.run_id, status)


def open_run(
    config: TrackingConfig | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> RunTracker:
    """Start an MLflow run and return the tracker bound to it.

    A disabled :class:`TrackingConfig` yields a no-op tracker, so callers do not
    branch on whether tracking is on.
    """
    config = config or TrackingConfig()
    if not config.enabled:
        return RunTracker(None)
    uri = resolve_tracking_uri(config.tracking_uri)
    if uri.startswith("file:"):
        allow_local_file_store()
    logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        run_name=run_name or config.run_name,
        tracking_uri=uri,
        tags=tags,
    )
    # Touching ``.experiment`` creates the run now rather than at the first
    # logged metric, so ``run_id`` is available to the caller straight away.
    _ = logger.experiment
    return RunTracker(logger)


@contextmanager
def tracked_run(
    config: TrackingConfig | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[RunTracker]:
    """``open_run`` as a context manager that marks a failed run as ``FAILED``."""
    tracker = open_run(config, run_name=run_name, tags=tags)
    try:
        yield tracker
    except BaseException:
        tracker.finish("FAILED")
        raise
    tracker.finish("FINISHED")


def describe_destination(config: TrackingConfig) -> str:
    """One line for the console saying where the run is being recorded."""
    if not config.enabled:
        return "tracking disabled"
    uri = resolve_tracking_uri(config.tracking_uri)
    return f"mlflow: experiment {config.experiment_name!r} at {uri}"

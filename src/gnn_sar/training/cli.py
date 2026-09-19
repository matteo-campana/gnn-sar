"""Command line entry point.

Seven subcommands, in the order a real study goes through them:

``gnn-sar sites``
    the pilot study areas — Roma, Viterbo, Milano — with the one grid every
    input of an area has to land on.
``gnn-sar search``
    query the Copernicus Data Space Ecosystem catalogue for the Sentinel-1 and
    Sentinel-2 products covering an area and a period, and optionally download
    them.
``gnn-sar preprocess``
    the local ingestor: raw products through SNAP (Sentinel-1, coherence) or
    straight onto the study grid (Sentinel-2 L2A).
``gnn-sar ingest``
    turn a directory of preprocessed GeoTIFFs into a cached study area — nodes,
    edges, composites, annual labels — following a JSON config.
``gnn-sar train``
    train and evaluate on such a study area, and write the sub-annual maps back
    out as GeoTIFFs.
``gnn-sar demo``
    the same training and evaluation on the synthetic scenario, with the
    baselines, needing no data at all. It is the fastest way to check that an
    installation works and to see what the comparison of OS3 reports.
``gnn-sar sweep``
    one ``train`` or ``demo`` invocation per point of a parameter grid, each as
    its own MLflow run. This is how the objective, the optimiser and the
    schedule get tuned rather than guessed.

``docs/guida_dati_sentinel.md`` walks through the first four end to end.

Both commands that train — ``train`` and ``demo`` — share the same runtime
flags. ``--accelerator``/``--devices``/``--precision`` are handed straight to
Lightning, so a GPU run is the default invocation with nothing added, and
``--tracking-uri``/``--experiment-name``/``--run-name`` decide where MLflow
records the run. Every run is tracked unless ``--no-tracking`` says otherwise.

They also share the optimisation flags. ``--optimizer``, ``--learning-rate``,
``--scheduler`` and ``--loss-aggregation`` name the four choices a run changes
most often; ``--set KEY=VALUE`` reaches **any** field of the config by its
dotted name, which is the same address a sweep file uses::

    gnn-sar demo --optimizer sgd --set training.optimizer.params.momentum=0.9 \
                 --scheduler cosine --set training.loss.smoothness_weight=0.2

Nothing is hard-coded behind those names: the optimisers live in
:mod:`gnn_sar.training.optim`, the loss terms in :mod:`gnn_sar.losses.registry`,
and both registries accept new entries from outside the package.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from gnn_sar.config import (
    PRECISIONS,
    ExperimentConfig,
    ModelConfig,
    apply_overrides,
)
from gnn_sar.data.synthetic import make_synthetic_scenario
from gnn_sar.losses.aggregation import AGGREGATIONS
from gnn_sar.losses.registry import LOSS_TERMS
from gnn_sar.models.baselines import (
    MLPBaseline,
    RandomForestBaseline,
    TemporalBaseline,
)
from gnn_sar.models.stgnn import SpatioTemporalGNN
from gnn_sar.training.datamodule import SpatioTemporalGraphDataModule
from gnn_sar.training.lightning_module import SoilSealingModule
from gnn_sar.training.optim import OPTIMIZERS, SCHEDULERS
from gnn_sar.training.sweep import SweepPoint, expand_sweep, load_sweep
from gnn_sar.training.tracking import RunTracker, describe_destination, tracked_run

__all__ = [
    "main",
    "run_demo",
    "run_search",
    "run_preprocess",
    "run_ingest_command",
    "run_sweep",
    "run_training",
    "build_parser",
]

_MODELS = {
    "stgnn": SpatioTemporalGNN,
    "temporal": TemporalBaseline,
    "mlp": MLPBaseline,
}
_RANDOM_FOREST = "rf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnn-sar", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="train and evaluate on the synthetic land-take scenario"
    )
    demo.add_argument(
        "--models",
        nargs="+",
        default=["stgnn", "temporal"],
        choices=sorted([*_MODELS, _RANDOM_FOREST]),
        help="which models to run; the gap between them is the ablation of OS3",
    )
    demo.add_argument("--config", type=Path, help="JSON experiment config")
    demo.add_argument("--max-epochs", type=int, help="override the config's max_epochs")
    demo.add_argument("--nodes-per-side", type=int, default=20)
    demo.add_argument("--n-time", type=int, default=24)
    demo.add_argument("--periods-per-year", type=int, default=12)
    demo.add_argument(
        "--contrast",
        type=float,
        default=0.3,
        help="how far the change signature sits from the speckle; 1.0 is an "
        "easy case, 0.3 (the default) is one where the relational information "
        "has something to contribute",
    )
    demo.add_argument(
        "--exact-labels",
        action="store_true",
        help="train on exact sub-annual labels instead of the weak annual ones; "
        "the difference is the price of weak supervision",
    )
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--output", type=Path, help="write the results as JSON")
    _add_runtime_arguments(demo)

    _add_sites_parser(subparsers)
    _add_search_parser(subparsers)
    _add_preprocess_parser(subparsers)
    _add_ingest_parser(subparsers)
    _add_train_parser(subparsers)
    _add_sweep_parser(subparsers)
    return parser


def _add_sweep_parser(subparsers: Any) -> None:
    sweep = subparsers.add_parser(
        "sweep",
        help="run one demo or train invocation per point of a parameter grid",
    )
    sweep.add_argument("config", type=Path, help="JSON sweep file")
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="print the runs the grid expands to and stop; every point is "
        "validated, so this catches an impossible combination in a second",
    )
    sweep.add_argument("--output", type=Path, help="write the results as JSON")


def _add_sites_parser(subparsers: Any) -> None:
    sites = subparsers.add_parser(
        "sites", help="the pilot study areas and the grid every input must share"
    )
    sites.add_argument("name", nargs="?", help="show only this one")


def _add_search_parser(subparsers: Any) -> None:
    search = subparsers.add_parser(
        "search",
        help="query the Copernicus Data Space Ecosystem catalogue (and download)",
    )
    search.add_argument(
        "--collection", default="SENTINEL-1", choices=["SENTINEL-1", "SENTINEL-2"]
    )
    search.add_argument(
        "--site",
        help="a pilot area (see `gnn-sar sites`), which fills in the bounding box",
    )
    search.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="area of interest in degrees, WGS84; required unless --site is given",
    )
    search.add_argument("--start", required=True, help="first sensing date, YYYY-MM-DD")
    search.add_argument("--end", required=True, help="last sensing date (exclusive)")
    search.add_argument(
        "--product-type",
        help="e.g. IW_GRDH_1S for Sentinel-1 GRD, S2MSI2A for Sentinel-2 L2A",
    )
    search.add_argument(
        "--orbit",
        type=int,
        help="relative orbit number; pinning one keeps the viewing geometry "
        "constant, which a backscatter time series needs",
    )
    search.add_argument(
        "--orbit-direction", choices=["ASCENDING", "DESCENDING"], help="pass direction"
    )
    search.add_argument(
        "--max-cloud-cover", type=float, help="Sentinel-2 only, in percent"
    )
    search.add_argument("--limit", type=int, default=100, help="page size, max 1000")
    search.add_argument(
        "--download",
        type=Path,
        help="download every result into this directory (needs CDSE_USERNAME "
        "and CDSE_PASSWORD)",
    )
    search.add_argument("--output", type=Path, help="write the results as JSON")


def _add_preprocess_parser(subparsers: Any) -> None:
    preprocess = subparsers.add_parser(
        "preprocess",
        help="raw Copernicus products to rasters on a study area's grid",
    )
    preprocess.add_argument(
        "kind",
        choices=["s1", "s2", "coherence"],
        help="s1: SNAP on GRD/RTC products. s2: stack an L2A product, no SNAP. "
        "coherence: SNAP on SLC pairs at a fixed temporal baseline",
    )
    preprocess.add_argument("--site", required=True, help="the study area")
    preprocess.add_argument(
        "--input", type=Path, required=True, help="directory of downloaded products"
    )
    preprocess.add_argument(
        "--output", type=Path, required=True, help="where the aligned rasters go"
    )
    preprocess.add_argument(
        "--pattern",
        default="*",
        help="glob selecting the products inside --input (default: everything)",
    )
    preprocess.add_argument(
        "--graph", type=Path, help="SNAP graph; defaults to the one shipped in docs/esempi"
    )
    preprocess.add_argument("--gpt", type=Path, help="path to SNAP's gpt executable")
    preprocess.add_argument("--memory", default="8G", help="JVM heap given to SNAP")
    preprocess.add_argument("--threads", type=int, default=4, help="SNAP's -q")
    preprocess.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="products processed at once; SNAP already threads internally, so "
        "raising this mostly multiplies the memory",
    )
    preprocess.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra graph variable, repeatable (e.g. --parameter subswath=IW2)",
    )
    preprocess.add_argument(
        "--baseline-days", type=int, default=12, help="coherence: temporal baseline"
    )
    preprocess.add_argument(
        "--tolerance-days", type=int, default=2, help="coherence: baseline tolerance"
    )
    preprocess.add_argument(
        "--bands",
        nargs="+",
        help="s2: band codes to stack (default: B02 B03 B04 B08 B11 B12)",
    )
    preprocess.add_argument(
        "--no-scl",
        action="store_true",
        help="s2: skip the scene-classification layer. Without it nothing is "
        "masked and cloud becomes change",
    )
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="keep SNAP's own output next to the aligned raster",
    )


def _add_ingest_parser(subparsers: Any) -> None:
    ingest = subparsers.add_parser(
        "ingest", help="build a study area from preprocessed GeoTIFFs"
    )
    ingest.add_argument("config", type=Path, help="JSON ingestion config")
    ingest.add_argument(
        "--output", type=Path, help="override the config's output path"
    )


def _add_train_parser(subparsers: Any) -> None:
    train = subparsers.add_parser(
        "train", help="train and evaluate on an ingested study area"
    )
    train.add_argument("area", type=Path, help="the .npz written by `gnn-sar ingest`")
    train.add_argument(
        "--model", default="stgnn", choices=sorted([*_MODELS, _RANDOM_FOREST])
    )
    train.add_argument("--config", type=Path, help="JSON experiment config")
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--output-dir",
        type=Path,
        help="write metrics, the checkpoint and the sub-annual maps here",
    )
    train.add_argument(
        "--no-maps",
        action="store_true",
        help="skip the GeoTIFF export even when the area carries a grid",
    )
    _add_runtime_arguments(train)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Hardware and tracking flags, shared by every command that trains.

    All of them default to ``None`` and override the config only when given, so
    a JSON config stays authoritative unless the command line contradicts it.
    """
    hardware = parser.add_argument_group("hardware")
    hardware.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu", "cuda", "mps", "tpu"],
        help="where to run; the default 'auto' picks a GPU when there is one",
    )
    hardware.add_argument(
        "--devices",
        help="how many devices, or 'auto'. A study area is a single full-batch "
        "graph, so more than one device replicates it instead of splitting it",
    )
    hardware.add_argument(
        "--precision",
        choices=list(PRECISIONS),
        help="Lightning precision; 'bf16-mixed' roughly halves the memory a "
        "study area needs on an Ampere or newer card (ignored on CPU)",
    )

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZERS),
        help="which optimiser to fit with (default: the config's, adamw)",
    )
    optimisation.add_argument(
        "--learning-rate", type=float, help="the optimiser's learning rate"
    )
    optimisation.add_argument(
        "--scheduler",
        choices=["none", *sorted(SCHEDULERS)],
        help="learning-rate schedule; a study area is one full-batch graph, so "
        "one step per epoch and the whole schedule spans --max-epochs",
    )
    optimisation.add_argument(
        "--loss-aggregation",
        choices=sorted(AGGREGATIONS),
        help="how a year's composites collapse into the one decision the "
        "annual label supervises",
    )
    optimisation.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override any config field by its dotted name, repeatable "
        "(e.g. --set training.loss.smoothness_weight=0.2 "
        "--set training.optimizer.params.momentum=0.9). The value is read as "
        "JSON, so numbers, booleans and lists work. Applied last, after the "
        "config file and the flags above",
    )

    tracking = parser.add_argument_group("mlflow tracking")
    tracking.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI; defaults to MLFLOW_TRACKING_URI, then to a "
        "local ./mlruns that `mlflow ui` reads as-is",
    )
    tracking.add_argument("--experiment-name", help="MLflow experiment to log under")
    tracking.add_argument("--run-name", help="name of the MLflow run")
    tracking.add_argument(
        "--no-tracking",
        action="store_true",
        help="do not record the run at all",
    )


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """The experiment config, with the command line layered on top of the file.

    Four layers, each more explicit than the last: the defaults, the JSON file
    (or the config a sweep handed down in ``args.resolved_config``), the typed
    flags, and finally ``--set``, which can reach any field at all and so has
    the last word.
    """
    config = getattr(args, "resolved_config", None)
    if config is None:
        config = (
            ExperimentConfig.from_json(args.config)
            if getattr(args, "config", None)
            else ExperimentConfig()
        )
    training_overrides: dict[str, Any] = {}
    if args.accelerator:
        training_overrides["accelerator"] = args.accelerator
    if args.devices:
        training_overrides["devices"] = (
            int(args.devices) if str(args.devices).isdigit() else args.devices
        )
    if args.precision:
        training_overrides["precision"] = args.precision
    if getattr(args, "max_epochs", None):
        # Folded into the config rather than kept beside it, so the value MLflow
        # records as ``training.max_epochs`` is the one the Trainer ran.
        training_overrides["max_epochs"] = args.max_epochs

    tracking_overrides: dict[str, Any] = {}
    if args.no_tracking:
        tracking_overrides["enabled"] = False
    if args.tracking_uri:
        tracking_overrides["tracking_uri"] = args.tracking_uri
    if args.experiment_name:
        tracking_overrides["experiment_name"] = args.experiment_name
    if args.run_name:
        tracking_overrides["run_name"] = args.run_name

    config = replace(
        config,
        training=replace(config.training, **training_overrides),
        tracking=replace(config.tracking, **tracking_overrides),
    )
    # A bad override is a typing mistake at the prompt, not a bug: it deserves
    # the message on its own rather than a traceback around it.
    try:
        config = apply_overrides(config, _optimisation_overrides(args))
        config = apply_overrides(config, _parse_overrides(getattr(args, "overrides", [])))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    _validate_registries(config)
    return config


def _optimisation_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The typed optimisation flags, as the dotted keys they are shorthand for."""
    flags = {
        "training.optimizer.name": getattr(args, "optimizer", None),
        "training.optimizer.learning_rate": getattr(args, "learning_rate", None),
        "training.scheduler.name": getattr(args, "scheduler", None),
        "training.loss.aggregation": getattr(args, "loss_aggregation", None),
    }
    return {key: value for key, value in flags.items() if value is not None}


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """``--set key=value`` pairs, with the value read as JSON where it can be.

    JSON first means ``0.2`` is a number, ``true`` is a boolean and ``[1, 2]``
    is a list; a bare word that is not valid JSON stays the string it looks
    like, so ``--set training.optimizer.name=sgd`` needs no quoting.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key.strip()] = value
    return overrides


def _validate_registries(config: ExperimentConfig) -> None:
    """Check every name the config selects before anything is built.

    The registries cannot be reached from :mod:`gnn_sar.config` without a cycle,
    so the names it carries are plain strings there. Resolving them here means a
    typo costs a second at start-up rather than an epoch of fitting — which is
    what makes a long sweep worth launching unattended.
    """
    training = config.training
    for name, registry, kind in (
        (training.optimizer.name, OPTIMIZERS, "optimizer"),
        (training.loss.aggregation, AGGREGATIONS, "loss aggregation"),
        (training.loss.effective_metric_aggregation, AGGREGATIONS, "metric aggregation"),
        *[(spec.name, LOSS_TERMS, "loss term") for spec in training.loss.resolved_terms()],
    ):
        if name not in registry:
            raise SystemExit(
                f"unknown {kind} {name!r}, expected one of {sorted(registry)}"
            )
    if training.scheduler.enabled and training.scheduler.name not in SCHEDULERS:
        raise SystemExit(
            f"unknown scheduler {training.scheduler.name!r}, expected 'none' or "
            f"one of {sorted(SCHEDULERS)}"
        )


def _build_trainer(config: ExperimentConfig, tracker: RunTracker) -> pl.Trainer:
    """The one Trainer every command uses.

    Lightning owns the device placement: the accelerator, the device count and
    the precision come from the config, and the datamodule's transfer hook puts
    the graph where the model already is. Nothing below this function mentions
    ``cuda``.
    """
    training = config.training
    torch.set_float32_matmul_precision(training.matmul_precision)
    # Without this the schedule would be invisible: a run would record the rate
    # it was configured with rather than the one it actually trained at.
    callbacks = (
        [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
        if training.scheduler.enabled and tracker.enabled
        else []
    )
    return pl.Trainer(
        max_epochs=training.max_epochs,
        callbacks=callbacks,
        accelerator=training.accelerator,
        devices=training.devices,
        precision=training.precision,
        gradient_clip_val=training.gradient_clip_val,
        log_every_n_steps=training.log_every_n_steps,
        # ``warn`` rather than ``True``: some of the scatter kernels
        # torch-geometric dispatches to have no deterministic implementation, and
        # hard-failing on them would make the flag unusable.
        deterministic="warn" if training.deterministic else False,
        # The checkpoint is written explicitly, once, next to the metrics, so
        # Lightning's own checkpoint callback would only duplicate it.
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=tracker.trainer_logger,
    )


def _describe_runtime(trainer: pl.Trainer, config: ExperimentConfig) -> str:
    """One line naming the device actually selected, and where the run is logged."""
    device = trainer.strategy.root_device
    training = config.training
    schedule = (
        training.scheduler.name if training.scheduler.enabled else "flat rate"
    )
    return (
        f"runtime: {device.type} x{trainer.num_devices}, "
        f"precision {training.precision} | "
        f"{training.optimizer.name} lr={training.optimizer.learning_rate:g} "
        f"({schedule}) | {describe_destination(config.tracking)}"
    )


def _build_module(
    name: str, config: ExperimentConfig, n_features: int
) -> SoilSealingModule:
    """The Lightning module for one of the compared models."""
    model_config = ModelConfig(**{**config.model.__dict__, "in_channels": n_features})
    return SoilSealingModule(
        model=_MODELS[name](model_config), training_config=config.training
    )


def run_demo(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Run the synthetic experiment and return one metric dict per model.

    Each model gets its own MLflow run, tagged with its name, so the ablation of
    OS3 is a single filter in the MLflow UI rather than a diff of stdout.
    """
    config = _resolve_config(args)
    pl.seed_everything(args.seed, workers=True)
    print(describe_destination(config.tracking))

    block = 4
    side = args.nodes_per_side * block
    scenario = make_synthetic_scenario(
        height=side,
        width=side,
        block=block,
        n_time=args.n_time,
        periods_per_year=args.periods_per_year,
        contrast=args.contrast,
        graph_config=config.graph,
        seed=args.seed,
    )
    graph = scenario.with_exact_labels() if args.exact_labels else scenario.graph
    scenario_params = {
        "scenario.nodes_per_side": args.nodes_per_side,
        "scenario.n_time": args.n_time,
        "scenario.periods_per_year": args.periods_per_year,
        "scenario.contrast": args.contrast,
        "scenario.exact_labels": args.exact_labels,
        "run.seed": args.seed,
    }

    results: dict[str, dict[str, float]] = {}
    for name in args.models:
        run_name = f"{args.run_name}-{name}" if args.run_name else name
        with tracked_run(
            config.tracking,
            run_name=run_name,
            tags={"command": "demo", "model": name, "data": "synthetic"},
        ) as tracker:
            tracker.log_params(config)
            tracker.log_params({**scenario_params, "run.model": name})
            metrics = _demo_one_model(name, graph, scenario, config, tracker)
            tracker.log_json(metrics, "metrics.json")
            results[name] = metrics
    return results


def _demo_one_model(
    name: str,
    graph: Any,
    scenario: Any,
    config: ExperimentConfig,
    tracker: RunTracker,
) -> dict[str, float]:
    """Fit and score one model of the comparison inside an open MLflow run."""
    datamodule = SpatioTemporalGraphDataModule(
        graph,
        graph_config=config.graph,
        prediction_horizon=config.model.prediction_horizon,
    )
    datamodule.setup()
    if name == _RANDOM_FOREST:
        # No Trainer, so its metrics have to be logged by hand.
        metrics = _run_random_forest(graph, datamodule)
        tracker.log_metrics(metrics)
        return metrics

    module = _build_module(name, config, datamodule.n_features)
    trainer = _build_trainer(config, tracker)
    print(_describe_runtime(trainer, config))
    trainer.fit(module, datamodule=datamodule)
    metrics = trainer.test(module, datamodule=datamodule, verbose=False)[0]

    # The exact onsets are never used for training; they are the yardstick.
    against_truth = _delay_against_truth(trainer, module, datamodule, scenario)
    tracker.log_metrics(against_truth)
    metrics.update(against_truth)
    return {key: float(value) for key, value in metrics.items()}


def _run_random_forest(
    graph: Any, datamodule: SpatioTemporalGraphDataModule
) -> dict[str, float]:
    """Fit the classical baseline and score it on the same annual target.

    It gets no delay metrics: a per-year decision has no date to be late about.
    """
    from gnn_sar.metrics.classification import sealing_metrics

    forest = RandomForestBaseline().fit(
        graph.features, graph.year_index, graph.annual_labels, datamodule.masks["train"]
    )
    probability = forest.predict_proba(graph.features, graph.year_index)
    test_mask = datamodule.masks["test"]
    metrics = sealing_metrics(
        (probability[:, test_mask] >= 0.5).astype(int),
        graph.annual_labels[:, test_mask],
    )
    return {f"test/annual_{key}": float(value) for key, value in metrics.items()}


def _delay_against_truth(
    trainer: pl.Trainer,
    module: SoilSealingModule,
    datamodule: SpatioTemporalGraphDataModule,
    scenario: Any,
) -> dict[str, float]:
    """Detection delay against the scenario's true onsets.

    The maps come from ``Trainer.predict`` rather than from the module directly,
    so the graph is moved to whatever device the fit ran on by the same hook the
    training loop used.
    """
    from gnn_sar.metrics.delay import detection_delay

    maps = trainer.predict(module, datamodule=datamodule)[0]
    mask = datamodule.masks["test"]
    delay = detection_delay(
        maps["onset"][mask], scenario.onset[mask], n_time=scenario.graph.n_time
    )
    return {f"truth/delay_{key}": value for key, value in delay.items()}


def run_sites(args: argparse.Namespace) -> str:
    """Print the pilot areas, or one of them."""
    from gnn_sar.data.sites import format_sites, get_site

    rendered = get_site(args.name).describe() if args.name else format_sites()
    print(rendered)
    return rendered


def run_search(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Query the CDSE catalogue, print the products and optionally fetch them."""
    from gnn_sar.data import cdse

    if args.site:
        from gnn_sar.data.sites import get_site

        bbox = list(get_site(args.site).bbox)
    elif args.bbox:
        bbox = list(args.bbox)
    else:
        raise SystemExit("give either --site or --bbox")
    args.bbox = bbox

    attributes: dict[str, Any] = {}
    if args.orbit is not None:
        attributes["relativeOrbitNumber"] = int(args.orbit)
    if args.orbit_direction:
        attributes["orbitDirection"] = args.orbit_direction

    expression = cdse.product_filter(
        collection=args.collection,
        start=args.start,
        end=args.end,
        footprint_wkt=cdse.bbox_to_wkt(*bbox),
        product_type=args.product_type,
        attributes=attributes or None,
        max_cloud_cover=args.max_cloud_cover,
    )
    products = cdse.search_products(expression, top=args.limit, expand_attributes=True)
    print(cdse.format_products(products))
    volume = sum(float(p.get("ContentLength", 0)) for p in products) / 1e9
    print(f"\n{len(products)} products, {volume:.1f} GB in total")

    if args.download and products:
        token = cdse.access_token()
        for position, product in enumerate(products, start=1):
            destination = Path(args.download) / f"{product['Name']}.zip"
            print(f"[{position}/{len(products)}] {destination.name}")
            cdse.download_product(product["Id"], destination, token)
    return products


def run_preprocess(args: argparse.Namespace) -> Any:
    """Run the local preprocessing chain for one sensor over one study area."""
    from gnn_sar.data import preprocess as steps
    from gnn_sar.data.sites import get_site

    site = get_site(args.site)
    products = sorted(p for p in Path(args.input).glob(args.pattern) if p.exists())
    if not products:
        raise SystemExit(f"no product matches {args.pattern!r} in {args.input}")
    parameters = dict(item.split("=", 1) for item in args.parameter)
    print(f"site   {site.name}: {site.grid().describe()}")
    print(f"input  {len(products)} products from {args.input}")

    if args.kind == "s2":
        report = steps.preprocess_sentinel2(
            products,
            args.output,
            site,
            bands=args.bands or steps.S2_DEFAULT_BANDS,
            scl=not args.no_scl,
            jobs=args.jobs,
            overwrite=args.overwrite,
        )
    elif args.kind == "s1":
        report = steps.preprocess_sentinel1(
            products,
            args.output,
            site,
            graph=args.graph or _shipped_graph("snap_s1_grd_preprocessing.xml"),
            gpt=args.gpt,
            memory=args.memory,
            threads=args.threads,
            jobs=args.jobs,
            overwrite=args.overwrite,
            keep_intermediate=args.keep_intermediate,
            parameters=parameters,
        )
    else:
        report = steps.preprocess_coherence(
            products,
            args.output,
            site,
            graph=args.graph or _shipped_graph("snap_s1_slc_coherence.xml"),
            baseline_days=args.baseline_days,
            tolerance_days=args.tolerance_days,
            gpt=args.gpt,
            memory=args.memory,
            threads=args.threads,
            jobs=args.jobs,
            overwrite=args.overwrite,
            keep_intermediate=args.keep_intermediate,
            parameters=parameters,
        )

    print(report.describe())
    return report


def _shipped_graph(name: str) -> Path:
    """The SNAP graph that comes with the repository, if it is there.

    An installed wheel carries no docs/, so a missing file is a clear message
    rather than a confusing SNAP error.
    """
    candidate = Path(__file__).resolve().parents[3] / "docs" / "esempi" / name
    if not candidate.is_file():
        raise SystemExit(
            f"the shipped graph {name} is not next to this installation; pass "
            "--graph with your own (the graphs live in docs/esempi/ of the "
            "repository)"
        )
    return candidate


def run_ingest_command(args: argparse.Namespace) -> Path:
    """Build a study area from a config and write it to disk."""
    from gnn_sar.data.ingest import IngestConfig, run_ingest

    config = IngestConfig.from_json(args.config)
    area = run_ingest(config)
    destination = Path(args.output) if args.output else config.resolve(config.output)
    area.save(destination)
    print("---")
    print(area.describe())
    print(f"\nwritten to {destination}")
    return destination


def run_training(args: argparse.Namespace) -> dict[str, float]:
    """Train one model on an ingested study area and report its metrics.

    Everything the run produced — the config, the metrics, the checkpoint and the
    sub-annual GeoTIFFs — is attached to a single MLflow run, so a result can be
    traced back to the exact settings that made it.
    """
    from gnn_sar.data.study_area import load_study_area

    config = _resolve_config(args)
    pl.seed_everything(args.seed, workers=True)
    print(describe_destination(config.tracking))

    area = load_study_area(args.area)
    print(area.describe())
    datamodule = SpatioTemporalGraphDataModule(
        area.graph,
        graph_config=config.graph,
        prediction_horizon=config.model.prediction_horizon,
    )
    datamodule.setup()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"{args.model}-{Path(args.area).stem}"
    with tracked_run(
        config.tracking,
        run_name=run_name,
        tags={"command": "train", "model": args.model, "area": Path(args.area).name},
    ) as tracker:
        tracker.log_params(config)
        tracker.log_params(
            {
                "run.model": args.model,
                "run.area": str(args.area),
                "run.seed": args.seed,
                "area.n_nodes": area.graph.n_nodes,
                "area.n_time": area.graph.n_time,
                "area.n_features": datamodule.n_features,
            }
        )
        tracker.log_json(config.to_dict(), "config.json")

        if args.model == _RANDOM_FOREST:
            metrics = _run_random_forest(area.graph, datamodule)
            tracker.log_metrics(metrics)
        else:
            metrics = _train_one_model(
                args, config, area, datamodule, output_dir, tracker
            )

        rendered = json.dumps(metrics, indent=2, sort_keys=True)
        if output_dir:
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(rendered, encoding="utf-8")
            tracker.log_artifact(metrics_path)
        else:
            tracker.log_json(metrics, "metrics.json")
    print(rendered)
    return metrics


def _train_one_model(
    args: argparse.Namespace,
    config: ExperimentConfig,
    area: Any,
    datamodule: SpatioTemporalGraphDataModule,
    output_dir: Path | None,
    tracker: RunTracker,
) -> dict[str, float]:
    """Fit, test, and write out the checkpoint and the maps of one model."""
    module = _build_module(args.model, config, datamodule.n_features)
    trainer = _build_trainer(config, tracker)
    print(_describe_runtime(trainer, config))
    trainer.fit(module, datamodule=datamodule)
    metrics = {
        key: float(value)
        for key, value in trainer.test(module, datamodule=datamodule, verbose=False)[
            0
        ].items()
    }
    if output_dir is None:
        return metrics

    checkpoint = output_dir / f"{args.model}.ckpt"
    trainer.save_checkpoint(checkpoint)
    if config.tracking.log_checkpoints:
        tracker.log_artifact(checkpoint, artifact_path="checkpoints")
    if not args.no_maps:
        written = _export_maps(trainer, module, datamodule, area, output_dir)
        if config.tracking.log_maps:
            tracker.log_artifacts(written, artifact_path="maps")
    return metrics


def _export_maps(
    trainer: pl.Trainer,
    module: SoilSealingModule,
    datamodule: SpatioTemporalGraphDataModule,
    area: Any,
    output_dir: Path,
) -> list[Path]:
    """Write the sub-annual products as GeoTIFFs on the study grid.

    Predictions are exported for **every** node, test split included: the split
    is there to make the reported metrics honest, not to blank out two thirds of
    the map an operator would use.

    ``Trainer.predict`` is what produces them, so the forward pass runs on the
    same device and at the same precision as the fit did.

    Returns:
        the files written, for the caller to attach to the MLflow run.
    """
    if area.grid is None:
        print("no georeferencing in the study area, skipping the GeoTIFF export")
        return []
    from gnn_sar.data.raster import write_raster

    maps = trainer.predict(module, datamodule=datamodule)[0]
    geometries = area.geometries
    written: list[Path] = []
    for name, values, dtype, nodata in (
        ("sealed_probability", maps["sealed_probability"], "float32", float("nan")),
        # -1 is the model's own "never detected", so it doubles as nodata.
        ("onset", maps["onset"][None, :], "int16", -1),
        ("early_warning", maps.get("early_warning"), "float32", float("nan")),
    ):
        if values is None:
            continue
        raster = np.stack([geometries.rasterise(layer) for layer in values])
        written.append(
            write_raster(
                output_dir / f"{name}.tif", raster, area.grid, nodata=nodata, dtype=dtype
            )
        )
    composites = ", ".join(area.periods.labels())
    print(
        f"maps written: {', '.join(path.name for path in written)}\n"
        f"band order of the multi-composite maps: {composites}"
    )
    return written


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run every point of a parameter grid, each as its own tracked run.

    The grid is expanded — and therefore fully validated — before the first fit
    starts, so a combination no config can express fails immediately instead of
    after the runs before it have already burned an afternoon.

    A point that raises is reported and the sweep carries on: one divergent
    corner of a grid should not cost the other nineteen.
    """
    sweep = load_sweep(args.config)
    points = expand_sweep(sweep)
    print(f"{len(points)} run(s) from {args.config}")
    for point in points:
        print(f"  [{point.index}] {point.label}")
    if args.dry_run:
        return []

    runner = run_demo if sweep.command == "demo" else run_training
    results: list[dict[str, Any]] = []
    for point in points:
        print(f"\n=== [{point.index + 1}/{len(points)}] {point.label} ===")
        entry: dict[str, Any] = {
            "index": point.index,
            "overrides": point.overrides,
            "seed": point.seed,
        }
        try:
            entry["metrics"] = runner(_point_namespace(sweep, point))
        except Exception as error:  # noqa: BLE001 - one bad point must not end the sweep
            entry["error"] = f"{type(error).__name__}: {error}"
            print(f"point {point.index} failed: {entry['error']}")
        results.append(entry)

    failed = [entry["index"] for entry in results if "error" in entry]
    print(f"\n{len(results) - len(failed)}/{len(results)} run(s) finished")
    if failed:
        print(f"failed: {failed}")
    return results


def _point_namespace(sweep: Any, point: SweepPoint) -> argparse.Namespace:
    """One sweep point as the command-line invocation it stands for.

    The sweep's ``argv`` goes through the real parser, so its arguments mean
    exactly what they mean on the command line and a mistake in the file is the
    same error it would be there. The point's config is attached rather than
    written to a temporary file, which is what ``_resolve_config`` reads it from.
    """
    args = build_parser().parse_args([sweep.command, *sweep.argv])
    args.resolved_config = point.config
    if point.seed is not None:
        args.seed = point.seed
    base_name = args.run_name or sweep.command
    args.run_name = f"{base_name}-{point.label}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "sites":
        run_sites(args)
        return 0
    if args.command == "preprocess":
        run_preprocess(args)
        return 0
    if args.command == "search":
        products = run_search(args)
        if args.output:
            args.output.write_text(
                json.dumps(products, indent=2), encoding="utf-8"
            )
        return 0
    if args.command == "ingest":
        run_ingest_command(args)
        return 0
    if args.command == "train":
        run_training(args)
        return 0
    if args.command == "sweep":
        results = run_sweep(args)
        if args.output:
            args.output.write_text(
                json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
            )
        return 0
    if args.command != "demo":  # pragma: no cover - argparse enforces the choice
        raise SystemExit(f"unknown command {args.command!r}")

    results = run_demo(args)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Command line entry point.

``gnn-sar demo`` runs the whole pipeline on the synthetic scenario — graph
construction, weak-supervision training, evaluation against both the annual and
(where available) the sub-annual truth, and the same run for the baselines. It
is the fastest way to check that an installation works and to see what the
comparison of OS3 actually reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from gnn_sar.config import ExperimentConfig, ModelConfig
from gnn_sar.data.synthetic import make_synthetic_scenario
from gnn_sar.models.baselines import (
    MLPBaseline,
    RandomForestBaseline,
    TemporalBaseline,
)
from gnn_sar.models.stgnn import SpatioTemporalGNN
from gnn_sar.training.datamodule import SpatioTemporalGraphDataModule
from gnn_sar.training.lightning_module import SoilSealingModule

__all__ = ["main", "run_demo", "build_parser"]

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
    demo.add_argument(
        "--features",
        nargs="*",
        default=["radar"],
        choices=["radar", "texture", "attributes"],
        help="which node-feature blocks to assemble (§5.2). 'radar' is the "
        "multitemporal VV/VH/coherence statistics, 'texture' adds per-segment "
        "GLCM, 'attributes' adds shape and proximity to the corridor",
    )
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--output", type=Path, help="write the results as JSON")
    return parser


def run_demo(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Run the synthetic experiment and return one metric dict per model."""
    config = (
        ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    )
    max_epochs = args.max_epochs or config.training.max_epochs
    pl.seed_everything(args.seed, workers=True)

    block = 4
    side = args.nodes_per_side * block
    scenario = make_synthetic_scenario(
        height=side,
        width=side,
        block=block,
        n_time=args.n_time,
        periods_per_year=args.periods_per_year,
        contrast=args.contrast,
        include_texture="texture" in args.features,
        include_attributes="attributes" in args.features,
        graph_config=config.graph,
        seed=args.seed,
    )
    graph = scenario.with_exact_labels() if args.exact_labels else scenario.graph

    results: dict[str, dict[str, float]] = {}
    for name in args.models:
        # Re-seed before each model, not once for the whole loop: otherwise the
        # second model inherits the RNG state the first one left behind and its
        # score depends on the order --models was given in.
        pl.seed_everything(args.seed, workers=True)
        datamodule = SpatioTemporalGraphDataModule(
            graph,
            graph_config=config.graph,
            prediction_horizon=config.model.prediction_horizon,
        )
        datamodule.setup()
        if name == _RANDOM_FOREST:
            results[name] = _run_random_forest(graph, datamodule)
            continue
        model_config = ModelConfig(
            **{**config.model.__dict__, "in_channels": datamodule.n_features}
        )
        module = SoilSealingModule(
            model=_MODELS[name](model_config), training_config=config.training
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=config.training.accelerator,
            devices=config.training.devices,
            gradient_clip_val=config.training.gradient_clip_val,
            log_every_n_steps=config.training.log_every_n_steps,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(module, datamodule=datamodule)
        metrics = trainer.test(module, datamodule=datamodule, verbose=False)[0]

        # The exact onsets are never used for training; they are the yardstick.
        metrics.update(_delay_against_truth(module, datamodule, scenario))
        results[name] = {key: float(value) for key, value in metrics.items()}
    return results


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
    module: SoilSealingModule,
    datamodule: SpatioTemporalGraphDataModule,
    scenario: Any,
) -> dict[str, float]:
    from gnn_sar.metrics.delay import detection_delay

    batch = datamodule.batch("test")
    mask = batch["node_mask"].numpy()
    maps = module.predict_maps(batch)
    delay = detection_delay(
        maps["onset"][mask], scenario.onset[mask], n_time=scenario.graph.n_time
    )
    return {f"truth/delay_{key}": value for key, value in delay.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "demo":  # pragma: no cover - argparse enforces the choice
        raise SystemExit(f"unknown command {args.command!r}")

    torch.set_float32_matmul_precision("high")
    results = run_demo(args)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

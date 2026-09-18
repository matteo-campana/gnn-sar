"""Command line entry point.

Six subcommands, in the order a real study goes through them:

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

``docs/guida_dati_sentinel.md`` walks through the first four end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
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

__all__ = [
    "main",
    "run_demo",
    "run_search",
    "run_preprocess",
    "run_ingest_command",
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

    _add_sites_parser(subparsers)
    _add_search_parser(subparsers)
    _add_preprocess_parser(subparsers)
    _add_ingest_parser(subparsers)
    _add_train_parser(subparsers)
    return parser


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
        graph_config=config.graph,
        seed=args.seed,
    )
    graph = scenario.with_exact_labels() if args.exact_labels else scenario.graph

    results: dict[str, dict[str, float]] = {}
    for name in args.models:
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
    """Train one model on an ingested study area and report its metrics."""
    from gnn_sar.data.study_area import load_study_area

    config = (
        ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    )
    max_epochs = args.max_epochs or config.training.max_epochs
    pl.seed_everything(args.seed, workers=True)

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

    if args.model == _RANDOM_FOREST:
        metrics = _run_random_forest(area.graph, datamodule)
    else:
        model_config = ModelConfig(
            **{**config.model.__dict__, "in_channels": datamodule.n_features}
        )
        module = SoilSealingModule(
            model=_MODELS[args.model](model_config), training_config=config.training
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
        metrics = {
            key: float(value)
            for key, value in trainer.test(
                module, datamodule=datamodule, verbose=False
            )[0].items()
        }
        if output_dir:
            trainer.save_checkpoint(output_dir / f"{args.model}.ckpt")
            if not args.no_maps:
                _export_maps(module, datamodule, area, output_dir)

    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    if output_dir:
        (output_dir / "metrics.json").write_text(rendered, encoding="utf-8")
    print(rendered)
    return metrics


def _export_maps(
    module: SoilSealingModule,
    datamodule: SpatioTemporalGraphDataModule,
    area: Any,
    output_dir: Path,
) -> None:
    """Write the sub-annual products as GeoTIFFs on the study grid.

    Predictions are exported for **every** node, test split included: the split
    is there to make the reported metrics honest, not to blank out two thirds of
    the map an operator would use.
    """
    if area.grid is None:
        print("no georeferencing in the study area, skipping the GeoTIFF export")
        return
    from gnn_sar.data.raster import write_raster

    maps = module.predict_maps(datamodule.batch("test"))
    geometries = area.geometries
    written = []
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_float32_matmul_precision("high")

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

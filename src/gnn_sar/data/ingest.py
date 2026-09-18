"""Declarative ingestion: a JSON file describing an area, and the graph for it.

:mod:`gnn_sar.data.study_area` is the library API — arrays in, study area out.
This module is the *operational* one: it takes a configuration that names
directories of preprocessed GeoTIFFs, reads them composite by composite, and
writes a cached study area to disk. It is what ``gnn-sar ingest`` runs, and what
``docs/guida_dati_sentinel.md`` walks through.

Two decisions shape it.

**It reads one composite at a time.** A study area of 20 × 20 km at 10 m is
2000 × 2000 pixels; five years of Sentinel-1 is a few hundred acquisitions. The
whole pixel stack would be tens of gigabytes, while a single month of it is a
few hundred megabytes, and once a composite is pooled onto the nodes it shrinks
by three orders of magnitude. So each composite is read, reduced, pooled and
dropped before the next one is opened; only the ``(T, N, D)`` result survives.

**It refuses what it cannot check.** Every raster must already be on one grid —
same CRS, same pixel size, same extent — because silently resampling inside the
ingestion is how a time series acquires a change signal that is not on the
ground. Preprocessing stays where it belongs: SNAP, ``gdalwarp``, or the
already-terrain-corrected CDSE products.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np

from gnn_sar.config import GraphConfig
from gnn_sar.data.labels import (
    annual_land_take_labels,
    corridor_from_raster,
    node_state_from_rasters,
)
from gnn_sar.data.raster import RasterGrid, list_rasters, read_raster
from gnn_sar.data.segmentation import NodeGeometries, aggregate_stack, slic_geometries
from gnn_sar.data.sentinel import (
    DEFAULT_INVALID_SCL,
    PeriodGrid,
    scl_valid_mask,
    sensing_date,
    sentinel1_channels,
    sentinel2_indices,
)
from gnn_sar.data.study_area import StudyArea, study_area_from_features

__all__ = [
    "SourceConfig",
    "SegmentationConfig",
    "LabelsConfig",
    "CorridorConfig",
    "IngestConfig",
    "run_ingest",
]

_DEFAULT_S1_BANDS = {"VV": 1, "VH": 2}
_DEFAULT_S2_BANDS = {"B02": 1, "B03": 2, "B04": 3, "B08": 4, "B11": 5, "B12": 6}


@dataclass(frozen=True)
class SourceConfig:
    """One sensor: where its rasters are and how they become channels.

    Attributes:
        name: a label, used in messages and channel names.
        kind: ``sentinel1``, ``sentinel2`` or ``raw`` (bands passed through).
        directory: folder of preprocessed, co-registered GeoTIFFs — one file
            per acquisition, each carrying its sensing date in the file name.
        pattern: glob selecting the acquisitions inside it.
        bands: band name to 1-based index in the file. Defaults to
            ``{"VV": 1, "VH": 2}`` for Sentinel-1 and to the six reflectance
            bands for Sentinel-2. Add ``"coherence"`` to a Sentinel-1 source if
            an InSAR coherence layer was stacked into the same file.
        scale: multiply the raw values by this before anything else — ``0.0001``
            turns Sentinel-2 L2A integers into reflectance.
        linear: Sentinel-1 only; whether the file holds linear power (SNAP
            calibration output, CDSE RTC) rather than dB.
        ratio: Sentinel-1 only; append the VV/VH ratio channel.
        indices: Sentinel-2 only; which spectral indices to compute.
        scl_band: Sentinel-2 only; 1-based index of the scene-classification
            layer, if it was stacked into the same file. Without it nothing is
            masked and cloud becomes "change".
        invalid_scl: scene-classification classes to reject.
        reducer: how the acquisitions inside one composite are combined.
        masked: whether a ``NaN`` means "not observed here" (so pooling must
            omit it) rather than "no data at all". Defaults to true for
            Sentinel-2 and false otherwise.
    """

    name: str
    kind: str = "sentinel1"
    directory: str = ""
    pattern: str = "*.tif"
    bands: dict[str, int] = field(default_factory=dict)
    scale: float = 1.0
    linear: bool = True
    ratio: bool = True
    indices: tuple[str, ...] = ("ndvi", "ndbi")
    scl_band: int | None = None
    invalid_scl: tuple[int, ...] = DEFAULT_INVALID_SCL
    reducer: str = "median"
    masked: bool | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("sentinel1", "sentinel2", "raw"):
            raise ValueError(
                f"kind must be 'sentinel1', 'sentinel2' or 'raw', got {self.kind!r}"
            )
        if self.reducer not in ("median", "mean", "min", "max"):
            raise ValueError(f"unknown reducer {self.reducer!r}")

    @property
    def band_map(self) -> dict[str, int]:
        if self.bands:
            return dict(self.bands)
        if self.kind == "sentinel1":
            return dict(_DEFAULT_S1_BANDS)
        if self.kind == "sentinel2":
            return dict(_DEFAULT_S2_BANDS)
        raise ValueError(f"source {self.name!r} of kind 'raw' must list its bands")

    @property
    def is_masked(self) -> bool:
        return self.kind == "sentinel2" if self.masked is None else bool(self.masked)


@dataclass(frozen=True)
class SegmentationConfig:
    """How the raster domain is cut into nodes — once, for the whole series.

    Attributes:
        method: ``grid`` (a regular tiling, deterministic and dependency-free),
            ``slic`` (super-pixels on a multitemporal composite, needs the
            ``segmentation`` extra) or ``labels`` (a rasterised cadastral or
            parcel layer, the closest to what the proposal describes).
        block: ``grid`` only — side of a tile in pixels. At 10 m, ``8`` gives
            80 m nodes, a reasonable unit for peri-urban land take.
        n_segments, compactness: ``slic`` only.
        raster: ``labels`` only — the parcel raster, on the study grid.
        band: which band of that raster holds the parcel ids.
        composites: ``slic`` only — how many composites to average into the
            image SLIC segments. Spreading them over the series is what makes
            the partition represent the whole period rather than one date.
    """

    method: str = "grid"
    block: int = 8
    n_segments: int = 2000
    compactness: float = 10.0
    raster: str | None = None
    band: int = 1
    composites: int = 8

    def __post_init__(self) -> None:
        if self.method not in ("grid", "slic", "labels"):
            raise ValueError(
                f"method must be 'grid', 'slic' or 'labels', got {self.method!r}"
            )
        if self.method == "labels" and not self.raster:
            raise ValueError("segmentation method 'labels' needs a raster")


@dataclass(frozen=True)
class LabelsConfig:
    """The annual cartography that supervises the model.

    Attributes:
        rasters: calendar year (as a string, JSON has no integer keys) to a
            raster of the **sealed state** that year — the ISPRA/SNPA soil
            consumption layer clipped to the study area and rasterised on the
            study grid. Land-take *transitions* are derived from consecutive
            states by :func:`gnn_sar.data.labels.annual_land_take_labels`.
        band: which band of those rasters to read.
        sealed_values: pixel values that mean "sealed". ``None`` treats any
            non-zero, non-nodata value as sealed, which fits a 0/1 mask.
        threshold: share of a node that must be sealed for the node to count as
            sealed.
        min_coverage: share of a node that must carry valid data for its state
            to be decided at all.
        pre_existing: what to do with land already sealed in the first year, see
            :func:`~gnn_sar.data.labels.annual_land_take_labels`.
    """

    rasters: dict[str, str] = field(default_factory=dict)
    band: int = 1
    sealed_values: tuple[int, ...] | None = None
    threshold: float = 0.5
    min_coverage: float = 0.5
    pre_existing: str = "ignore"

    def __post_init__(self) -> None:
        if not self.rasters:
            raise ValueError("labels need at least one yearly raster")
        for year in self.rasters:
            if not str(year).isdigit():
                raise ValueError(f"{year!r} is not a calendar year")


@dataclass(frozen=True)
class CorridorConfig:
    """The infrastructure mask that restricts connectivity edges."""

    raster: str
    band: int = 1
    min_fraction: float = 0.0


@dataclass(frozen=True)
class IngestConfig:
    """Everything ``gnn-sar ingest`` needs, and nothing it can guess.

    Attributes:
        sources: one entry per sensor; their channels are concatenated.
        segmentation: how nodes are defined.
        period: composite length — ``"month"``, ``"dekad"`` or a number of days.
        start: force the first composite (``YYYY-MM-DD``); defaults to the
            composite holding the earliest acquisition. Pin it when several
            areas must share a time axis.
        n_periods: force the number of composites.
        statistics: per-node statistics of every channel.
        labels: the annual cartography, or ``None`` for an unsupervised area.
        corridor: the infrastructure mask, or ``None``.
        graph: edge-construction settings.
        output: where the ``.npz`` cache is written.
        root: directory every relative path is resolved against. Defaults to the
            folder holding the config file.

    Any key starting with ``//`` is ignored, at any level, so a config can carry
    comments.
    """

    sources: tuple[SourceConfig, ...] = ()
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    period: str | int = "month"
    start: str | None = None
    n_periods: int | None = None
    statistics: tuple[str, ...] = ("mean", "std")
    labels: LabelsConfig | None = None
    corridor: CorridorConfig | None = None
    graph: GraphConfig = field(default_factory=GraphConfig)
    output: str = "study_area.npz"
    root: str = "."

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one source is required")
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique, got {names}")

    def resolve(self, path: str | Path) -> Path:
        """A configured path, relative to :attr:`root`."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else Path(self.root) / candidate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], root: str | Path = "."
    ) -> "IngestConfig":
        data = {"root": str(root), **dict(payload)}
        return _build(cls, data)

    @classmethod
    def from_json(cls, path: str | Path) -> "IngestConfig":
        """Load a config, resolving relative paths against the file's folder."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload, root=payload.get("root", path.parent))


def run_ingest(config: IngestConfig, verbose: bool = True) -> StudyArea:
    """Build a study area from the rasters the config points at.

    Returns:
        The assembled :class:`~gnn_sar.data.study_area.StudyArea`. Writing it to
        ``config.output`` is the caller's job — ``gnn-sar ingest`` does it.
    """
    log = print if verbose else _silent

    catalogue = {source.name: _catalogue(config, source, log) for source in config.sources}
    reference = _reference_grid(config, catalogue)
    log(f"grid             {reference.describe()}")

    every_date = [date for entries in catalogue.values() for _, date in entries]
    periods = PeriodGrid.covering(
        every_date,
        period=config.period,  # type: ignore[arg-type]
        start=config.start,
        n_periods=config.n_periods,
    )
    log(
        f"composites       {periods.n_periods} "
        f"({periods.labels()[0]} … {periods.labels()[-1]}, {periods.period})"
    )

    geometries = _build_geometries(config, catalogue, reference, log)
    log(f"nodes            {geometries.n_nodes}")

    blocks, names = [], []
    for source in config.sources:
        values, channel_names = _pool_source(
            config, source, catalogue[source.name], periods, geometries, reference, log
        )
        blocks.append(values)
        names.extend(
            f"{channel}_{statistic}"
            for channel in channel_names
            for statistic in config.statistics
        )
    features = np.concatenate(blocks, axis=2)

    annual_labels = _read_labels(config, geometries, reference, log)
    corridor = _read_corridor(config, geometries, reference, log)

    area = study_area_from_features(
        features=features,
        channel_names=names,
        geometries=geometries,
        periods=periods,
        graph_config=config.graph,
        corridor=corridor,
        annual_labels=annual_labels,
        grid=reference,
        metadata={
            "config": config.to_dict(),
            "acquisitions": {
                name: [path.name for path, _ in entries]
                for name, entries in catalogue.items()
            },
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return area


def _catalogue(
    config: IngestConfig, source: SourceConfig, log: Any
) -> list[tuple[Path, datetime]]:
    """Every acquisition of a source, with its sensing date, oldest first."""
    directory = config.resolve(source.directory)
    paths = list_rasters(directory, source.pattern)
    if not paths:
        raise FileNotFoundError(
            f"source {source.name!r}: no file matches {source.pattern!r} in {directory}"
        )
    entries = sorted(((path, sensing_date(path)) for path in paths), key=lambda e: e[1])
    log(
        f"source {source.name:<10} {len(entries)} acquisitions, "
        f"{entries[0][1].date()} … {entries[-1][1].date()}"
    )
    return entries


def _reference_grid(
    config: IngestConfig, catalogue: Mapping[str, list[tuple[Path, datetime]]]
) -> RasterGrid:
    first = catalogue[config.sources[0].name][0][0]
    _, grid = read_raster(first, bands=[1])
    return grid


def _build_geometries(
    config: IngestConfig,
    catalogue: Mapping[str, list[tuple[Path, datetime]]],
    reference: RasterGrid,
    log: Any,
) -> NodeGeometries:
    segmentation = config.segmentation
    if segmentation.method == "grid":
        return NodeGeometries.regular_grid(
            reference.height, reference.width, block=segmentation.block
        )
    if segmentation.method == "labels":
        assert segmentation.raster is not None  # checked in __post_init__
        values, grid = read_raster(
            config.resolve(segmentation.raster), bands=[segmentation.band]
        )
        if not grid.matches(reference):
            raise ValueError(
                f"the parcel raster ({grid.describe()}) is not on the study grid "
                f"({reference.describe()})"
            )
        log(f"nodes from       {segmentation.raster}")
        # Nodata becomes its own parcel id, hence its own node: clip the parcel
        # layer to the area of interest rather than leaving a hole in it.
        parcels = np.nan_to_num(values[0], nan=-1).astype(np.int64)
        return NodeGeometries.from_label_image(parcels)

    # "slic": segment a multitemporal composite, so the partition represents the
    # whole series rather than whichever date happened to be first.
    source = config.sources[0]
    entries = catalogue[source.name]
    wanted = min(segmentation.composites, len(entries))
    spread = np.linspace(0, len(entries) - 1, wanted).round().astype(int)
    chosen = [entries[i] for i in spread]
    log(f"nodes from       SLIC over {len(chosen)} composites of {source.name}")
    stack = np.stack(
        [_read_channels(config, source, path, reference)[0] for path, _ in chosen]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(stack, axis=0)
    composite = np.nan_to_num(composite, nan=float(np.nanmin(composite)))
    return slic_geometries(
        composite,
        n_segments=segmentation.n_segments,
        compactness=segmentation.compactness,
    )


def _read_channels(
    config: IngestConfig,
    source: SourceConfig,
    path: Path,
    reference: RasterGrid | None,
) -> tuple[np.ndarray, list[str]]:
    """One acquisition as ``(C, H, W)`` model channels, plus their names."""
    band_map = source.band_map
    wanted = list(band_map)
    indexes = [band_map[name] for name in wanted]
    if source.scl_band is not None:
        indexes.append(source.scl_band)

    values, grid = read_raster(path, bands=indexes)
    if reference is not None and not grid.matches(reference):
        raise ValueError(
            f"{path.name} ({grid.describe()}) is not on the study grid "
            f"({reference.describe()}); align it with gdalwarp before ingesting"
        )

    scl = values[-1] if source.scl_band is not None else None
    bands = {
        name: values[position] * source.scale for position, name in enumerate(wanted)
    }

    if source.kind == "sentinel1":
        return sentinel1_channels(bands, linear=source.linear, ratio=source.ratio)
    if source.kind == "sentinel2":
        valid = None if scl is None else scl_valid_mask(scl, source.invalid_scl)
        return sentinel2_indices(bands, source.indices, valid=valid)
    return np.stack([bands[name] for name in wanted]), [
        f"{source.name}_{name}" for name in wanted
    ]


def _pool_source(
    config: IngestConfig,
    source: SourceConfig,
    entries: Sequence[tuple[Path, datetime]],
    periods: PeriodGrid,
    geometries: NodeGeometries,
    reference: RasterGrid,
    log: Any,
) -> tuple[np.ndarray, list[str]]:
    """Composite and pool one source, reading a single composite at a time."""
    index = periods.index_of([date for _, date in entries])
    reducer = {
        "median": (np.median, np.nanmedian),
        "mean": (np.mean, np.nanmean),
        "min": (np.min, np.nanmin),
        "max": (np.max, np.nanmax),
    }[source.reducer][1 if source.is_masked else 0]
    nan_policy = "omit" if source.is_masked else "propagate"

    # ``None`` marks a composite no acquisition fell into; it becomes NaN and is
    # carried forward once the whole feature tensor exists.
    pooled: list[np.ndarray | None] = []
    names: list[str] = []
    empty = 0
    for period in range(periods.n_periods):
        members = [entries[i][0] for i in np.flatnonzero(index == period)]
        if not members:
            empty += 1
            pooled.append(None)
            continue
        stack = []
        for path in members:
            channels, names = _read_channels(config, source, path, reference)
            stack.append(channels)
        with warnings.catch_warnings():
            # A pixel masked in every acquisition of the composite is expected;
            # it is filled from the neighbouring composites further down.
            warnings.simplefilter("ignore", RuntimeWarning)
            composite = reducer(np.stack(stack), axis=0)
        pooled.append(
            aggregate_stack(
                composite[None, ...],
                geometries,
                statistics=config.statistics,
                nan_policy=nan_policy,
            )[0]
        )

    width = next(block.shape[1] for block in pooled if block is not None)
    filled = [
        np.full((geometries.n_nodes, width), np.nan) if block is None else block
        for block in pooled
    ]
    if empty:
        log(
            f"source {source.name:<10} {empty} composites without an acquisition, "
            "carried forward from the previous one"
        )
    return np.stack(filled), names


def _read_labels(
    config: IngestConfig, geometries: NodeGeometries, reference: RasterGrid, log: Any
) -> dict[int, np.ndarray] | None:
    if config.labels is None:
        return None
    settings = config.labels
    years = sorted(int(year) for year in settings.rasters)

    masks, valids = [], []
    for year in years:
        values, grid = read_raster(
            config.resolve(settings.rasters[str(year)]), bands=[settings.band]
        )
        if not grid.matches(reference):
            raise ValueError(
                f"the {year} cartography ({grid.describe()}) is not on the study "
                f"grid ({reference.describe()})"
            )
        layer = values[0]
        valid = ~np.isnan(layer)
        sealed = (
            np.isin(np.nan_to_num(layer, nan=-1), list(settings.sealed_values))
            if settings.sealed_values
            else np.nan_to_num(layer, nan=0.0) > 0
        )
        masks.append(sealed)
        valids.append(valid)

    state = node_state_from_rasters(
        masks,
        geometries,
        threshold=settings.threshold,
        min_coverage=settings.min_coverage,
        valid_masks=valids,
    )
    annual = annual_land_take_labels(state, pre_existing=settings.pre_existing)
    log(
        f"labels           {len(years)} years {years}, "
        f"{int((annual == 1).sum())} land-take nodes, "
        f"{int((annual == -1).sum())} unknown year-node pairs"
    )
    return {year: annual[position] for position, year in enumerate(years)}


def _read_corridor(
    config: IngestConfig, geometries: NodeGeometries, reference: RasterGrid, log: Any
) -> np.ndarray | None:
    if config.corridor is None:
        return None
    values, grid = read_raster(
        config.resolve(config.corridor.raster), bands=[config.corridor.band]
    )
    if not grid.matches(reference):
        raise ValueError(
            f"the corridor raster ({grid.describe()}) is not on the study grid "
            f"({reference.describe()})"
        )
    corridor = corridor_from_raster(
        np.nan_to_num(values[0], nan=0.0), geometries, config.corridor.min_fraction
    )
    log(f"corridor         {int(corridor.sum())} of {geometries.n_nodes} nodes")
    return corridor


def _silent(*_args: object, **_kwargs: object) -> None:
    return None


def _build(cls: type, payload: Mapping[str, Any]) -> Any:
    """Instantiate nested dataclasses, lists of dataclasses included.

    Keys starting with ``//`` are dropped: JSON has no comments, and an
    ingestion config is exactly the kind of file that needs one.
    """
    payload = {
        key: value for key, value in payload.items() if not str(key).startswith("//")
    }
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"unknown keys for {cls.__name__}: {sorted(unknown)} "
            f"(expected any of {sorted(known)})"
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        annotation = hints[name]
        nested = _dataclass_of(annotation)
        if isinstance(value, Mapping) and nested is not None:
            kwargs[name] = _build(nested, value)
        elif isinstance(value, (list, tuple)) and nested is not None:
            kwargs[name] = tuple(_build(nested, item) for item in value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _dataclass_of(annotation: Any) -> type | None:
    """The dataclass inside ``X``, ``X | None`` or ``tuple[X, ...]``, if any."""
    if is_dataclass(annotation) and isinstance(annotation, type):
        return annotation
    for argument in getattr(annotation, "__args__", ()):  # Optional / tuple / list
        found = _dataclass_of(argument)
        if found is not None:
            return found
    return None

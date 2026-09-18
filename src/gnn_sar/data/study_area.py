"""Assembling a real study area: rasters in, :class:`SpatioTemporalGraph` out.

This is the module the synthetic generator stands in for. Given per-sensor
stacks of aligned acquisitions and the dates they were sensed, it performs in
order the four steps of the proposal's Figure 2 that sit between the archive
and the network:

1. **regularise time** — irregular revisits become the fixed composite axis
   ``T`` of a :class:`~gnn_sar.data.sentinel.PeriodGrid`, one reduction per
   source (median for SAR, cloud-aware median for optical);
2. **pool onto nodes** — the composites are aggregated over the *fixed* node
   partition, giving ``(T, N, D)``;
3. **wire the graph** — adjacency plus corridor-restricted connectivity;
4. **attach the supervision** — annual labels aligned, calendar year by
   calendar year, with the composites they supervise.

The result is exactly what :func:`gnn_sar.data.synthetic.make_synthetic_scenario`
produces, so everything downstream — datamodule, models, losses, metrics — is
unchanged whether the features came from a simulator or from Sentinel-1.

:class:`StudyArea` also carries what the graph itself does not need but an
operational product does: the georeferencing, the channel names and which
calendar span each composite covers. :meth:`StudyArea.save` writes all of it to
one ``.npz``, so the expensive ingestion runs once and every later experiment
starts from the cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np

from gnn_sar.config import GraphConfig
from gnn_sar.data.features import composite_by_period, fill_temporal_gaps
from gnn_sar.data.graph import SpatioTemporalGraph, build_spatial_edges
from gnn_sar.data.raster import RasterGrid
from gnn_sar.data.segmentation import NodeGeometries, aggregate_stack
from gnn_sar.data.sentinel import PeriodGrid
from gnn_sar.data.targets import IGNORE_INDEX

__all__ = [
    "SourceStack",
    "StudyArea",
    "align_annual_labels",
    "composite_sources",
    "build_study_area",
    "study_area_from_features",
    "load_study_area",
]


@dataclass(frozen=True)
class SourceStack:
    """One sensor's acquisitions, already on the study grid.

    Attributes:
        values: ``(K, C, H, W)`` — ``K`` acquisitions of ``C`` channels. The
            channels are the model's, not the product's: dB backscatter and a
            ratio for Sentinel-1, spectral indices for Sentinel-2. Build them
            with :func:`gnn_sar.data.sentinel.sentinel1_channels` and
            :func:`~gnn_sar.data.sentinel.sentinel2_indices`.
        dates: ``K`` sensing dates, one per acquisition.
        names: ``C`` channel names, kept so a trained model can say which
            channel it read.
        reducer: how acquisitions inside one composite are combined.
        ignore_nan: reduce over the valid samples only. Set it for any source
            carrying a mask — Sentinel-2 under cloud — and leave it off for
            Sentinel-1, where a ``NaN`` means a real hole (a shadowed slope,
            the edge of a swath) that should propagate rather than be averaged
            away.
    """

    values: np.ndarray
    dates: Sequence[datetime | date]
    names: Sequence[str]
    reducer: str = "median"
    ignore_nan: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 4:
            raise ValueError(f"values must be (K, C, H, W), got {values.shape}")
        if len(self.dates) != values.shape[0]:
            raise ValueError(
                f"got {values.shape[0]} acquisitions but {len(self.dates)} dates"
            )
        if len(self.names) != values.shape[1]:
            raise ValueError(
                f"got {values.shape[1]} channels but {len(self.names)} names"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[2]), int(self.values.shape[3]))


@dataclass(frozen=True)
class StudyArea:
    """A real area of interest, ready to train on and to map back.

    Attributes:
        graph: what the model consumes.
        geometries: the fixed node partition, also the way back to a raster.
        periods: the composite axis, i.e. what "composite 7" means as a date.
        channel_names: ``D`` names, in feature order.
        year_labels: the calendar year behind each row of ``annual_labels``.
        grid: the georeferencing, when the area came from GeoTIFFs.
        corridor: the infrastructure mask used for connectivity edges.
        metadata: free-form provenance — product ids, preprocessing chain.
    """

    graph: SpatioTemporalGraph
    geometries: NodeGeometries
    periods: PeriodGrid
    channel_names: list[str]
    year_labels: list[int] = field(default_factory=list)
    grid: RasterGrid | None = None
    corridor: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        """A one-paragraph summary, printed by ``gnn-sar ingest``."""
        graph = self.graph
        labels = graph.annual_labels
        positives = 0 if labels is None else int((labels == 1).sum())
        unknown = 0 if labels is None else int((labels == IGNORE_INDEX).sum())
        composites = self.periods.labels()
        return "\n".join(
            [
                f"nodes            {graph.n_nodes}",
                f"composites       {graph.n_time} "
                f"({composites[0]} … {composites[-1]}, {self.periods.period})",
                f"channels         {graph.n_features} ({', '.join(self.channel_names)})",
                f"spatial edges    {graph.spatial_edge_index.shape[1]}",
                f"years            {self.year_labels}",
                f"land-take nodes  {positives} positive, {unknown} unknown "
                f"(of {0 if labels is None else labels.size} year-node pairs)",
            ]
        )

    def save(self, path: str | Path) -> Path:
        """Write the whole study area to a single compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph = self.graph
        payload: dict[str, np.ndarray] = {
            "features": graph.features,
            "spatial_edge_index": graph.spatial_edge_index,
            "spatial_edge_type": graph.spatial_edge_type,
            "labels_image": self.geometries.labels,
            "centroids": self.geometries.centroids,
            "areas": self.geometries.areas,
            "channel_names": np.asarray(self.channel_names, dtype=object),
            "year_labels": np.asarray(self.year_labels, dtype=np.int64),
            "meta": np.asarray(json.dumps(self._meta()), dtype=object),
        }
        for name, value in (
            ("annual_labels", graph.annual_labels),
            ("year_index", graph.year_index),
            ("node_labels", graph.node_labels),
            ("corridor", self.corridor),
        ):
            if value is not None:
                payload[name] = value
        np.savez_compressed(path, **payload)
        return path

    def _meta(self) -> dict[str, object]:
        return {
            "period": self.periods.period,
            "period_start": self.periods.start.isoformat(),
            "n_periods": self.periods.n_periods,
            "grid": None if self.grid is None else self.grid.__dict__,
            "metadata": self.metadata,
        }


def load_study_area(path: str | Path) -> StudyArea:
    """Read back what :meth:`StudyArea.save` wrote."""
    with np.load(Path(path), allow_pickle=True) as payload:
        stored = {key: payload[key] for key in payload.files}
    meta = json.loads(str(stored["meta"].item()))
    grid = (
        None
        if meta["grid"] is None
        else RasterGrid(**{**meta["grid"], "transform": tuple(meta["grid"]["transform"])})
    )

    geometries = NodeGeometries(
        labels=stored["labels_image"],
        centroids=stored["centroids"],
        areas=stored["areas"],
    )
    graph = SpatioTemporalGraph(
        features=stored["features"],
        spatial_edge_index=stored["spatial_edge_index"],
        spatial_edge_type=stored["spatial_edge_type"],
        node_labels=stored.get("node_labels"),
        annual_labels=stored.get("annual_labels"),
        year_index=stored.get("year_index"),
        geometries=geometries,
    )
    return StudyArea(
        graph=graph,
        geometries=geometries,
        periods=PeriodGrid(
            period=meta["period"],
            start=date.fromisoformat(meta["period_start"]),
            n_periods=int(meta["n_periods"]),
        ),
        channel_names=[str(name) for name in stored["channel_names"]],
        year_labels=[int(year) for year in stored["year_labels"]],
        grid=grid,
        corridor=stored.get("corridor"),
        metadata=meta["metadata"],
    )


def align_annual_labels(
    labels_by_year: Mapping[int, np.ndarray],
    periods: PeriodGrid,
    n_nodes: int | None = None,
) -> np.ndarray:
    """``(Y, N)`` labels laid out on the calendar years the composites span.

    The rows of ``annual_labels`` are indexed by
    :meth:`~gnn_sar.data.sentinel.PeriodGrid.year_index`, so they must cover
    *every* calendar year the series touches, in order and without holes. A year
    with no cartography is filled with ``-1``: unknown, not negative. Passing
    labels for a year outside the series is an error rather than a silent drop,
    because it almost always means the series and the cartography were lined up
    wrongly by one year.
    """
    wanted = periods.year_labels()
    unexpected = sorted(set(labels_by_year) - set(wanted))
    if unexpected:
        raise ValueError(
            f"labels given for years {unexpected}, which the composites "
            f"{periods.labels()[0]} … {periods.labels()[-1]} do not cover "
            f"(they span {wanted})"
        )
    if not labels_by_year and n_nodes is None:
        raise ValueError("n_nodes is required when no labels are given")

    sizes = {int(np.asarray(v).shape[0]) for v in labels_by_year.values()}
    if len(sizes) > 1:
        raise ValueError(f"the yearly labels disagree on the node count: {sorted(sizes)}")
    size = n_nodes if n_nodes is not None else sizes.pop()

    rows = []
    for year in wanted:
        if year in labels_by_year:
            row = np.asarray(labels_by_year[year], dtype=np.int64)
            if row.shape != (size,):
                raise ValueError(
                    f"labels for {year} must be ({size},), got {row.shape}"
                )
            rows.append(row)
        else:
            rows.append(np.full(size, IGNORE_INDEX, dtype=np.int64))
    return np.stack(rows)


def composite_sources(
    sources: Sequence[SourceStack],
    periods: PeriodGrid,
    fill: str = "previous",
) -> tuple[np.ndarray, list[str]]:
    """Reduce every source onto the shared composite axis and concatenate.

    Sources may be acquired on completely different dates — Sentinel-1 every
    few days regardless of weather, Sentinel-2 only when it is clear — which is
    the whole reason the composite axis exists: after this step they are
    channels of one ``(T, C, H, W)`` array and nothing downstream needs to know
    they came from two satellites.

    Returns:
        ``(composites, names)``.
    """
    if not sources:
        raise ValueError("at least one source is required")
    shapes = {source.shape for source in sources}
    if len(shapes) > 1:
        raise ValueError(f"the sources sit on different rasters: {sorted(shapes)}")

    stacks, names = [], []
    for source in sources:
        index = periods.index_of(list(source.dates))
        composites = composite_by_period(
            source.values,
            index,
            n_periods=periods.n_periods,
            reducer=source.reducer,  # type: ignore[arg-type]
            fill=fill,  # type: ignore[arg-type]
            ignore_nan=source.ignore_nan,
        )
        if source.ignore_nan:
            # Cloud that never cleared inside a composite leaves per-pixel
            # holes that the period-level fill above cannot see.
            composites = fill_temporal_gaps(composites)
        stacks.append(composites)
        names.extend(source.names)
    return np.concatenate(stacks, axis=1), list(names)


def build_study_area(
    sources: Sequence[SourceStack],
    geometries: NodeGeometries,
    periods: PeriodGrid | None = None,
    period: str | int = "month",
    statistics: Sequence[str] = ("mean", "std"),
    nan_policy: str = "auto",
    graph_config: GraphConfig | None = None,
    corridor: np.ndarray | None = None,
    annual_labels: np.ndarray | Mapping[int, np.ndarray] | None = None,
    node_labels: np.ndarray | None = None,
    grid: RasterGrid | None = None,
    metadata: Mapping[str, object] | None = None,
) -> StudyArea:
    """Build a study area from aligned rasters, dates, geometries and labels.

    Args:
        sources: one :class:`SourceStack` per sensor.
        geometries: the node partition, computed **once** for the whole series
            (see :mod:`gnn_sar.data.segmentation`).
        periods: the composite axis. ``None`` derives the shortest one covering
            every acquisition of every source.
        period: composite length used when ``periods`` is ``None``.
        statistics: per-node statistics of each channel — ``mean`` is the level,
            ``std`` the within-node dispersion that stands in for texture.
        nan_policy: how pooling treats masked pixels, see
            :func:`~gnn_sar.data.segmentation.aggregate_stack`. ``"auto"`` uses
            ``"omit"`` as soon as one source is masked and ``"propagate"``
            otherwise.
        graph_config: edge-construction settings.
        corridor: ``(N,)`` boolean infrastructure mask, see
            :func:`gnn_sar.data.labels.corridor_from_raster`.
        annual_labels: either a ready ``(Y, N)`` array indexed by
            ``periods.year_index()``, or a mapping ``calendar year -> (N,)``
            which :func:`align_annual_labels` lays out. ``None`` builds an
            unsupervised graph — useful to run inference over a new area with a
            model trained elsewhere, and rejected by the datamodule for training.
        node_labels: optional ``(T, N)`` exact sub-annual truth, where a
            validation campaign established the dates.
        grid: the georeferencing of the rasters, kept for the output maps.
        metadata: free-form provenance stored with the area.

    Returns:
        A :class:`StudyArea`.
    """
    if periods is None:
        every_date = [d for source in sources for d in source.dates]
        periods = PeriodGrid.covering(every_date, period=period)  # type: ignore[arg-type]

    composites, channel_names = composite_sources(sources, periods)
    if composites.shape[-2:] != geometries.shape:
        raise ValueError(
            f"the rasters {composites.shape[-2:]} do not match the geometries "
            f"{geometries.shape}"
        )
    if nan_policy == "auto":
        nan_policy = "omit" if any(s.ignore_nan for s in sources) else "propagate"

    features = aggregate_stack(
        composites, geometries, statistics=statistics, nan_policy=nan_policy
    )
    names = [
        f"{channel}_{statistic}" for channel in channel_names for statistic in statistics
    ]
    return study_area_from_features(
        features=features,
        channel_names=names,
        geometries=geometries,
        periods=periods,
        graph_config=graph_config,
        corridor=corridor,
        annual_labels=annual_labels,
        node_labels=node_labels,
        grid=grid,
        metadata=metadata,
    )


def study_area_from_features(
    features: np.ndarray,
    channel_names: Sequence[str],
    geometries: NodeGeometries,
    periods: PeriodGrid,
    graph_config: GraphConfig | None = None,
    corridor: np.ndarray | None = None,
    annual_labels: np.ndarray | Mapping[int, np.ndarray] | None = None,
    node_labels: np.ndarray | None = None,
    grid: RasterGrid | None = None,
    metadata: Mapping[str, object] | None = None,
) -> StudyArea:
    """Wire an already-pooled ``(T, N, D)`` feature tensor into a study area.

    The half of :func:`build_study_area` that does not touch pixels: edges,
    label alignment and packaging. :mod:`gnn_sar.data.ingest` calls it directly
    because it pools each composite as it reads it, instead of holding the whole
    pixel stack in memory.

    Residual ``NaN`` are closed along time (a node under cloud all month takes
    its last clear value) and anything still missing afterwards is an error: a
    node never observed cannot be trained on, and letting it through would turn
    into a silent ``NaN`` loss several steps later.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError(f"features must be (T, N, D), got {features.shape}")
    if features.shape[0] != periods.n_periods:
        raise ValueError(
            f"features carry {features.shape[0]} composites but the period grid "
            f"describes {periods.n_periods}"
        )
    if features.shape[1] != geometries.n_nodes:
        raise ValueError(
            f"features carry {features.shape[1]} nodes but the geometries hold "
            f"{geometries.n_nodes}"
        )
    if len(channel_names) != features.shape[2]:
        raise ValueError(
            f"got {features.shape[2]} channels but {len(channel_names)} names"
        )

    features = fill_temporal_gaps(features)
    missing = np.isnan(features).any(axis=(0, 2))
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} of {geometries.n_nodes} nodes are never observed "
            "in at least one channel (no valid pixel in any composite). Drop the "
            "sources or the composites that are empty over them, widen the area, "
            "or relax the masking before building the graph."
        )

    edge_index, edge_type = build_spatial_edges(geometries, graph_config, corridor)
    year_index = periods.year_index()

    labels: np.ndarray | None
    if isinstance(annual_labels, Mapping):
        labels = align_annual_labels(annual_labels, periods, geometries.n_nodes)
    elif annual_labels is not None:
        labels = np.asarray(annual_labels, dtype=np.int64)
        expected = int(year_index.max()) + 1
        if labels.shape != (expected, geometries.n_nodes):
            raise ValueError(
                f"annual_labels must be ({expected}, {geometries.n_nodes}) to match "
                f"the {periods.n_periods} composites spanning {periods.year_labels()}, "
                f"got {labels.shape}"
            )
    else:
        labels = None

    graph = SpatioTemporalGraph(
        features=features,
        spatial_edge_index=edge_index,
        spatial_edge_type=edge_type,
        node_labels=node_labels,
        annual_labels=labels,
        year_index=None if labels is None else year_index,
        geometries=geometries,
    )
    return StudyArea(
        graph=graph,
        geometries=geometries,
        periods=periods,
        channel_names=list(channel_names),
        year_labels=periods.year_labels(),
        grid=grid,
        corridor=corridor,
        metadata=dict(metadata or {}),
    )

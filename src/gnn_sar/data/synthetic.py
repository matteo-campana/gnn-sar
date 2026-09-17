"""A synthetic land-take scenario.

Sentinel-1 archives are large and the ISPRA/SNPA cartography is not
redistributable, so the repository ships a generator instead of data. It is not
a simulator of radar physics; it reproduces the three properties the method
depends on, and nothing else:

1. **Topological propagation** — new sealing appears next to already sealed
   nodes and along an infrastructure corridor, not uniformly at random. This is
   the structure a graph can exploit and a per-pixel model cannot.
2. **A signature spread over time** — a coherence drop and a backscatter dip
   during construction, then a durable backscatter rise. Detecting it requires
   looking at the sequence, not at one date.
3. **Weak annual labels** — the exact onset is generated but withheld; what the
   pipeline is given by default is the annual aggregate, exactly like the
   official cartography.

It makes the tests and the demo runnable end to end; it is *not* evidence about
the method's accuracy on real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnn_sar.config import GraphConfig
from gnn_sar.data.features import to_decibel
from gnn_sar.data.graph import SpatioTemporalGraph, build_spatial_edges
from gnn_sar.data.segmentation import NodeGeometries, aggregate_stack

__all__ = ["SyntheticLandTakeScenario", "make_synthetic_scenario"]

# Linear-power means, expressed in dB for readability.
_SIGMA0_DB = {
    "natural_vv": -15.0,
    "natural_vh": -22.0,
    "construction_vv": -12.0,
    "construction_vh": -20.0,
    "sealed_vv": -7.0,
    "sealed_vh": -14.0,
}
_COHERENCE = {"stable": 0.72, "construction": 0.18, "sealed": 0.85}


@dataclass(frozen=True)
class SyntheticLandTakeScenario:
    """Everything a synthetic study area exposes.

    Attributes:
        graph: the assembled spatio-temporal graph, carrying only the annual
            labels — the supervision a real run would have.
        stack: ``(T, 3, H, W)`` pixel raster of ``[VV dB, VH dB, coherence]``.
        geometries: the fixed node partition.
        onset: ``(N,)`` composite index at which each node seals, ``-1`` if it
            never does. Withheld from ``graph``; use it only for evaluation.
        corridor: ``(N,)`` boolean mask of the infrastructure corridor.
        periods_per_year: composites per year (12 for monthly compositing).
    """

    graph: SpatioTemporalGraph
    stack: np.ndarray
    geometries: NodeGeometries
    onset: np.ndarray
    corridor: np.ndarray
    periods_per_year: int

    @property
    def state_labels(self) -> np.ndarray:
        """``(T, N)`` ground-truth sealed state, monotone non-decreasing in time."""
        times = np.arange(self.graph.n_time)[:, None]
        sealed = self.onset[None, :]
        return ((sealed >= 0) & (times >= sealed)).astype(np.int64)

    def with_exact_labels(self) -> SpatioTemporalGraph:
        """The same graph, additionally carrying the exact sub-annual truth.

        Useful to measure the cost of weak supervision: train once on the
        annual labels, once on this, and compare.
        """
        return SpatioTemporalGraph(
            features=self.graph.features,
            spatial_edge_index=self.graph.spatial_edge_index,
            spatial_edge_type=self.graph.spatial_edge_type,
            node_labels=self.state_labels,
            annual_labels=self.graph.annual_labels,
            year_index=self.graph.year_index,
            geometries=self.geometries,
        )


def make_synthetic_scenario(
    height: int = 48,
    width: int = 48,
    block: int = 4,
    n_time: int = 24,
    periods_per_year: int = 12,
    base_hazard: float = 0.002,
    neighbor_hazard: float = 0.12,
    corridor_hazard: float = 0.03,
    n_seeds: int = 3,
    construction_periods: int = 2,
    looks: int = 4,
    contrast: float = 1.0,
    heterogeneity_db: float = 1.2,
    graph_config: GraphConfig | None = None,
    seed: int | None = 0,
) -> SyntheticLandTakeScenario:
    """Generate a study area, its SAR-like stack and its weak annual labels.

    Args:
        height, width: raster size in pixels.
        block: super-pixel side; the node count is ``ceil(h/b) * ceil(w/b)``.
        n_time: number of composites ``T``.
        periods_per_year: composites per year; ``n_time`` need not be a multiple.
        base_hazard: per-period probability that an isolated node seals.
        neighbor_hazard: added hazard per already-sealed graph neighbour. This
            is what makes the process relational.
        corridor_hazard: added hazard for nodes on the infrastructure corridor.
        n_seeds: initial built-up nuclei the expansion grows from.
        construction_periods: length of the low-coherence construction phase.
        looks: equivalent number of looks of the multiplicative speckle; lower
            means noisier.
        contrast: scales how far the construction and sealed signatures sit
            from the natural one. ``1.0`` is a clean, easily separable case;
            values around ``0.3`` put the change signature inside the speckle,
            which is where the relational information starts to matter.
        heterogeneity_db: standard deviation of the per-node, time-constant
            offset standing in for land-cover variety. It cannot be mistaken
            for a change signal, but it does hide one.
        graph_config: edge-construction settings.
        seed: RNG seed, ``None`` for non-reproducible output.

    Returns:
        A :class:`SyntheticLandTakeScenario`.
    """
    if construction_periods < 0:
        raise ValueError("construction_periods must be >= 0")
    if looks < 1:
        raise ValueError("looks must be >= 1")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be >= 1")
    if contrast <= 0:
        raise ValueError("contrast must be > 0")
    if heterogeneity_db < 0:
        raise ValueError("heterogeneity_db must be >= 0")

    rng = np.random.default_rng(seed)
    graph_config = graph_config or GraphConfig()

    geometries = NodeGeometries.regular_grid(height, width, block=block)
    n_nodes = geometries.n_nodes
    corridor = _corridor_mask(geometries)
    edge_index, edge_type = build_spatial_edges(geometries, graph_config, corridor)

    onset = _simulate_onsets(
        n_nodes=n_nodes,
        n_time=n_time,
        edge_index=edge_index,
        corridor=corridor,
        base_hazard=base_hazard,
        neighbor_hazard=neighbor_hazard,
        corridor_hazard=corridor_hazard,
        n_seeds=n_seeds,
        rng=rng,
    )
    stack = _render_stack(
        geometries=geometries,
        onset=onset,
        n_time=n_time,
        construction_periods=construction_periods,
        looks=looks,
        contrast=contrast,
        heterogeneity_db=heterogeneity_db,
        rng=rng,
    )
    features = aggregate_stack(stack, geometries, statistics=("mean", "std"))

    year_index = np.arange(n_time) // periods_per_year
    n_years = int(year_index.max()) + 1
    annual_labels = np.zeros((n_years, n_nodes), dtype=np.int64)
    sealed = onset >= 0
    annual_labels[year_index[onset[sealed]], np.flatnonzero(sealed)] = 1

    graph = SpatioTemporalGraph(
        features=features,
        spatial_edge_index=edge_index,
        spatial_edge_type=edge_type,
        annual_labels=annual_labels,
        year_index=year_index,
        geometries=geometries,
    )
    return SyntheticLandTakeScenario(
        graph=graph,
        stack=stack,
        geometries=geometries,
        onset=onset,
        corridor=corridor,
        periods_per_year=periods_per_year,
    )


def _corridor_mask(geometries: NodeGeometries) -> np.ndarray:
    """Nodes crossed by a diagonal "road" spanning the domain."""
    height, width = geometries.shape
    rows, cols = geometries.centroids[:, 0], geometries.centroids[:, 1]
    # Distance of each centroid to the line row = col * height / width.
    expected = cols * (height / max(width, 1))
    tolerance = max(height, width) / 24.0
    return np.abs(rows - expected) <= tolerance


def _simulate_onsets(
    *,
    n_nodes: int,
    n_time: int,
    edge_index: np.ndarray,
    corridor: np.ndarray,
    base_hazard: float,
    neighbor_hazard: float,
    corridor_hazard: float,
    n_seeds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run the contagion process that decides when each node seals."""
    onset = np.full(n_nodes, -1, dtype=np.int64)
    sealed = np.zeros(n_nodes, dtype=bool)

    seeds = rng.choice(n_nodes, size=min(n_seeds, n_nodes), replace=False)
    sealed[seeds] = True
    onset[seeds] = 0

    src, dst = (edge_index[0], edge_index[1]) if edge_index.size else (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )
    degree = np.maximum(np.bincount(dst, minlength=n_nodes), 1)

    for t in range(1, n_time):
        sealed_neighbors = np.bincount(dst, weights=sealed[src], minlength=n_nodes)
        hazard = (
            base_hazard
            + neighbor_hazard * (sealed_neighbors / degree)
            + corridor_hazard * corridor
        )
        fires = (rng.random(n_nodes) < np.clip(hazard, 0.0, 1.0)) & ~sealed
        onset[fires] = t
        sealed |= fires
    return onset


def _render_stack(
    *,
    geometries: NodeGeometries,
    onset: np.ndarray,
    n_time: int,
    construction_periods: int,
    looks: int,
    contrast: float,
    heterogeneity_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Turn per-node onsets into a speckled ``(T, 3, H, W)`` raster."""
    height, width = geometries.shape
    labels = geometries.labels

    times = np.arange(n_time)[:, None]
    sealed = onset[None, :]
    has_onset = sealed >= 0
    # Phase of each (t, node): 0 natural, 1 under construction, 2 sealed.
    building = has_onset & (times >= sealed - construction_periods) & (times < sealed)
    done = has_onset & (times >= sealed)

    def blend(natural: float, construction: float, built: float) -> np.ndarray:
        """Per-(time, node) level, with the change signature scaled by contrast."""
        node_db = np.full((n_time, geometries.n_nodes), natural, dtype=np.float64)
        node_db[building] = natural + contrast * (construction - natural)
        node_db[done] = natural + contrast * (built - natural)
        return node_db

    vv_db = blend(
        _SIGMA0_DB["natural_vv"],
        _SIGMA0_DB["construction_vv"],
        _SIGMA0_DB["sealed_vv"],
    )
    vh_db = blend(
        _SIGMA0_DB["natural_vh"],
        _SIGMA0_DB["construction_vh"],
        _SIGMA0_DB["sealed_vh"],
    )
    coherence = blend(
        _COHERENCE["stable"], _COHERENCE["construction"], _COHERENCE["sealed"]
    )

    # Per-node offsets standing in for land-cover heterogeneity, constant in
    # time so they cannot be mistaken for a change signal.
    offsets = rng.normal(0.0, heterogeneity_db, size=(1, geometries.n_nodes))

    stack = np.empty((n_time, 3, height, width), dtype=np.float64)
    for t in range(n_time):
        vv_power = 10.0 ** ((vv_db[t] + offsets[0]) / 10.0)
        vh_power = 10.0 ** ((vh_db[t] + offsets[0]) / 10.0)
        speckle_vv = rng.gamma(looks, 1.0 / looks, size=(height, width))
        speckle_vh = rng.gamma(looks, 1.0 / looks, size=(height, width))
        stack[t, 0] = to_decibel(vv_power[labels] * speckle_vv)
        stack[t, 1] = to_decibel(vh_power[labels] * speckle_vh)
        stack[t, 2] = np.clip(
            coherence[t][labels]
            + rng.normal(0.0, 0.05 / min(contrast, 1.0), size=(height, width)),
            0.0,
            1.0,
        )
    return stack

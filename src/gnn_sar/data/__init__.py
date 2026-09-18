"""Phase 1–2: from Sentinel rasters to a spatio-temporal graph.

The modules that need an optional dependency — :mod:`gnn_sar.data.raster` and
:mod:`gnn_sar.data.ingest` (rasterio), :mod:`gnn_sar.data.preprocess` (rasterio
and ESA SNAP), :mod:`gnn_sar.data.cdse` (requests) — are *not* re-exported here:
importing this package must stay cheap and pure-NumPy. Import them from their
own modules.
"""

from gnn_sar.data.features import (
    FeatureStandardizer,
    composite_by_period,
    fill_temporal_gaps,
    to_decibel,
)
from gnn_sar.data.graph import (
    EDGE_ADJACENCY,
    EDGE_CONNECTIVITY,
    EDGE_SELF_LOOP,
    EDGE_TEMPORAL,
    SpatioTemporalGraph,
    adjacency_edges,
    block_diagonal,
    build_spatial_edges,
    connectivity_edges,
    temporal_edges,
)
from gnn_sar.data.labels import (
    annual_land_take_labels,
    corridor_from_raster,
    node_fraction,
    node_state_from_rasters,
)
from gnn_sar.data.segmentation import (
    NodeGeometries,
    aggregate_stack,
    slic_geometries,
)
from gnn_sar.data.sites import PILOT_SITES, StudySite, get_site
from gnn_sar.data.sentinel import (
    PeriodGrid,
    scl_valid_mask,
    sensing_date,
    sentinel1_channels,
    sentinel2_indices,
)
from gnn_sar.data.study_area import (
    SourceStack,
    StudyArea,
    build_study_area,
    load_study_area,
    study_area_from_features,
)
from gnn_sar.data.synthetic import (
    SyntheticLandTakeScenario,
    make_synthetic_scenario,
)

__all__ = [
    "EDGE_ADJACENCY",
    "EDGE_CONNECTIVITY",
    "EDGE_SELF_LOOP",
    "EDGE_TEMPORAL",
    "FeatureStandardizer",
    "NodeGeometries",
    "PILOT_SITES",
    "PeriodGrid",
    "SourceStack",
    "SpatioTemporalGraph",
    "StudyArea",
    "StudySite",
    "SyntheticLandTakeScenario",
    "adjacency_edges",
    "aggregate_stack",
    "annual_land_take_labels",
    "block_diagonal",
    "build_spatial_edges",
    "build_study_area",
    "composite_by_period",
    "connectivity_edges",
    "corridor_from_raster",
    "fill_temporal_gaps",
    "get_site",
    "load_study_area",
    "make_synthetic_scenario",
    "node_fraction",
    "node_state_from_rasters",
    "scl_valid_mask",
    "sensing_date",
    "sentinel1_channels",
    "sentinel2_indices",
    "slic_geometries",
    "study_area_from_features",
    "temporal_edges",
    "to_decibel",
]

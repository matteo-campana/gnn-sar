"""Phase 1–2: from Sentinel-1 rasters to a spatio-temporal graph."""

from gnn_sar.data.attributes import (
    broadcast_static,
    concatenate_features,
    geometric_attributes,
    proximity_attributes,
)
from gnn_sar.data.features import (
    FeatureStandardizer,
    composite_by_period,
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
from gnn_sar.data.segmentation import (
    NodeGeometries,
    aggregate_stack,
    slic_geometries,
)
from gnn_sar.data.texture import GLCM_PROPERTIES, glcm_texture, glcm_texture_stack
from gnn_sar.data.synthetic import (
    SyntheticLandTakeScenario,
    make_synthetic_scenario,
)

__all__ = [
    "EDGE_ADJACENCY",
    "GLCM_PROPERTIES",
    "EDGE_CONNECTIVITY",
    "EDGE_SELF_LOOP",
    "EDGE_TEMPORAL",
    "FeatureStandardizer",
    "NodeGeometries",
    "SpatioTemporalGraph",
    "SyntheticLandTakeScenario",
    "adjacency_edges",
    "aggregate_stack",
    "block_diagonal",
    "broadcast_static",
    "build_spatial_edges",
    "composite_by_period",
    "concatenate_features",
    "connectivity_edges",
    "geometric_attributes",
    "glcm_texture",
    "glcm_texture_stack",
    "proximity_attributes",
    "make_synthetic_scenario",
    "slic_geometries",
    "temporal_edges",
    "to_decibel",
]

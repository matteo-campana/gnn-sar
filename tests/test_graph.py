from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.config import GraphConfig
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
from gnn_sar.data.segmentation import NodeGeometries


def test_adjacency_links_exactly_the_touching_segments(label_image):
    geometries = NodeGeometries.from_label_image(label_image)

    edges = adjacency_edges(geometries)

    # 0-1, 0-2, 1-3, 2-3 touch; 0-3 and 1-2 only meet at a corner.
    assert {tuple(edge) for edge in edges.T} == {(0, 1), (0, 2), (1, 3), (2, 3)}


def test_adjacency_of_a_single_segment_is_empty():
    geometries = NodeGeometries.from_label_image(np.zeros((4, 4), dtype=int))

    assert adjacency_edges(geometries).shape == (2, 0)


def test_connectivity_respects_the_corridor_mask():
    geometries = NodeGeometries.regular_grid(8, 8, block=2)
    corridor = np.zeros(geometries.n_nodes, dtype=bool)
    corridor[[0, 1, 2]] = True

    edges = connectivity_edges(geometries, n_neighbors=3, corridor=corridor)

    assert edges.size > 0
    assert set(np.unique(edges)) <= {0, 1, 2}


def test_connectivity_honours_the_radius():
    geometries = NodeGeometries.regular_grid(8, 8, block=2)

    near = connectivity_edges(geometries, n_neighbors=4, radius=2.1)
    far = connectivity_edges(geometries, n_neighbors=4, radius=None)

    assert near.shape[1] < far.shape[1]


def test_connectivity_rejects_a_mismatched_corridor():
    geometries = NodeGeometries.regular_grid(4, 4, block=2)

    with pytest.raises(ValueError, match="corridor must be"):
        connectivity_edges(geometries, 2, corridor=np.ones(3, dtype=bool))


def test_edge_types_partition_the_edge_set():
    geometries = NodeGeometries.regular_grid(12, 12, block=2)
    config = GraphConfig(n_neighbors=6, self_loops=True)

    edge_index, edge_type = build_spatial_edges(geometries, config)

    assert edge_index.shape[1] == edge_type.shape[0]
    assert set(np.unique(edge_type)) == {EDGE_ADJACENCY, EDGE_CONNECTIVITY, EDGE_SELF_LOOP}
    # No edge is claimed by two families: the columns are unique.
    assert np.unique(edge_index, axis=1).shape[1] == edge_index.shape[1]


def test_undirected_construction_is_symmetric():
    geometries = NodeGeometries.regular_grid(6, 6, block=2)

    edge_index, _ = build_spatial_edges(geometries, GraphConfig(n_neighbors=2))

    forward = {tuple(edge) for edge in edge_index.T}
    assert forward == {(dst, src) for src, dst in forward}


def test_temporal_edges_only_ever_point_forward():
    edges = temporal_edges(n_nodes=5, n_time=4, order=2)

    source_frame, target_frame = edges[0] // 5, edges[1] // 5
    assert np.all(source_frame < target_frame)
    assert np.all(edges[0] % 5 == edges[1] % 5)  # same node, different frame
    assert edges.shape[1] == 5 * (3 + 2)  # 3 lag-1 pairs plus 2 lag-2 pairs


def test_temporal_edges_rejects_order_zero():
    with pytest.raises(ValueError, match="order must be >= 1"):
        temporal_edges(3, 3, order=0)


def test_block_diagonal_keeps_every_frame_separate():
    edges = np.array([[0, 1], [1, 2]])

    replicated = block_diagonal(edges, n_nodes=3, n_time=4)

    assert replicated.shape == (2, 8)
    assert np.all(replicated[0] // 3 == replicated[1] // 3)
    assert set(np.unique(replicated // 3)) == {0, 1, 2, 3}


def _graph(n_time: int = 4, n_nodes: int = 9) -> SpatioTemporalGraph:
    geometries = NodeGeometries.regular_grid(6, 6, block=2)
    edge_index, edge_type = build_spatial_edges(geometries, GraphConfig(n_neighbors=2))
    year_index = np.arange(n_time) // 2
    return SpatioTemporalGraph(
        features=np.zeros((n_time, n_nodes, 3)),
        spatial_edge_index=edge_index,
        spatial_edge_type=edge_type,
        annual_labels=np.zeros((int(year_index.max()) + 1, n_nodes), dtype=np.int64),
        year_index=year_index,
        geometries=geometries,
    )


def test_super_graph_is_block_lower_triangular():
    graph = _graph()

    edge_index, edge_type = graph.super_edge_index(order=1)

    frames = edge_index // graph.n_nodes
    assert np.all(frames[0] <= frames[1])
    assert set(np.unique(edge_type)) <= {EDGE_ADJACENCY, EDGE_CONNECTIVITY, EDGE_TEMPORAL}
    temporal = edge_type == EDGE_TEMPORAL
    assert np.all(frames[0][temporal] + 1 == frames[1][temporal])


def test_graph_rejects_edges_pointing_outside_the_node_set():
    with pytest.raises(ValueError, match="outside the graph"):
        SpatioTemporalGraph(
            features=np.zeros((2, 3, 1)),
            spatial_edge_index=np.array([[0], [7]]),
            spatial_edge_type=np.array([0]),
        )


def test_graph_rejects_annual_labels_without_a_year_index():
    with pytest.raises(ValueError, match="require year_index"):
        SpatioTemporalGraph(
            features=np.zeros((2, 3, 1)),
            spatial_edge_index=np.zeros((2, 0), dtype=np.int64),
            spatial_edge_type=np.zeros(0, dtype=np.int64),
            annual_labels=np.zeros((1, 3), dtype=np.int64),
        )


def test_window_keeps_the_topology_and_slices_time():
    graph = _graph(n_time=6)

    windowed = graph.window(2, 3)

    assert windowed.n_time == 3
    assert windowed.year_index.tolist() == graph.year_index[2:5].tolist()
    assert np.array_equal(windowed.spatial_edge_index, graph.spatial_edge_index)


def test_window_rejects_a_range_past_the_series():
    with pytest.raises(ValueError, match="out of range"):
        _graph(n_time=4).window(3, 3)


def test_splits_are_disjoint_exhaustive_and_geographic():
    graph = _graph()

    masks = graph.split_nodes((0.5, 0.25, 0.25), axis="col")

    stacked = np.stack(list(masks.values()))
    assert stacked.sum(axis=0).tolist() == [1] * graph.n_nodes
    columns = graph.geometries.centroids[:, 1]
    # A geographic cut means the splits occupy separate bands of the domain.
    assert columns[masks["train"]].max() <= columns[masks["test"]].min()


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="summing to 1"):
        _graph().split_nodes((0.5, 0.2, 0.2))

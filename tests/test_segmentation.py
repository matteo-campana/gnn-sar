from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from gnn_sar.data.segmentation import NodeGeometries, aggregate_stack, slic_geometries


def test_from_label_image_makes_ids_contiguous():
    geometries = NodeGeometries.from_label_image(np.array([[10, 10], [40, 99]]))

    assert geometries.n_nodes == 3
    assert sorted(np.unique(geometries.labels)) == [0, 1, 2]
    assert geometries.areas.tolist() == [2.0, 1.0, 1.0]


def test_centroids_are_the_mean_pixel_position(label_image):
    geometries = NodeGeometries.from_label_image(label_image)

    # Segment 0 covers the top-left 2x2 block.
    assert geometries.centroids[0].tolist() == [0.5, 0.5]
    # Segment 3 is the single bottom-right pixel.
    assert geometries.centroids[3].tolist() == [2.0, 2.0]


def test_regular_grid_covers_every_pixel_exactly_once():
    geometries = NodeGeometries.regular_grid(10, 7, block=3)

    assert geometries.areas.sum() == 70
    assert geometries.n_nodes == 4 * 3  # ceil(10/3) x ceil(7/3)
    assert geometries.labels.shape == (10, 7)


def test_regular_grid_rejects_a_non_positive_block():
    with pytest.raises(ValueError, match="block must be >= 1"):
        NodeGeometries.regular_grid(4, 4, block=0)


def test_aggregate_stack_matches_a_direct_per_segment_reduction():
    rng = np.random.default_rng(0)
    geometries = NodeGeometries.regular_grid(8, 8, block=2)
    stack = rng.normal(size=(3, 2, 8, 8))

    features = aggregate_stack(stack, geometries, ("mean", "std", "min", "max"))

    assert features.shape == (3, geometries.n_nodes, 8)
    mask = geometries.pixel_mask(5)
    band = stack[2, 1][mask]
    # Band 1 occupies channels 4..7, in the order the statistics were requested.
    assert features[2, 5, 4] == pytest.approx(band.mean())
    assert features[2, 5, 5] == pytest.approx(band.std())
    assert features[2, 5, 6] == pytest.approx(band.min())
    assert features[2, 5, 7] == pytest.approx(band.max())


def test_aggregate_stack_rejects_a_mismatched_raster():
    geometries = NodeGeometries.regular_grid(8, 8, block=2)

    with pytest.raises(ValueError, match="does not match geometries"):
        aggregate_stack(np.zeros((2, 1, 4, 4)), geometries)


def test_aggregate_stack_rejects_an_unknown_statistic():
    geometries = NodeGeometries.regular_grid(4, 4, block=2)

    with pytest.raises(ValueError, match="unknown statistics"):
        aggregate_stack(np.zeros((1, 1, 4, 4)), geometries, ("median",))


def test_rasterise_is_the_inverse_of_labelling(label_image):
    geometries = NodeGeometries.from_label_image(label_image)

    painted = geometries.rasterise(np.array([7.0, 8.0, 9.0, 10.0]))

    assert painted.shape == label_image.shape
    assert painted[0, 0] == 7.0 and painted[2, 2] == 10.0


def test_rasterise_rejects_a_wrong_length():
    geometries = NodeGeometries.regular_grid(4, 4, block=2)

    with pytest.raises(ValueError, match="expected leading dimension"):
        geometries.rasterise(np.zeros(3))


def test_slic_geometries_names_its_optional_dependency_when_it_is_missing():
    if importlib.util.find_spec("skimage") is not None:
        pytest.skip("scikit-image is installed, so the fallback path is unreachable")

    with pytest.raises(ImportError, match="scikit-image"):
        slic_geometries(np.zeros((1, 8, 8)))


def test_slic_geometries_segments_a_composite_when_scikit_image_is_available():
    pytest.importorskip("skimage")

    geometries = slic_geometries(np.random.default_rng(0).normal(size=(2, 16, 16)), 8)

    assert geometries.n_nodes >= 1
    assert geometries.shape == (16, 16)
    assert geometries.areas.sum() == 256

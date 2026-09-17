from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.attributes import (
    broadcast_static,
    concatenate_features,
    geometric_attributes,
    proximity_attributes,
)
from gnn_sar.data.segmentation import NodeGeometries


@pytest.fixture
def shapes() -> NodeGeometries:
    """Two 2x2 squares over two 1x4 strips — same area, different shape."""
    return NodeGeometries.from_label_image(
        np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    )


def test_equal_areas_give_equal_log_area(shapes):
    attributes = geometric_attributes(shapes)

    assert attributes[:, 0].tolist() == [pytest.approx(np.log1p(4.0))] * 4


def test_a_square_is_more_compact_than_a_strip(shapes):
    compactness = geometric_attributes(shapes)[:, 1]

    # 4*pi*A/P^2: the 2x2 square has perimeter 8, the 1x4 strip 10.
    assert compactness[0] == pytest.approx(4 * np.pi * 4 / 64)
    assert compactness[2] == pytest.approx(4 * np.pi * 4 / 100)
    assert compactness[0] > compactness[2]


def test_elongation_separates_strips_from_squares(shapes):
    elongation = geometric_attributes(shapes)[:, 2]

    assert elongation[0] == pytest.approx(0.0)
    assert elongation[2] == pytest.approx(0.75)


def test_shape_attributes_do_not_encode_position():
    """A translated unit must look identical, or the geographic split leaks."""
    left = NodeGeometries.from_label_image(
        np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    )
    right = NodeGeometries.from_label_image(
        np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    )

    assert np.allclose(geometric_attributes(left)[0], geometric_attributes(right)[0])


def test_proximity_is_one_on_the_reference_layer_and_decays_away(shapes):
    features, names = proximity_attributes(shapes, {"road": np.array([True] + [False] * 3)})

    assert names == ["road"]
    assert features[0, 0] == pytest.approx(1.0)
    assert np.all(features[1:, 0] < 1.0)
    assert np.all(features > 0.0)


def test_a_closer_node_scores_higher(shapes):
    features, _ = proximity_attributes(
        shapes, {"road": np.array([True, False, False, False])}
    )
    distances = np.linalg.norm(shapes.centroids - shapes.centroids[0], axis=1)

    # The ranking of the feature must invert the ranking of the distance.
    assert np.argsort(-features[:, 0]).tolist() == np.argsort(distances).tolist()


def test_an_empty_reference_layer_yields_a_zero_channel_rather_than_failing(shapes):
    features, _ = proximity_attributes(shapes, {"road": np.zeros(4, dtype=bool)})

    assert features.shape == (4, 1)
    assert np.all(features == 0.0)


def test_several_reference_layers_come_back_in_a_stable_order(shapes):
    features, names = proximity_attributes(
        shapes,
        {"road": np.array([True, False, False, False]), "built": np.array([False] * 3 + [True])},
    )

    assert names == ["built", "road"]
    assert features.shape == (4, 2)


def test_proximity_rejects_a_mismatched_reference(shapes):
    with pytest.raises(ValueError, match="must be"):
        proximity_attributes(shapes, {"road": np.ones(9, dtype=bool)})


def test_proximity_rejects_a_non_positive_decay(shapes):
    with pytest.raises(ValueError, match="decay must be > 0"):
        proximity_attributes(shapes, {"road": np.ones(4, dtype=bool)}, decay=0.0)


def test_broadcast_repeats_the_same_row_at_every_composite():
    static = np.array([[1.0, 2.0], [3.0, 4.0]])

    broadcast = broadcast_static(static, 5)

    assert broadcast.shape == (5, 2, 2)
    assert np.all(broadcast[0] == broadcast[4])


def test_broadcast_returns_a_writable_copy():
    broadcast = broadcast_static(np.zeros((2, 2)), 3)

    broadcast[0, 0, 0] = 9.0  # must not raise, and must not touch the other frames

    assert broadcast[1, 0, 0] == 0.0


def test_concatenate_joins_blocks_on_the_channel_axis():
    joined = concatenate_features(np.zeros((3, 4, 2)), np.ones((3, 4, 5)))

    assert joined.shape == (3, 4, 7)
    assert np.all(joined[..., 2:] == 1.0)


def test_concatenate_rejects_blocks_that_disagree_on_shape():
    with pytest.raises(ValueError, match="disagree on"):
        concatenate_features(np.zeros((3, 4, 2)), np.zeros((3, 5, 2)))


def test_concatenate_rejects_an_empty_call():
    with pytest.raises(ValueError, match="at least one feature block"):
        concatenate_features()

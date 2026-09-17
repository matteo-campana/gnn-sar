from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.features import FeatureStandardizer, composite_by_period, to_decibel


def test_to_decibel_matches_the_definition():
    assert to_decibel(np.array([1.0, 0.1])) == pytest.approx([0.0, -10.0], abs=1e-6)


def test_to_decibel_floors_instead_of_returning_minus_infinity():
    assert to_decibel(np.array([0.0]), floor=-35.0).tolist() == [-35.0]


def test_to_decibel_rejects_negative_power():
    with pytest.raises(ValueError, match="non-negative"):
        to_decibel(np.array([-1.0]))


def test_composite_reduces_each_period_independently():
    acquisitions = np.array([[1.0], [3.0], [10.0], [100.0]])

    composites = composite_by_period(acquisitions, np.array([0, 0, 1, 1]))

    assert composites[:, 0].tolist() == [2.0, 55.0]


def test_composite_carries_the_previous_value_into_an_empty_period():
    acquisitions = np.array([[5.0], [9.0]])

    composites = composite_by_period(acquisitions, np.array([1, 3]), n_periods=5)

    # Period 0 is back-filled, 2 carries 1 forward, 4 carries 3 forward.
    assert composites[:, 0].tolist() == [5.0, 5.0, 5.0, 9.0, 9.0]


def test_composite_can_mark_gaps_instead_of_filling_them():
    composites = composite_by_period(
        np.array([[5.0]]), np.array([1]), n_periods=3, fill="nan"
    )

    assert np.isnan(composites[0, 0]) and np.isnan(composites[2, 0])
    assert composites[1, 0] == 5.0


def test_composite_rejects_a_series_too_short_for_its_indices():
    with pytest.raises(ValueError, match="cannot hold period index"):
        composite_by_period(np.array([[1.0]]), np.array([4]), n_periods=2)


def test_composite_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="period indices"):
        composite_by_period(np.zeros((3, 1)), np.array([0, 1]))


def test_standardizer_uses_only_the_masked_nodes():
    features = np.zeros((2, 4, 1))
    features[:, :2, 0] = [[0.0, 2.0], [0.0, 2.0]]  # training nodes: mean 1, std 1
    features[:, 2:, 0] = 100.0  # held-out nodes must not move the statistics
    mask = np.array([True, True, False, False])

    standardizer = FeatureStandardizer().fit(features, mask)

    assert standardizer.mean[0] == pytest.approx(1.0)
    assert standardizer.scale[0] == pytest.approx(1.0)


def test_standardizer_leaves_a_constant_channel_finite():
    features = np.full((2, 3, 1), 7.0)

    transformed = FeatureStandardizer().fit_transform(features)

    assert np.all(transformed == 0.0)


def test_standardizer_refuses_to_transform_before_fitting():
    with pytest.raises(RuntimeError, match="must be fitted"):
        FeatureStandardizer().transform(np.zeros((1, 1, 1)))


def test_standardizer_rejects_a_channel_count_mismatch():
    standardizer = FeatureStandardizer().fit(np.zeros((2, 3, 4)))

    with pytest.raises(ValueError, match="expected 4 channels"):
        standardizer.transform(np.zeros((2, 3, 5)))

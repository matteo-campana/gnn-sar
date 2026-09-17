from __future__ import annotations

import numpy as np
import pytest
import torch

from gnn_sar.metrics.classification import confusion_matrix, sealing_metrics
from gnn_sar.metrics.delay import (
    detection_delay,
    early_warning_lead_time,
    onset_from_probability,
)


def test_onset_is_the_first_composite_above_the_threshold():
    probability = np.array([[0.1, 0.9], [0.8, 0.9], [0.9, 0.9]])

    assert onset_from_probability(probability, 0.5).tolist() == [1, 0]


def test_a_node_never_crossing_the_threshold_reports_minus_one():
    assert onset_from_probability(np.array([[0.1], [0.2]]), 0.5).tolist() == [-1]


def test_persistence_delays_a_detection_until_it_is_confirmed():
    # A single spike at t=1, then a real, sustained detection from t=3.
    probability = np.array([[0.1], [0.9], [0.1], [0.9], [0.9]])

    assert onset_from_probability(probability, 0.5, min_persistence=1).tolist() == [1]
    assert onset_from_probability(probability, 0.5, min_persistence=2).tolist() == [3]


def test_persistence_longer_than_the_series_detects_nothing():
    assert onset_from_probability(np.ones((2, 3)), 0.5, min_persistence=5).tolist() == [
        -1,
        -1,
        -1,
    ]


def test_onset_accepts_a_torch_tensor():
    assert onset_from_probability(torch.tensor([[0.2], [0.7]])).tolist() == [1]


def test_onset_rejects_a_non_positive_persistence():
    with pytest.raises(ValueError, match="min_persistence must be >= 1"):
        onset_from_probability(np.ones((2, 2)), min_persistence=0)


def test_delay_is_positive_when_the_model_is_late():
    delay = detection_delay(np.array([3, 1]), np.array([1, 1]))

    assert delay["mean_delay"] == pytest.approx(1.0)
    assert delay["detection_rate"] == pytest.approx(1.0)


def test_delay_is_negative_when_construction_is_caught_early():
    delay = detection_delay(np.array([2]), np.array([5]))

    assert delay["mean_delay"] == pytest.approx(-3.0)


def test_a_missed_node_is_charged_the_rest_of_the_series():
    without_penalty = detection_delay(np.array([-1, 2]), np.array([4, 2]))
    with_penalty = detection_delay(np.array([-1, 2]), np.array([4, 2]), n_time=10)

    assert without_penalty["mean_delay"] == pytest.approx(0.0)
    # The missed node is charged 10 - 4 = 6 composites, so silence is not free.
    assert with_penalty["mean_delay"] == pytest.approx(3.0)
    assert with_penalty["n_missed"] == 1.0
    assert with_penalty["detection_rate"] == pytest.approx(0.5)


def test_false_alarms_are_counted_over_the_nodes_that_never_sealed():
    delay = detection_delay(np.array([1, -1, 2, 0]), np.array([1, -1, -1, -1]))

    assert delay["false_alarm_rate"] == pytest.approx(2 / 3)


def test_delay_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ"):
        detection_delay(np.array([1, 2]), np.array([1]))


def test_lead_time_counts_only_alarms_raised_before_the_transition():
    # Node 0 is warned at t=0 and seals at t=3; node 1 is "warned" only after.
    probability = np.array([[0.9, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.9]])

    lead = early_warning_lead_time(probability, np.array([3, 2]))

    assert lead["mean_lead"] == pytest.approx(3.0)
    assert lead["alarm_rate"] == pytest.approx(0.5)


def test_lead_time_reports_no_lead_when_nothing_is_anticipated():
    lead = early_warning_lead_time(np.zeros((3, 2)), np.array([1, 2]))

    assert np.isnan(lead["mean_lead"])
    assert lead["alarm_rate"] == 0.0


def test_lead_time_rejects_a_node_count_mismatch():
    with pytest.raises(ValueError, match="nodes in probabilities"):
        early_warning_lead_time(np.zeros((3, 2)), np.array([1, 2, 3]))


def test_confusion_matrix_skips_unlabelled_entries():
    matrix = confusion_matrix([1, 0, 1, 1], [1, 0, 0, -1])

    assert matrix.tolist() == [[1, 1], [0, 1]]


def test_confusion_matrix_rejects_non_binary_values():
    with pytest.raises(ValueError, match="must be binary"):
        confusion_matrix([2, 0], [1, 0])


def test_sealing_metrics_reproduce_the_textbook_values():
    # tp = 2, fp = 1, fn = 1, tn = 4
    prediction = [1, 1, 1, 0, 0, 0, 0, 0]
    target = [1, 1, 0, 1, 0, 0, 0, 0]

    metrics = sealing_metrics(prediction, target)

    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert metrics["iou"] == pytest.approx(2 / 4)
    assert metrics["support"] == 3.0


def test_sealing_metrics_agree_with_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    prediction = rng.integers(0, 2, size=200)
    target = rng.integers(0, 2, size=200)

    metrics = sealing_metrics(prediction, target)

    assert metrics["f1"] == pytest.approx(sklearn_metrics.f1_score(target, prediction))
    assert metrics["kappa"] == pytest.approx(
        sklearn_metrics.cohen_kappa_score(target, prediction)
    )
    assert metrics["iou"] == pytest.approx(
        sklearn_metrics.jaccard_score(target, prediction)
    )


def test_perfect_agreement_gives_a_kappa_of_one():
    assert sealing_metrics([1, 0, 1, 0], [1, 0, 1, 0])["kappa"] == pytest.approx(1.0)

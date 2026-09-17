from __future__ import annotations

import pytest
import torch

from gnn_sar.config import LossConfig
from gnn_sar.data.targets import IGNORE_INDEX
from gnn_sar.losses.weak import WeakTemporalSupervisionLoss, aggregate_year_logit
from gnn_sar.models.stgnn import STGNNOutput


def _output(onset_logits: torch.Tensor, horizon: torch.Tensor | None = None) -> STGNNOutput:
    return STGNNOutput(
        onset_logits=onset_logits,
        horizon_logits=horizon,
        embeddings=torch.zeros(*onset_logits.shape, 1),
    )


def test_noisy_or_is_the_probability_of_at_least_one_onset():
    logits = torch.randn(4, 7)
    expected = 1.0 - torch.prod(1.0 - torch.sigmoid(logits), dim=0)

    aggregated = torch.sigmoid(aggregate_year_logit(logits, "noisy_or"))

    assert torch.allclose(aggregated, expected, atol=1e-5)


def test_noisy_or_stays_finite_at_saturated_logits():
    logits = torch.tensor([[-60.0, 60.0], [-60.0, -60.0]], requires_grad=True)

    aggregated = aggregate_year_logit(logits, "noisy_or")
    aggregated.sum().backward()

    assert torch.isfinite(aggregated).all()
    assert torch.isfinite(logits.grad).all()


def test_max_aggregation_picks_the_strongest_composite():
    logits = torch.tensor([[-2.0], [3.0], [0.0]])

    assert aggregate_year_logit(logits, "max").tolist() == [3.0]


def test_logsumexp_approaches_max_as_the_temperature_falls():
    logits = torch.tensor([[-2.0], [3.0], [0.0]])

    cold = aggregate_year_logit(logits, "logsumexp", temperature=0.01)

    assert cold.item() == pytest.approx(3.0, abs=1e-2)


def test_aggregation_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate_year_logit(torch.zeros(2, 2), "mean")  # type: ignore[arg-type]


def test_aggregation_rejects_an_empty_year():
    with pytest.raises(ValueError, match="at least one composite"):
        aggregate_year_logit(torch.zeros(0, 2))


def _fixture(n_time: int = 4, n_nodes: int = 6):
    year_index = torch.tensor([0, 0, 1, 1])[:n_time]
    annual_labels = torch.zeros(2, n_nodes, dtype=torch.long)
    annual_labels[1, :2] = 1  # two nodes sealed in the second year
    return year_index, annual_labels


def test_a_confident_correct_prediction_beats_a_confident_wrong_one():
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(
        LossConfig(smoothness_weight=0.0, sparsity_weight=0.0, horizon_weight=0.0)
    )

    correct = torch.full((4, 6), -8.0)
    correct[2, :2] = 6.0  # onset inside the labelled year, for the right nodes
    wrong = torch.full((4, 6), -8.0)
    wrong[2, 4:] = 6.0  # onset for the wrong nodes

    correct_loss = criterion(
        _output(correct), year_index=year_index, annual_labels=annual_labels
    )["total"]
    wrong_loss = criterion(
        _output(wrong), year_index=year_index, annual_labels=annual_labels
    )["total"]

    assert correct_loss < wrong_loss
    assert correct_loss < 0.05


def test_an_onset_in_the_wrong_year_is_penalised():
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(
        LossConfig(smoothness_weight=0.0, sparsity_weight=0.0, horizon_weight=0.0)
    )

    right_year = torch.full((4, 6), -8.0)
    right_year[3, :2] = 6.0
    wrong_year = torch.full((4, 6), -8.0)
    wrong_year[0, :2] = 6.0  # same nodes, but a year too early

    assert (
        criterion(_output(right_year), year_index=year_index, annual_labels=annual_labels)[
            "total"
        ]
        < criterion(
            _output(wrong_year), year_index=year_index, annual_labels=annual_labels
        )["total"]
    )


def test_the_sub_annual_placement_inside_the_labelled_year_is_not_penalised():
    """The heart of weak supervision: the year is supervised, the month is free."""
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(
        LossConfig(smoothness_weight=0.0, sparsity_weight=0.0, horizon_weight=0.0)
    )

    early = torch.full((4, 6), -8.0)
    early[2, :2] = 6.0
    late = torch.full((4, 6), -8.0)
    late[3, :2] = 6.0

    early_loss = criterion(
        _output(early), year_index=year_index, annual_labels=annual_labels
    )["total"]
    late_loss = criterion(
        _output(late), year_index=year_index, annual_labels=annual_labels
    )["total"]

    assert early_loss == pytest.approx(late_loss.item(), abs=1e-6)


def test_nodes_outside_the_mask_do_not_contribute():
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(
        LossConfig(smoothness_weight=0.0, sparsity_weight=0.0, horizon_weight=0.0)
    )
    logits = torch.full((4, 6), -8.0)
    logits[2, :2] = 6.0

    mask = torch.zeros(6, dtype=torch.bool)
    mask[:2] = True
    masked = criterion(
        _output(logits),
        year_index=year_index,
        annual_labels=annual_labels,
        node_mask=mask,
    )["total"]

    # With only the two positive nodes in scope the loss is still near zero,
    # and changing a held-out node leaves it untouched.
    disturbed = logits.clone()
    disturbed[:, 3] = 9.0
    assert masked < 0.05
    assert masked == pytest.approx(
        criterion(
            _output(disturbed),
            year_index=year_index,
            annual_labels=annual_labels,
            node_mask=mask,
        )["total"].item(),
        abs=1e-6,
    )


def test_unknown_annual_labels_are_skipped():
    year_index, annual_labels = _fixture()
    annual_labels[:] = IGNORE_INDEX
    criterion = WeakTemporalSupervisionLoss(
        LossConfig(smoothness_weight=0.0, sparsity_weight=0.0, horizon_weight=0.0)
    )

    terms = criterion(
        _output(torch.randn(4, 6)), year_index=year_index, annual_labels=annual_labels
    )

    assert terms["weak_annual"].item() == 0.0


def test_the_sparsity_term_punishes_spreading_a_year_over_every_composite():
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(LossConfig())

    concentrated = torch.full((4, 6), -8.0)
    concentrated[2] = 6.0
    spread = torch.full((4, 6), 1.0)  # a bit of onset probability everywhere

    assert (
        criterion(
            _output(concentrated), year_index=year_index, annual_labels=annual_labels
        )["sparsity"]
        < criterion(_output(spread), year_index=year_index, annual_labels=annual_labels)[
            "sparsity"
        ]
    )


def test_the_smoothness_term_punishes_disagreement_across_an_edge():
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(LossConfig())
    edges = torch.tensor([[0, 1], [1, 0]])

    agreeing = torch.zeros(4, 6)
    disagreeing = torch.zeros(4, 6)
    disagreeing[:, 0] = 8.0
    disagreeing[:, 1] = -8.0

    kwargs = {
        "year_index": year_index,
        "annual_labels": annual_labels,
        "spatial_edge_index": edges,
    }
    assert (
        criterion(_output(agreeing), **kwargs)["smoothness"].item() == 0.0
        < criterion(_output(disagreeing), **kwargs)["smoothness"].item()
    )


def test_the_strong_term_is_used_when_exact_dates_are_available():
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(LossConfig())
    state = torch.zeros(4, 6, dtype=torch.long)
    state[2:, :2] = 1

    logits = torch.full((4, 6), -8.0)
    logits[2, :2] = 6.0

    terms = criterion(
        _output(logits),
        year_index=year_index,
        annual_labels=annual_labels,
        state_labels=state,
    )

    assert terms["supervised"].item() < 0.05


def test_the_horizon_term_only_scores_the_labelled_entries():
    year_index, annual_labels = _fixture()
    criterion = WeakTemporalSupervisionLoss(LossConfig())
    horizon_logits = torch.full((4, 6), -8.0)
    target = torch.full((4, 6), IGNORE_INDEX, dtype=torch.long)
    target[0, :] = 0  # the model is confidently negative, and right

    terms = criterion(
        _output(torch.full((4, 6), -8.0), horizon_logits),
        year_index=year_index,
        annual_labels=annual_labels,
        horizon_target=target,
    )

    assert terms["horizon"].item() < 0.01


def test_the_loss_rejects_a_year_index_of_the_wrong_length():
    _, annual_labels = _fixture()

    with pytest.raises(ValueError, match="year_index must be"):
        WeakTemporalSupervisionLoss()(
            _output(torch.zeros(4, 6)),
            year_index=torch.zeros(3, dtype=torch.long),
            annual_labels=annual_labels,
        )


def test_the_total_is_the_weighted_sum_of_its_parts():
    year_index, annual_labels = _fixture()
    config = LossConfig(smoothness_weight=0.5, sparsity_weight=0.25, horizon_weight=0.0)
    criterion = WeakTemporalSupervisionLoss(config)

    terms = criterion(
        _output(torch.randn(4, 6)),
        year_index=year_index,
        annual_labels=annual_labels,
        spatial_edge_index=torch.tensor([[0, 1], [1, 0]]),
    )

    expected = (
        terms["weak_annual"]
        + config.sparsity_weight * terms["sparsity"]
        + config.smoothness_weight * terms["smoothness"]
    )
    assert terms["total"].item() == pytest.approx(expected.item(), abs=1e-6)


def test_gradients_reach_the_logits():
    year_index, annual_labels = _fixture()
    logits = torch.randn(4, 6, requires_grad=True)

    WeakTemporalSupervisionLoss()(
        _output(logits, torch.randn(4, 6)),
        year_index=year_index,
        annual_labels=annual_labels,
        spatial_edge_index=torch.tensor([[0, 1], [1, 0]]),
    )["total"].backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0

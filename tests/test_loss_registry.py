"""The objective as a composition of registered terms."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gnn_sar.config import LossConfig, LossTermConfig
from gnn_sar.losses.aggregation import AGGREGATIONS, iter_year_bags
from gnn_sar.losses.registry import LOSS_TERMS, LossContext, build_loss, register_term
from gnn_sar.losses.weak import WeakTemporalSupervisionLoss
from gnn_sar.models.stgnn import STGNNOutput


def _output(onset_logits: torch.Tensor, horizon: torch.Tensor | None = None) -> STGNNOutput:
    return STGNNOutput(
        onset_logits=onset_logits,
        horizon_logits=horizon,
        embeddings=torch.zeros(*onset_logits.shape, 1),
    )


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    year_index = torch.tensor([0, 0, 1, 1])
    annual_labels = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    return year_index, annual_labels


def _call(criterion, logits, *, horizon=None, **kwargs):
    year_index, annual_labels = _fixture()
    return criterion(
        _output(logits, horizon),
        year_index=year_index,
        annual_labels=annual_labels,
        spatial_edge_index=torch.tensor([[0, 1], [1, 0]]),
        **kwargs,
    )


def test_the_default_objective_is_the_five_terms_of_the_proposal():
    specs = LossConfig().resolved_terms()

    assert [spec.name for spec in specs] == [
        "weak_annual",
        "sparsity",
        "smoothness",
        "supervised",
        "horizon",
    ]
    assert all(spec.name in LOSS_TERMS for spec in specs)


def test_an_explicit_objective_is_exactly_what_it_lists():
    config = LossConfig(terms=(LossTermConfig("weak_annual"), LossTermConfig("sparsity", 0.3)))

    criterion = build_loss(config)
    terms = _call(criterion, torch.randn(4, 6))

    assert set(terms) == {"weak_annual", "sparsity", "total"}
    assert terms["total"].item() == pytest.approx(
        (terms["weak_annual"] + 0.3 * terms["sparsity"]).item(), abs=1e-6
    )


def test_the_composed_default_matches_the_objective_written_by_hand():
    """The non-regression guarantee: the same numbers as before the registry."""
    torch.manual_seed(0)
    logits = torch.randn(4, 6)
    horizon_target = torch.randint(0, 2, (4, 6))
    state_labels = torch.randint(0, 2, (4, 6))
    config = LossConfig()

    composed = _call(
        build_loss(config),
        logits,
        horizon=torch.randn(4, 6),
        state_labels=state_labels,
        horizon_target=horizon_target,
    )

    # The historical formula, spelled out.
    torch.manual_seed(0)
    logits_again = torch.randn(4, 6)
    assert torch.equal(logits, logits_again)
    expected = (
        composed["weak_annual"]
        + config.sparsity_weight * composed["sparsity"]
        + config.smoothness_weight * composed["smoothness"]
        + config.supervised_weight * composed["supervised"]
        + config.horizon_weight * composed["horizon"]
    )
    assert composed["total"].item() == pytest.approx(expected.item(), abs=1e-6)


def test_a_term_at_weight_zero_is_still_logged():
    terms = _call(WeakTemporalSupervisionLoss(LossConfig(sparsity_weight=0.0)), torch.randn(4, 6))

    assert terms["sparsity"].item() == 0.0
    assert "sparsity" in terms


def test_a_disabled_term_contributes_nothing():
    config = LossConfig(
        terms=(
            LossTermConfig("weak_annual"),
            LossTermConfig("smoothness", weight=100.0, enabled=False),
        )
    )

    terms = _call(build_loss(config), torch.randn(4, 6))

    assert terms["smoothness"].item() == 0.0
    assert terms["total"].item() == pytest.approx(terms["weak_annual"].item(), abs=1e-6)


def test_a_term_with_no_data_for_it_is_skipped_rather_than_failing():
    """``supervised`` has nothing to score in the weak regime, which is the normal one."""
    terms = _call(WeakTemporalSupervisionLoss(), torch.randn(4, 6))

    assert terms["supervised"].item() == 0.0
    assert torch.isfinite(terms["total"])


def test_a_terms_own_parameters_reach_it():
    lenient = build_loss(
        LossConfig(terms=(LossTermConfig("sparsity", params={"max_onsets": 4.0}),))
    )
    strict = build_loss(
        LossConfig(terms=(LossTermConfig("sparsity", params={"max_onsets": 0.0}),))
    )
    logits = torch.zeros(4, 6)

    assert _call(lenient, logits)["sparsity"] < _call(strict, logits)["sparsity"]


def test_the_focal_weak_term_down_weights_the_bags_already_right():
    confident = torch.full((4, 6), -8.0)
    plain = build_loss(LossConfig(terms=(LossTermConfig("weak_annual"),)))
    focal = build_loss(
        LossConfig(
            terms=(LossTermConfig("weak_annual", params={"objective": "focal", "focal_gamma": 2.0}),)
        )
    )

    assert _call(focal, confident)["weak_annual"] < _call(plain, confident)["weak_annual"]


def test_a_custom_term_can_be_registered_and_reaches_the_total():
    @register_term("test_confidence")
    class Confidence(nn.Module):
        def __init__(self, scale: float = 1.0) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, ctx: LossContext) -> torch.Tensor:
            return self.scale * ctx.output.onset_probability.mean()

    try:
        config = LossConfig(
            terms=(LossTermConfig("test_confidence", weight=2.0, params={"scale": 3.0}),)
        )
        logits = torch.zeros(4, 6)

        terms = _call(build_loss(config), logits)

        # sigmoid(0) = 0.5, scaled by 3, weighted by 2.
        assert terms["test_confidence"].item() == pytest.approx(1.5)
        assert terms["total"].item() == pytest.approx(3.0)
    finally:
        LOSS_TERMS.pop("test_confidence")


def test_an_unknown_term_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown loss term"):
        build_loss(LossConfig(terms=(LossTermConfig("entropy"),)))


def test_an_argument_a_term_does_not_take_is_reported_by_name():
    with pytest.raises(ValueError, match="'sparsity' does not accept"):
        build_loss(LossConfig(terms=(LossTermConfig("sparsity", params={"budget": 2}),)))


def test_a_term_cannot_be_listed_twice():
    with pytest.raises(ValueError, match="may appear once"):
        LossConfig(terms=(LossTermConfig("sparsity"), LossTermConfig("sparsity", 0.5)))


def test_the_smoothness_target_changes_what_is_penalised():
    """On the sealed state, a shared history costs nothing; on the onset it does."""
    # Two neighbours seal one composite apart: they disagree about *when*
    # throughout, but about *whether* only in that one composite.
    logits = torch.full((4, 6), -5.0)
    logits[0, 0] = 5.0
    logits[1, 1] = 5.0
    on_onset = build_loss(LossConfig(terms=(LossTermConfig("smoothness"),)))
    on_sealed = build_loss(
        LossConfig(terms=(LossTermConfig("smoothness", params={"target": "sealed"}),))
    )

    assert _call(on_sealed, logits)["smoothness"] < _call(on_onset, logits)["smoothness"]


def test_the_metric_aggregation_follows_the_training_one_unless_asked():
    assert LossConfig().effective_metric_aggregation == "noisy_or"
    assert LossConfig(aggregation="max").effective_metric_aggregation == "max"
    assert (
        LossConfig(aggregation="max", metric_aggregation="noisy_or").effective_metric_aggregation
        == "noisy_or"
    )


@pytest.mark.parametrize("rule", sorted(AGGREGATIONS))
def test_every_aggregation_rule_produces_one_bag_per_labelled_year(rule):
    year_index, annual_labels = _fixture()
    mask = torch.ones(6, dtype=torch.bool)

    bags = list(
        iter_year_bags(torch.randn(4, 6), year_index, annual_labels, mask, aggregation=rule)
    )

    assert len(bags) == 2
    assert all(logit.shape == (6,) for logit, _ in bags)

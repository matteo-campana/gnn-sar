"""Dotted overrides and the parameter sweep built on them."""

from __future__ import annotations

import json

import pytest

from gnn_sar.config import (
    ExperimentConfig,
    OptimizerConfig,
    TrainingConfig,
    apply_overrides,
)
from gnn_sar.training.cli import build_parser, _parse_overrides, _resolve_config, run_sweep
from gnn_sar.training.sweep import expand_sweep, load_sweep


def _sweep_file(tmp_path, **payload):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_dotted_key_reaches_a_nested_field():
    config = apply_overrides(
        ExperimentConfig(),
        {"training.optimizer.name": "sgd", "training.loss.smoothness_weight": 0.2},
    )

    assert config.training.optimizer.name == "sgd"
    assert config.training.loss.smoothness_weight == pytest.approx(0.2)


def test_a_dotted_key_reaches_inside_a_free_form_params_mapping():
    config = apply_overrides(
        ExperimentConfig(), {"training.optimizer.params.momentum": 0.9}
    )

    assert config.training.optimizer.params == {"momentum": 0.9}


def test_an_override_leaves_everything_else_alone():
    base = ExperimentConfig()

    config = apply_overrides(base, {"training.max_epochs": 7})

    assert config.training.max_epochs == 7
    assert config.model == base.model
    assert config.training.loss == base.training.loss


def test_an_unknown_dotted_key_is_reported_rather_than_ignored():
    with pytest.raises(ValueError, match="has no field 'lr'"):
        apply_overrides(ExperimentConfig(), {"training.optimizer.lr": 0.1})


def test_an_override_goes_through_the_configs_own_validation():
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        apply_overrides(ExperimentConfig(), {"training.optimizer.learning_rate": 0.0})


def test_an_overridden_config_still_survives_a_json_round_trip():
    config = apply_overrides(
        ExperimentConfig(),
        {
            "training.scheduler.name": "cosine",
            "training.loss.terms": [{"name": "weak_annual", "weight": 1.0}],
        },
    )

    assert ExperimentConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a.b=0.2", 0.2),
        ("a.b=3", 3),
        ("a.b=true", True),
        ("a.b=[1, 2]", [1, 2]),
        ("a.b=sgd", "sgd"),
        ("a.b=null", None),
    ],
)
def test_a_set_value_is_read_as_json_where_it_can_be(raw, expected):
    assert _parse_overrides([raw]) == {"a.b": expected}


def test_a_set_pair_without_a_value_is_refused():
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _parse_overrides(["training.max_epochs"])


def test_the_command_line_layers_on_top_of_the_config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"training": {"max_epochs": 3, "optimizer": {"name": "adam"}}}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["demo", "--config", str(path), "--optimizer", "sgd", "--scheduler", "cosine"]
    )

    config = _resolve_config(args)

    assert config.training.max_epochs == 3  # from the file
    assert config.training.optimizer.name == "sgd"  # the flag wins
    assert config.training.scheduler.name == "cosine"


def test_set_has_the_last_word_over_the_typed_flags():
    args = build_parser().parse_args(
        [
            "demo",
            "--learning-rate",
            "0.01",
            "--set",
            "training.optimizer.learning_rate=0.002",
        ]
    )

    assert _resolve_config(args).training.optimizer.learning_rate == pytest.approx(0.002)


def test_the_optimizer_section_is_the_canonical_place_for_the_rate():
    """The shorthand seeds the section; the section wins when it speaks."""
    seeded = TrainingConfig(learning_rate=5e-3)
    assert seeded.optimizer.learning_rate == pytest.approx(5e-3)

    stated = TrainingConfig(
        learning_rate=5e-3, optimizer=OptimizerConfig(learning_rate=2e-4)
    )
    assert stated.optimizer.learning_rate == pytest.approx(2e-4)
    # Whichever won, reading either spelling reports what the run will use.
    assert stated.learning_rate == pytest.approx(2e-4)


def test_an_unknown_name_is_refused_before_anything_is_built():
    args = build_parser().parse_args(["demo", "--set", "training.scheduler.name=triangular"])

    with pytest.raises(SystemExit, match="unknown scheduler"):
        _resolve_config(args)


def test_an_unknown_loss_term_is_refused_before_anything_is_built():
    args = build_parser().parse_args(
        ["demo", "--set", 'training.loss.terms=[{"name": "entropy"}]']
    )

    with pytest.raises(SystemExit, match="unknown loss term"):
        _resolve_config(args)


def test_a_grid_expands_to_the_product_of_its_axes(tmp_path):
    path = _sweep_file(
        tmp_path,
        grid={
            "training.optimizer.learning_rate": [1e-3, 3e-4],
            "training.scheduler.name": ["none", "cosine"],
        },
        seeds=[0, 1],
    )

    points = expand_sweep(load_sweep(path))

    assert len(points) == 2 * 2 * 2
    assert {point.config.training.scheduler.name for point in points} == {"none", "cosine"}
    assert {point.seed for point in points} == {0, 1}


def test_every_point_carries_the_settings_that_made_it(tmp_path):
    path = _sweep_file(
        tmp_path,
        base={"training": {"max_epochs": 4}},
        grid={"training.loss.smoothness_weight": [0.0, 0.5]},
    )

    points = expand_sweep(load_sweep(path))

    assert [point.config.training.loss.smoothness_weight for point in points] == [0.0, 0.5]
    assert all(point.config.training.max_epochs == 4 for point in points)
    assert points[1].label == "loss.smoothness_weight=0.5"


def test_a_random_sweep_is_reproducible(tmp_path):
    payload = dict(
        grid={"training.optimizer.learning_rate": [1e-2, 1e-3, 1e-4]},
        strategy="random",
        n_samples=5,
        seed=7,
    )
    first = expand_sweep(load_sweep(_sweep_file(tmp_path, **payload)))
    second = expand_sweep(load_sweep(_sweep_file(tmp_path, **payload)))

    assert [point.overrides for point in first] == [point.overrides for point in second]
    assert len(first) == 5


def test_a_cap_applies_after_expansion(tmp_path):
    path = _sweep_file(
        tmp_path,
        grid={"training.max_epochs": [1, 2, 3, 4, 5, 6]},
        max_runs=2,
    )

    assert len(expand_sweep(load_sweep(path))) == 2


def test_an_impossible_point_is_found_before_the_first_fit(tmp_path):
    path = _sweep_file(
        tmp_path, grid={"training.optimizer.learning_rate": [1e-3, -1.0]}
    )

    with pytest.raises(ValueError, match="sweep point 1"):
        expand_sweep(load_sweep(path))


def test_an_unknown_key_in_the_sweep_file_is_reported(tmp_path):
    path = _sweep_file(tmp_path, grid={}, epochs=10)

    with pytest.raises(ValueError, match="unknown keys in the sweep file"):
        load_sweep(path)


def test_a_sweep_runs_every_point_end_to_end(tmp_path):
    path = _sweep_file(
        tmp_path,
        command="demo",
        argv=[
            "--models",
            "stgnn",
            "--nodes-per-side",
            "4",
            "--n-time",
            "6",
            "--periods-per-year",
            "3",
            "--no-tracking",
            "--accelerator",
            "cpu",
        ],
        base={"training": {"max_epochs": 1}},
        grid={"training.scheduler.name": ["none", "cosine"]},
    )
    args = build_parser().parse_args(["sweep", str(path)])

    results = run_sweep(args)

    assert len(results) == 2
    assert all("error" not in entry for entry in results)
    assert all("test/annual_f1" in entry["metrics"]["stgnn"] for entry in results)
    assert [entry["overrides"] for entry in results] == [
        {"training.scheduler.name": "none"},
        {"training.scheduler.name": "cosine"},
    ]


def test_a_dry_run_validates_without_fitting(tmp_path):
    path = _sweep_file(tmp_path, grid={"training.max_epochs": [1, 2]})
    args = build_parser().parse_args(["sweep", str(path), "--dry-run"])

    assert run_sweep(args) == []

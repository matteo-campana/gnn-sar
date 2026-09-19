"""MLflow tracking: where a run is written, and what ends up in it."""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from gnn_sar.config import ExperimentConfig, TrackingConfig
from gnn_sar.training.cli import build_parser, run_demo
from gnn_sar.training.tracking import (
    DEFAULT_LOCAL_STORE,
    RunTracker,
    describe_destination,
    flatten_params,
    open_run,
    resolve_tracking_uri,
    tracked_run,
)


@pytest.fixture
def store(tmp_path, monkeypatch) -> TrackingConfig:
    """Tracking pointed at a file store of this test's own."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return TrackingConfig(
        experiment_name="test", tracking_uri=str(tmp_path / "mlruns")
    )


def _client(config: TrackingConfig) -> MlflowClient:
    return MlflowClient(tracking_uri=resolve_tracking_uri(config.tracking_uri))


def _only_run(config: TrackingConfig):
    client = _client(config)
    experiment = client.get_experiment_by_name(config.experiment_name)
    assert experiment is not None, "the experiment was never created"
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    return client, runs[0]


# --------------------------------------------------------------------- the URI


def test_an_explicit_uri_wins_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://ignored:5000")

    assert resolve_tracking_uri(str(tmp_path)) == tmp_path.resolve().as_uri()


def test_the_environment_is_the_second_choice(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.local:5000")

    assert resolve_tracking_uri() == "http://tracking.local:5000"


def test_the_last_resort_is_a_local_store_next_to_the_working_directory(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    uri = resolve_tracking_uri()

    assert uri == Path(DEFAULT_LOCAL_STORE).resolve().as_uri()
    assert uri.startswith("file:")


def test_a_relative_path_is_made_absolute(monkeypatch):
    """Otherwise the store would follow the working directory of the caller."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert resolve_tracking_uri("runs/here") == Path("runs/here").resolve().as_uri()


def test_a_server_uri_is_passed_through_untouched(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert resolve_tracking_uri("sqlite:///mlflow.db") == "sqlite:///mlflow.db"
    assert resolve_tracking_uri("databricks://x") == "databricks://x"


# ------------------------------------------------------------------ parameters


def test_a_nested_config_flattens_to_dotted_parameter_names():
    flat = flatten_params(ExperimentConfig())

    assert "training.loss.positive_weight" in flat
    assert "graph.n_neighbors" in flat
    assert not any(isinstance(value, dict) for value in flat.values())


def test_flattening_accepts_a_plain_mapping_too():
    assert flatten_params({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


# ----------------------------------------------------------- the disabled form


def test_a_disabled_tracker_accepts_every_call_and_records_nothing(tmp_path):
    tracker = open_run(TrackingConfig(enabled=False))

    tracker.log_params(ExperimentConfig())
    tracker.log_metrics({"test/annual_f1": 1.0})
    tracker.log_artifact(tmp_path)
    tracker.log_json({"a": 1}, "a.json")
    tracker.set_tags({"model": "stgnn"})
    tracker.finish()

    assert not tracker.enabled
    assert tracker.run_id is None
    assert list(tmp_path.iterdir()) == []


def test_a_disabled_tracker_hands_the_trainer_false_not_none():
    """``None`` would make Lightning fall back to its own default logger."""
    assert RunTracker(None).trainer_logger is False


def test_describe_destination_says_when_tracking_is_off():
    assert describe_destination(TrackingConfig(enabled=False)) == "tracking disabled"


# ------------------------------------------------------------------- a real run


def test_a_run_records_parameters_metrics_and_artefacts(store, tmp_path):
    artefact = tmp_path / "metrics.json"
    artefact.write_text('{"f1": 1.0}', encoding="utf-8")

    with tracked_run(store, run_name="stgnn", tags={"model": "stgnn"}) as tracker:
        tracker.log_params(ExperimentConfig())
        tracker.log_metrics({"test/annual_f1": 0.75})
        tracker.log_artifact(artefact)
        run_id = tracker.run_id

    client, run = _only_run(store)
    assert run.info.run_id == run_id
    assert run.info.run_name == "stgnn"
    assert run.info.status == "FINISHED"
    assert run.data.tags["model"] == "stgnn"
    assert run.data.params["training.loss.positive_weight"] == "1.0"
    assert run.data.metrics["test/annual_f1"] == pytest.approx(0.75)
    assert [item.path for item in client.list_artifacts(run_id)] == ["metrics.json"]


def test_metrics_that_are_not_numbers_are_skipped(store):
    """Delay metrics are NaN whenever a split holds no change at all."""
    with tracked_run(store) as tracker:
        tracker.log_metrics(
            {"good": 1.0, "nan": float("nan"), "text": "n/a", "none": None}
        )

    _, run = _only_run(store)
    assert set(run.data.metrics) == {"good"}


def test_a_failed_run_is_marked_failed_rather_than_left_running(store):
    with pytest.raises(RuntimeError, match="boom"):
        with tracked_run(store):
            raise RuntimeError("boom")

    _, run = _only_run(store)
    assert run.info.status == "FAILED"


def test_a_json_payload_can_be_attached_without_a_file_on_disk(store):
    with tracked_run(store) as tracker:
        tracker.log_json({"n_nodes": 64}, "area.json", artifact_path="inputs")

    client, run = _only_run(store)
    assert [item.path for item in client.list_artifacts(run.info.run_id, "inputs")] == [
        "inputs/area.json"
    ]


def test_a_missing_artefact_is_ignored_rather_than_crashing_the_run(store, tmp_path):
    """A run must not be lost because an optional output was never produced."""
    with tracked_run(store) as tracker:
        tracker.log_artifact(tmp_path / "never-written.tif")

    client, run = _only_run(store)
    assert client.list_artifacts(run.info.run_id) == []
    assert run.info.status == "FINISHED"


# ------------------------------------------------------------ through the CLI


def test_the_demo_records_one_run_per_compared_model(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    args = build_parser().parse_args(
        [
            "demo",
            "--models",
            "temporal",
            "rf",
            "--max-epochs",
            "2",
            "--nodes-per-side",
            "8",
            "--n-time",
            "8",
            "--periods-per-year",
            "4",
            "--accelerator",
            "cpu",
            "--experiment-name",
            "ablation",
            "--tracking-uri",
            str(tmp_path / "mlruns"),
        ]
    )

    run_demo(args)

    config = TrackingConfig(
        experiment_name="ablation", tracking_uri=str(tmp_path / "mlruns")
    )
    client = _client(config)
    experiment = client.get_experiment_by_name("ablation")
    runs = {run.info.run_name: run for run in client.search_runs([experiment.experiment_id])}

    assert set(runs) == {"temporal", "rf"}
    for name, run in runs.items():
        assert run.data.tags["model"] == name
        assert run.data.tags["command"] == "demo"
        assert run.info.status == "FINISHED"
        # The annual metric is the one every model, the forest included, reports.
        assert "test/annual_accuracy" in run.data.metrics
        assert run.data.params["scenario.n_time"] == "8"
        assert [item.path for item in client.list_artifacts(run.info.run_id)] == [
            "metrics.json"
        ]


def test_no_tracking_leaves_no_store_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    args = build_parser().parse_args(
        [
            "demo",
            "--models",
            "mlp",
            "--max-epochs",
            "2",
            "--nodes-per-side",
            "8",
            "--n-time",
            "8",
            "--periods-per-year",
            "4",
            "--accelerator",
            "cpu",
            "--no-tracking",
        ]
    )

    results = run_demo(args)

    assert "test/annual_accuracy" in results["mlp"]
    assert list(tmp_path.iterdir()) == []


def test_the_command_line_overrides_are_the_parameters_that_get_recorded(
    tmp_path, monkeypatch
):
    """A flag must not quietly differ from the value the run says it used."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    args = build_parser().parse_args(
        [
            "demo",
            "--models",
            "mlp",
            "--max-epochs",
            "3",
            "--nodes-per-side",
            "8",
            "--n-time",
            "8",
            "--periods-per-year",
            "4",
            "--accelerator",
            "cpu",
            "--precision",
            "bf16-mixed",
            "--experiment-name",
            "overrides",
            "--tracking-uri",
            str(tmp_path / "mlruns"),
        ]
    )

    run_demo(args)

    config = TrackingConfig(
        experiment_name="overrides", tracking_uri=str(tmp_path / "mlruns")
    )
    _, run = _only_run(config)
    assert run.data.params["training.max_epochs"] == "3"
    assert run.data.params["training.accelerator"] == "cpu"
    assert run.data.params["training.precision"] == "bf16-mixed"
    # and the epochs actually ran, rather than only being advertised
    assert len(_client(config).get_metric_history(run.info.run_id, "train/total")) == 3

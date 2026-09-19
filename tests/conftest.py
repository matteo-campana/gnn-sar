"""Shared fixtures: one small synthetic study area, built once per session."""

from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.synthetic import SyntheticLandTakeScenario, make_synthetic_scenario


@pytest.fixture(autouse=True)
def _isolated_mlflow_store(tmp_path, monkeypatch):
    """Keep every test's MLflow runs out of the working directory.

    Tracking is on by default, so without this a test that trains would create a
    real ``./mlruns`` inside the checkout. Tests that care about what was logged
    point at their own store explicitly and override this.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())


@pytest.fixture(scope="session")
def scenario() -> SyntheticLandTakeScenario:
    return make_synthetic_scenario(
        height=32, width=32, block=4, n_time=12, periods_per_year=6, seed=0
    )


@pytest.fixture
def label_image() -> np.ndarray:
    """A 3x3 raster with four segments, two of which do not touch.

        0 0 1
        0 0 1
        2 2 3
    """
    return np.array([[0, 0, 1], [0, 0, 1], [2, 2, 3]])

"""The declarative ingestion, end to end, on GeoTIFFs written by the test."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gnn_sar.data.ingest import IngestConfig, run_ingest
from gnn_sar.data.raster import RasterGrid
from gnn_sar.data.study_area import load_study_area
from gnn_sar.data.targets import IGNORE_INDEX

pytest.importorskip(
    "rasterio", reason="the geo extra is not installed (poetry install --extras geo)"
)
from gnn_sar.data.raster import write_raster  # noqa: E402  (after the skip)

SIZE = 16
GRID = RasterGrid(SIZE, SIZE, (300000.0, 10.0, 0.0, 4600000.0, 0.0, -10.0), "EPSG:32633")
SEALED = (slice(0, 4), slice(0, 4))  # the corner that gets consumed, i.e. node 0


@pytest.fixture
def area_root(tmp_path):
    """A two-year study area: monthly Sentinel-1, one corner sealed in 2023."""
    rng = np.random.default_rng(0)
    (tmp_path / "s1").mkdir()
    (tmp_path / "labels").mkdir()

    for year in (2022, 2023):
        for month in range(1, 13):
            sealed_now = year == 2023 and month >= 6
            vv = rng.gamma(8, 0.03, size=(SIZE, SIZE))
            vh = rng.gamma(8, 0.008, size=(SIZE, SIZE))
            if sealed_now:
                vv[SEALED] *= 6.0
                vh[SEALED] *= 4.0
            name = (
                f"S1A_IW_GRDH_1SDV_{year}{month:02d}12T051253_"
                f"{year}{month:02d}12T051318_030631_038276_9A2C.tif"
            )
            write_raster(tmp_path / "s1" / name, np.stack([vv, vh]), GRID, nodata=None)

    for year, sealed in ((2022, False), (2023, True)):
        mask = np.zeros((SIZE, SIZE))
        if sealed:
            mask[SEALED] = 1
        write_raster(tmp_path / "labels" / f"consumo_{year}.tif", mask, GRID, nodata=None)

    roads = np.zeros((SIZE, SIZE))
    roads[6:9, :] = 1
    write_raster(tmp_path / "roads.tif", roads, GRID, nodata=None)
    return tmp_path


def _config(area_root, **overrides):
    payload = {
        "sources": [
            {
                "name": "s1",
                "kind": "sentinel1",
                "directory": "s1",
                "bands": {"VV": 1, "VH": 2},
            }
        ],
        "segmentation": {"method": "grid", "block": 4},
        "period": "month",
        "labels": {
            "rasters": {
                "2022": "labels/consumo_2022.tif",
                "2023": "labels/consumo_2023.tif",
            }
        },
        "corridor": {"raster": "roads.tif"},
        "output": "study_area.npz",
        **overrides,
    }
    path = area_root / "ingest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return IngestConfig.from_json(path)


def test_the_config_resolves_its_paths_against_its_own_folder(area_root):
    config = _config(area_root)

    assert config.resolve("s1") == area_root / "s1"
    assert config.sources[0].band_map == {"VV": 1, "VH": 2}


def test_an_unknown_key_in_the_config_is_named(area_root):
    with pytest.raises(ValueError, match="unknown keys for SourceConfig"):
        IngestConfig.from_dict({"sources": [{"name": "s1", "polarisation": "VV"}]})


def test_a_comment_key_is_ignored_so_a_config_can_be_annotated():
    config = IngestConfig.from_dict(
        {"//": "area pilota", "sources": [{"name": "s1", "//": "orbita 22"}]}
    )

    assert config.sources[0].name == "s1"


def test_the_shipped_example_config_is_valid():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "docs/esempi/ingest_viterbo.json"

    config = IngestConfig.from_json(example)

    assert [source.name for source in config.sources] == ["s1", "s2"]
    assert config.labels is not None and len(config.labels.rasters) == 4


def test_the_ingestion_produces_a_graph_the_datamodule_can_train_on(area_root):
    area = run_ingest(_config(area_root), verbose=False)

    assert area.graph.n_time == 24
    assert area.graph.n_nodes == 16
    assert area.channel_names == [
        "VV_db_mean",
        "VV_db_std",
        "VH_db_mean",
        "VH_db_std",
        "VVVH_ratio_db_mean",
        "VVVH_ratio_db_std",
    ]
    assert area.periods.labels()[0] == "2022-01"
    assert area.graph.year_index.tolist() == [0] * 12 + [1] * 12
    assert np.isfinite(area.graph.features).all()


def test_the_cartography_becomes_a_land_take_label_in_the_year_of_the_transition(area_root):
    area = run_ingest(_config(area_root), verbose=False)

    labels = area.graph.annual_labels
    assert labels.shape == (2, 16)
    assert labels[0, 0] == 0 and labels[1, 0] == 1  # node 0 consumed during 2023
    assert (labels[:, 1:] == 0).all()  # everything else never is


def test_the_corridor_marks_every_node_the_road_touches(area_root):
    area = run_ingest(_config(area_root), verbose=False)

    assert area.corridor is not None
    # The road spans pixel rows 6..8, which straddle two rows of 4x4 nodes.
    assert int(area.corridor.sum()) == 8


def test_a_year_without_cartography_stays_unknown(area_root):
    config = _config(
        area_root, labels={"rasters": {"2023": "labels/consumo_2023.tif"}}
    )

    area = run_ingest(config, verbose=False)

    # One state raster cannot say whether the corner was consumed in 2023 or
    # before, so the node is excluded rather than called a positive.
    assert (area.graph.annual_labels[0] == IGNORE_INDEX).all()
    assert area.graph.annual_labels[1, 0] == IGNORE_INDEX


def test_a_raster_off_the_study_grid_is_refused(area_root):
    shifted = RasterGrid(
        SIZE, SIZE, (400000.0, 10.0, 0.0, 4600000.0, 0.0, -10.0), "EPSG:32633"
    )
    write_raster(area_root / "roads.tif", np.ones((SIZE, SIZE)), shifted, nodata=None)

    with pytest.raises(ValueError, match="corridor raster.*not on the study grid"):
        run_ingest(_config(area_root), verbose=False)


def test_a_source_directory_without_matching_files_says_so(area_root):
    with pytest.raises(FileNotFoundError, match="no file matches"):
        run_ingest(
            _config(
                area_root,
                sources=[{"name": "s1", "directory": "s1", "pattern": "*.jp2"}],
            ),
            verbose=False,
        )


def test_the_ingested_area_can_be_cached_and_reloaded(area_root):
    config = _config(area_root)
    area = run_ingest(config, verbose=False)

    restored = load_study_area(area.save(config.resolve(config.output)))

    assert restored.grid == GRID
    assert restored.year_labels == [2022, 2023]
    assert restored.metadata["config"]["period"] == "month"
    assert len(restored.metadata["acquisitions"]["s1"]) == 24


def test_nodes_can_come_from_slic_over_a_multitemporal_composite(area_root):
    pytest.importorskip("skimage", reason="needs the segmentation extra")

    area = run_ingest(
        _config(
            area_root,
            segmentation={"method": "slic", "n_segments": 16, "composites": 4},
        ),
        verbose=False,
    )

    # SLIC decides the exact count; what matters is that the partition covers
    # the raster and stays fixed for the whole series.
    assert area.geometries.n_nodes > 1
    assert area.geometries.shape == (SIZE, SIZE)
    assert area.graph.features.shape[:2] == (24, area.geometries.n_nodes)

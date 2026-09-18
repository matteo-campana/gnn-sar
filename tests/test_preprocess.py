"""The local ingestor. SNAP itself is never invoked here — only what surrounds it."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from gnn_sar.data.preprocess import (
    PreprocessReport,
    SnapNotFound,
    coherence_pairs,
    declared_parameters,
    find_gpt,
    gpt_command,
)
from gnn_sar.data.sites import StudySite

rasterio = pytest.importorskip(
    "rasterio", reason="the geo extra is not installed (poetry install --extras geo)"
)

from gnn_sar.data.preprocess import (  # noqa: E402  (after the skip)
    align_to_site,
    find_sentinel2_bands,
    preprocess_sentinel2,
    stack_sentinel2_product,
)
from gnn_sar.data.raster import read_raster, write_raster  # noqa: E402

SITE = StudySite(
    name="prova",
    description="a small test area",
    bbox=(12.00, 42.00, 12.02, 42.02),
    crs="EPSG:32633",
    extent=(300000.0, 4650000.0, 300400.0, 4650300.0),  # 40 x 30 px at 10 m
)


# --- talking to SNAP ---------------------------------------------------------


def test_the_gpt_command_carries_the_graph_memory_and_parameters():
    command = gpt_command(
        "graph.xml", {"input": "a.zip", "output": "b.tif"}, gpt="/opt/snap/bin/gpt"
    )

    assert command[:2] == ["/opt/snap/bin/gpt", "graph.xml"]
    assert "-Pinput=a.zip" in command and "-Poutput=b.tif" in command
    assert command[command.index("-c") + 1] == "8G"


def test_an_explicit_gpt_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(SnapNotFound, match="not an executable file"):
        find_gpt(tmp_path / "nowhere" / "gpt")


def test_gpt_is_taken_from_the_environment_when_present(tmp_path, monkeypatch):
    fake = tmp_path / "gpt"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("GNN_SAR_GPT", str(fake))

    assert find_gpt() == fake


def test_a_missing_snap_explains_where_to_get_it(monkeypatch):
    monkeypatch.delenv("GNN_SAR_GPT", raising=False)
    monkeypatch.delenv("GPT_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("gnn_sar.data.preprocess._GPT_CANDIDATES", ())

    with pytest.raises(SnapNotFound, match="step.esa.int"):
        find_gpt()


def test_only_the_variables_a_graph_declares_are_passed_to_it(tmp_path):
    graph = tmp_path / "graph.xml"
    graph.write_text("<graph><file>${input}</file><crs>${crs}</crs></graph>")

    kept = declared_parameters(graph, {"crs": "EPSG:32633", "subswath": "IW2"})

    assert kept == {"crs": "EPSG:32633"}


# --- putting a raster on the study grid --------------------------------------


def test_alignment_lands_on_the_site_grid_exactly(tmp_path):
    # A source with a different extent and a coarser pixel than the site.
    source_grid = SITE.grid()
    from gnn_sar.data.raster import RasterGrid

    other = RasterGrid(20, 30, (299800.0, 20.0, 0.0, 4650400.0, 0.0, -20.0), "EPSG:32633")
    write_raster(tmp_path / "in.tif", np.full((1, 20, 30), 7.0), other, nodata=None)

    align_to_site(tmp_path / "in.tif", tmp_path / "out.tif", SITE)

    values, grid = read_raster(tmp_path / "out.tif")
    assert grid.matches(source_grid)
    assert values.shape == (1, *SITE.shape)
    assert np.nanmax(values) == pytest.approx(7.0)


def test_alignment_keeps_the_band_order_asked_for(tmp_path):
    grid = SITE.grid()
    values = np.stack(
        [np.full(SITE.shape, 1.0), np.full(SITE.shape, 2.0), np.full(SITE.shape, 3.0)]
    )
    write_raster(tmp_path / "in.tif", values, grid, nodata=None)

    align_to_site(tmp_path / "in.tif", tmp_path / "out.tif", SITE, bands=[3, 1])

    out, _ = read_raster(tmp_path / "out.tif")
    assert out[0, 0, 0] == pytest.approx(3.0)
    assert out[1, 0, 0] == pytest.approx(1.0)


# --- Sentinel-2 --------------------------------------------------------------


def _safe_product(root: Path, name: str, resolutions=("10m", "20m")) -> Path:
    """A minimal .SAFE-like tree: one file per band and resolution."""
    product = root / name
    grid = SITE.grid()
    for resolution in resolutions:
        folder = product / "GRANULE" / "L2A_T33TTG" / "IMG_DATA" / f"R{resolution}"
        folder.mkdir(parents=True)
        bands = (
            ("B02", "B03", "B04", "B08")
            if resolution == "10m"
            else ("B11", "B12", "SCL")
        )
        for position, band in enumerate(bands):
            values = np.full((1, *SITE.shape), float(position + 1))
            if band == "SCL":
                values = np.full((1, *SITE.shape), 4.0)
                values[0, :5, :] = 9.0  # a band of cloud
            write_raster(
                folder / f"T33TTG_20240117T100319_{band}_{resolution}.tif",
                values,
                grid,
                nodata=None,
            )
    return product


def test_the_bands_are_found_inside_a_safe_tree(tmp_path):
    product = _safe_product(tmp_path, "S2B_MSIL2A_20240117T100319.SAFE")

    found = find_sentinel2_bands(product)

    assert set(found) == {"B02", "B03", "B04", "B08", "B11", "B12", "SCL"}
    assert found["B02"].name.endswith("_B02_10m.tif")
    assert found["SCL"].name.endswith("_SCL_20m.tif")


def test_the_finest_resolution_of_a_band_wins(tmp_path):
    product = _safe_product(tmp_path, "S2B.SAFE")
    coarse = product / "GRANULE" / "L2A_T33TTG" / "IMG_DATA" / "R60m"
    coarse.mkdir(parents=True)
    write_raster(
        coarse / "T33TTG_20240117T100319_B02_60m.tif",
        np.zeros((1, *SITE.shape)),
        SITE.grid(),
        nodata=None,
    )

    assert find_sentinel2_bands(product)["B02"].name.endswith("_B02_10m.tif")


def test_a_missing_band_is_named_rather_than_skipped(tmp_path):
    product = _safe_product(tmp_path, "S2B.SAFE", resolutions=("10m",))

    with pytest.raises(FileNotFoundError, match=r"\['B11', 'B12'\]"):
        find_sentinel2_bands(product)


def test_an_archive_is_not_read_in_place(tmp_path):
    archive = tmp_path / "S2B_MSIL2A_20240117T100319.zip"
    archive.write_bytes(b"not unzipped")

    with pytest.raises(FileNotFoundError, match="unzip the .SAFE product first"):
        find_sentinel2_bands(archive)


def test_a_stacked_product_keeps_the_declared_band_order(tmp_path):
    product = _safe_product(tmp_path, "S2B_MSIL2A_20240117T100319.SAFE")

    stack_sentinel2_product(product, tmp_path / "out.tif", SITE, bands=("B04", "B08"))

    values, grid = read_raster(tmp_path / "out.tif")
    assert values.shape == (3, *SITE.shape)  # B04, B08, SCL
    assert grid.matches(SITE.grid())
    assert values[0, 0, 0] == pytest.approx(3.0)  # B04 is the third 10 m band
    assert values[1, 0, 0] == pytest.approx(4.0)  # B08 the fourth


def test_the_scene_classification_is_not_interpolated(tmp_path):
    product = _safe_product(tmp_path, "S2B.SAFE")

    stack_sentinel2_product(product, tmp_path / "out.tif", SITE)

    scl = read_raster(tmp_path / "out.tif")[0][-1]
    # Nearest-neighbour resampling: only the classes that existed survive.
    assert set(np.unique(scl)) <= {4.0, 9.0}


def test_a_batch_skips_what_is_already_there_and_records_what_failed(tmp_path):
    good = _safe_product(tmp_path / "in", "S2B_MSIL2A_20240117T100319.SAFE")
    broken = tmp_path / "in" / "S2A_MSIL2A_20240210T100319.SAFE"
    broken.mkdir()

    quiet = {"log": lambda *_: None}
    first = preprocess_sentinel2([good, broken], tmp_path / "out", SITE, **quiet)
    second = preprocess_sentinel2([good], tmp_path / "out", SITE, **quiet)

    assert [path.name for path in first.written] == [f"{good.stem}.tif"]
    assert len(first.failed) == 1 and first.failed[0][0] == broken
    assert not first.ok
    assert second.skipped and not second.written


def test_the_report_summarises_a_batch():
    report = PreprocessReport(
        written=[Path("a.tif")], skipped=[Path("b.tif")], failed=[(Path("c.zip"), "boom")]
    )

    rendered = report.describe()

    assert "written  1" in rendered and "skipped  1" in rendered
    assert "c.zip: boom" in rendered


# --- coherence pairs ---------------------------------------------------------


def _dates(*days: int) -> list[datetime]:
    return [datetime(2024, 1, 1) + timedelta(days=day) for day in days]


def test_pairs_are_formed_at_the_requested_baseline():
    dates = _dates(0, 12, 24, 36)

    assert coherence_pairs(dates, baseline_days=12, tolerance_days=0) == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_a_baseline_the_series_does_not_offer_produces_nothing():
    # Mixing baselines would put a decorrelation signal in the series, so a
    # 6-day request over a 12-day series yields no pair rather than a fallback.
    assert coherence_pairs(_dates(0, 12, 24), baseline_days=6, tolerance_days=1) == []


def test_the_tolerance_admits_a_slightly_irregular_revisit():
    dates = _dates(0, 13)

    assert coherence_pairs(dates, baseline_days=12, tolerance_days=2) == [(0, 1)]
    assert coherence_pairs(dates, baseline_days=12, tolerance_days=0) == []


def test_pairs_are_indices_into_the_dates_in_chronological_order():
    dates = [datetime(2024, 1, 13), datetime(2024, 1, 1)]  # deliberately unsorted

    assert coherence_pairs(dates, baseline_days=12, tolerance_days=0) == [(1, 0)]


def test_a_date_can_be_allowed_more_than_one_partner():
    dates = _dates(0, 11, 12, 13)

    pairs = coherence_pairs(dates, baseline_days=12, tolerance_days=1, max_per_date=2)

    assert (0, 2) in pairs and len([p for p in pairs if p[0] == 0]) == 2


def test_an_impossible_baseline_is_refused():
    with pytest.raises(ValueError, match="baseline_days must be >= 1"):
        coherence_pairs(_dates(0, 12), baseline_days=0)

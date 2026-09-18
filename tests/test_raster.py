from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.raster import (
    RasterGrid,
    list_rasters,
    read_aligned_stack,
    read_raster,
    write_raster,
)

rasterio = pytest.importorskip(
    "rasterio", reason="the geo extra is not installed (poetry install --extras geo)"
)

GRID = RasterGrid(4, 6, (300000.0, 10.0, 0.0, 4600000.0, 0.0, -10.0), "EPSG:32633")


def _write(path, values, grid=GRID, **kwargs):
    return write_raster(path, np.asarray(values, dtype=float), grid, **kwargs)


def test_the_grid_reports_pixel_size_and_origin_from_the_gdal_transform():
    assert GRID.pixel_size == (10.0, 10.0)
    assert GRID.origin == (300000.0, 4600000.0)
    assert GRID.shape == (4, 6)


def test_two_grids_match_only_when_they_describe_the_same_pixels():
    shifted = RasterGrid(4, 6, (300010.0, 10.0, 0.0, 4600000.0, 0.0, -10.0), "EPSG:32633")
    reprojected = RasterGrid(4, 6, GRID.transform, "EPSG:4326")

    assert GRID.matches(GRID)
    assert not GRID.matches(shifted)
    assert not GRID.matches(reprojected)


def test_a_raster_survives_a_round_trip(tmp_path):
    values = np.arange(24, dtype=float).reshape(1, 4, 6)

    path = _write(tmp_path / "one.tif", values, nodata=None)
    read, grid = read_raster(path)

    assert read == pytest.approx(values)
    assert grid.matches(GRID)


def test_nodata_comes_back_as_nan_rather_than_as_a_value(tmp_path):
    values = np.full((1, 4, 6), 5.0)
    values[0, 0, 0] = -9999.0

    path = _write(tmp_path / "nodata.tif", values, nodata=-9999.0)
    read, _ = read_raster(path)

    assert np.isnan(read[0, 0, 0])
    assert read[0, 1, 1] == pytest.approx(5.0)


def test_bands_are_read_in_the_order_asked_for(tmp_path):
    values = np.stack([np.full((4, 6), 1.0), np.full((4, 6), 2.0)])

    path = _write(tmp_path / "two.tif", values, nodata=None)
    read, _ = read_raster(path, bands=[2, 1])

    assert read[0, 0, 0] == pytest.approx(2.0)
    assert read[1, 0, 0] == pytest.approx(1.0)


def test_asking_for_a_band_that_does_not_exist_says_how_many_there_are(tmp_path):
    path = _write(tmp_path / "one.tif", np.zeros((1, 4, 6)), nodata=None)

    with pytest.raises(ValueError, match="has 1 bands"):
        read_raster(path, bands=[3])


def test_a_stack_on_one_grid_reads_as_one_array(tmp_path):
    paths = [
        _write(tmp_path / f"{i}.tif", np.full((1, 4, 6), float(i)), nodata=None)
        for i in range(3)
    ]

    values, grid = read_aligned_stack(paths)

    assert values.shape == (3, 1, 4, 6)
    assert grid.matches(GRID)


def test_a_raster_off_the_grid_is_refused_and_names_both_files(tmp_path):
    reference = _write(tmp_path / "reference.tif", np.zeros((1, 4, 6)), nodata=None)
    other = _write(
        tmp_path / "other.tif",
        np.zeros((1, 4, 6)),
        grid=RasterGrid(4, 6, (300000.0, 20.0, 0.0, 4600000.0, 0.0, -20.0), "EPSG:32633"),
        nodata=None,
    )

    with pytest.raises(ValueError, match="other.tif.*not on the same grid.*reference.tif"):
        read_aligned_stack([reference, other])


def test_writing_values_of_the_wrong_size_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not match the grid"):
        _write(tmp_path / "bad.tif", np.zeros((1, 3, 3)))


def test_rasters_are_listed_in_name_order(tmp_path):
    for name in ("b.tif", "a.tif", "c.txt"):
        _write(tmp_path / name, np.zeros((1, 4, 6)), nodata=None) if name.endswith(
            ".tif"
        ) else (tmp_path / name).write_text("")

    assert [path.name for path in list_rasters(tmp_path)] == ["a.tif", "b.tif"]

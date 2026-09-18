from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from gnn_sar.data.sentinel import (
    DEFAULT_INVALID_SCL,
    PeriodGrid,
    normalized_difference,
    scl_valid_mask,
    sensing_date,
    sentinel1_channels,
    sentinel2_indices,
)

S1_NAME = "S1A_IW_GRDH_1SDV_20240117T051253_20240117T051318_052112_064B7A_9A2C.SAFE"
S2_NAME = "S2B_MSIL2A_20240117T100319_N0510_R122_T33TTG_20240117T121500.SAFE"


def test_the_sensing_date_is_read_from_a_sentinel1_product_name():
    assert sensing_date(S1_NAME) == datetime(2024, 1, 17, 5, 12, 53)


def test_the_sensing_date_is_read_from_a_sentinel2_product_name():
    assert sensing_date(S2_NAME) == datetime(2024, 1, 17, 10, 3, 19)


def test_the_sensing_date_is_read_from_a_granule_file():
    assert sensing_date("T33TTG_20240117T100319_B02_10m.jp2") == datetime(
        2024, 1, 17, 10, 3, 19
    )


def test_a_plain_date_in_a_renamed_file_is_enough():
    assert sensing_date("/data/s1/viterbo_20240117_vv.tif") == datetime(2024, 1, 17)


def test_a_name_without_a_date_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="no sensing date"):
        sensing_date("scene_final_v2.tif")


def test_monthly_composites_span_every_month_the_acquisitions_touch():
    grid = PeriodGrid.covering(["2023-11-04", "2024-01-30"], "month")

    assert grid.start == date(2023, 11, 1)
    assert grid.n_periods == 3
    assert grid.labels() == ["2023-11", "2023-12", "2024-01"]


def test_an_acquisition_is_mapped_to_the_composite_holding_it():
    grid = PeriodGrid.covering(["2023-11-04", "2024-01-30"], "month")

    assert grid.index_of(["2023-11-30", "2023-12-01", "2024-01-02"]).tolist() == [0, 1, 2]


def test_dekads_cut_every_month_into_three():
    grid = PeriodGrid.covering(["2023-01-02", "2023-02-25"], "dekad")

    assert grid.n_periods == 6
    assert grid.labels()[:4] == ["2023-01-D1", "2023-01-D2", "2023-01-D3", "2023-02-D1"]
    assert grid.index_of(["2023-01-11", "2023-01-21", "2023-01-31"]).tolist() == [1, 2, 2]


def test_a_period_can_also_be_a_number_of_days():
    grid = PeriodGrid.covering(["2023-01-01", "2023-01-25"], 12)

    assert grid.n_periods == 3
    assert grid.index_of(["2023-01-13"]).tolist() == [1]


def test_the_year_index_is_contiguous_and_names_its_calendar_years():
    grid = PeriodGrid.covering(["2022-11-01", "2023-02-01"], "month")

    assert grid.year_index().tolist() == [0, 0, 1, 1]
    assert grid.year_labels() == [2022, 2023]


def test_an_acquisition_outside_the_grid_is_an_error_not_a_silent_drop():
    grid = PeriodGrid(period="month", start=date(2023, 1, 1), n_periods=2)

    with pytest.raises(ValueError, match="outside the grid"):
        grid.index_of(["2023-03-01"])


def test_a_grid_cannot_start_after_its_own_acquisitions():
    with pytest.raises(ValueError, match="after the earliest acquisition"):
        PeriodGrid.covering(["2023-01-15"], "month", start="2023-02-01")


def test_backscatter_becomes_decibels_and_keeps_the_ratio():
    bands = {"VV": np.array([[1.0]]), "VH": np.array([[0.1]])}

    values, names = sentinel1_channels(bands)

    assert names == ["VV_db", "VH_db", "VVVH_ratio_db"]
    assert values[:, 0, 0] == pytest.approx([0.0, -10.0, 10.0], abs=1e-6)


def test_nodata_backscatter_does_not_become_a_backscatter_level():
    # A calibrated product marks nodata with 0 or a small negative; neither is
    # a physical level, so neither may survive the dB conversion as one.
    values, _ = sentinel1_channels({"VV": np.array([[0.0, -0.01, 1.0]])})

    assert np.isnan(values[0, 0, :2]).all()
    assert values[0, 0, 2] == pytest.approx(0.0, abs=1e-6)


def test_a_coherence_band_is_passed_through_untouched():
    coherence = np.array([[0.42]])

    values, names = sentinel1_channels({"VV": np.array([[1.0]]), "coherence": coherence})

    assert names == ["VV_db", "coherence"]
    assert values[1] == pytest.approx(coherence)


def test_sentinel1_needs_at_least_one_polarisation():
    with pytest.raises(ValueError, match="at least one polarisation"):
        sentinel1_channels({"coherence": np.zeros((2, 2))})


def test_the_normalised_difference_is_the_definition():
    assert normalized_difference(np.array([3.0]), np.array([1.0])) == pytest.approx([0.5])


def test_a_vanishing_sum_gives_nan_rather_than_an_exception():
    assert np.isnan(normalized_difference(np.array([1.0]), np.array([-1.0])))[0]


def test_cloud_classes_are_rejected_and_ground_classes_are_kept():
    scl = np.array([[4, 5, 6, 8, 9, 3]])

    valid = scl_valid_mask(scl)

    assert valid.tolist() == [[True, True, True, False, False, False]]
    assert set(DEFAULT_INVALID_SCL) >= {8, 9, 10}


def test_indices_are_masked_where_the_scene_classification_rejects_the_pixel():
    bands = {
        "B02": np.full((1, 2), 0.1),
        "B04": np.full((1, 2), 0.2),
        "B08": np.full((1, 2), 0.6),
        "B11": np.full((1, 2), 0.3),
    }
    valid = np.array([[True, False]])

    values, names = sentinel2_indices(bands, ("ndvi", "ndbi", "bsi"), valid=valid)

    assert names == ["s2_ndvi", "s2_ndbi", "s2_bsi"]
    assert values[0, 0, 0] == pytest.approx(0.5)  # (0.6 - 0.2) / 0.8
    assert np.isnan(values[:, 0, 1]).all()


def test_an_index_says_which_band_it_is_missing():
    with pytest.raises(ValueError, match="needs band B08"):
        sentinel2_indices({"B04": np.zeros((2, 2))}, ("ndvi",))

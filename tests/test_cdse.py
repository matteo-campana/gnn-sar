"""The query builders only: nothing here touches the network."""

from __future__ import annotations

from datetime import date

import pytest

from gnn_sar.data.cdse import (
    CdseCredentials,
    bbox_to_wkt,
    format_products,
    product_filter,
)


def test_a_bounding_box_becomes_a_closed_wkt_polygon():
    wkt = bbox_to_wkt(11.5, 42.2, 12.0, 42.6)

    assert wkt.startswith("POLYGON((") and wkt.endswith("))")
    points = wkt[len("POLYGON((") : -2].split(", ")
    assert len(points) == 5 and points[0] == points[-1]  # closed ring


def test_an_inverted_bounding_box_is_refused():
    with pytest.raises(ValueError, match="west < east"):
        bbox_to_wkt(12.0, 42.2, 11.5, 42.6)


def test_the_filter_pins_the_collection_and_the_sensing_interval():
    expression = product_filter("SENTINEL-1", date(2023, 1, 1), date(2024, 1, 1))

    assert "Collection/Name eq 'SENTINEL-1'" in expression
    assert "ContentDate/Start ge 2023-01-01T00:00:00.000Z" in expression
    assert "ContentDate/Start lt 2024-01-01T00:00:00.000Z" in expression


def test_a_product_type_is_matched_as_a_string_attribute():
    expression = product_filter(
        "SENTINEL-1", "2023-01-01", "2023-02-01", product_type="IW_GRDH_1S"
    )

    assert "att/Name eq 'productType'" in expression
    assert "att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S'" in expression


def test_a_relative_orbit_is_matched_as_an_integer_attribute():
    expression = product_filter(
        "SENTINEL-1", "2023-01-01", "2023-02-01", attributes={"relativeOrbitNumber": 22}
    )

    assert "OData.CSC.IntegerAttribute" in expression
    assert "att/OData.CSC.IntegerAttribute/Value eq 22" in expression


def test_cloud_cover_filters_on_a_double_attribute():
    expression = product_filter(
        "SENTINEL-2", "2023-01-01", "2023-02-01", max_cloud_cover=20
    )

    assert "att/Name eq 'cloudCover'" in expression
    assert "att/OData.CSC.DoubleAttribute/Value lt 20.0" in expression


def test_an_area_of_interest_becomes_an_intersects_clause():
    expression = product_filter(
        "SENTINEL-1", "2023-01-01", "2023-02-01", footprint_wkt=bbox_to_wkt(11, 42, 12, 43)
    )

    assert "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((" in expression


def test_missing_credentials_point_at_the_variables_to_set(monkeypatch):
    monkeypatch.delenv("CDSE_USERNAME", raising=False)
    monkeypatch.delenv("CDSE_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="CDSE_USERNAME and CDSE_PASSWORD"):
        CdseCredentials.from_env()


def test_credentials_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("CDSE_USERNAME", "someone@example.org")
    monkeypatch.setenv("CDSE_PASSWORD", "secret")

    assert CdseCredentials.from_env().username == "someone@example.org"


def test_products_are_listed_with_their_date_and_size():
    line = format_products(
        [
            {
                "Name": "S1A_IW_GRDH_1SDV_20240117T051253.SAFE",
                "ContentDate": {"Start": "2024-01-17T05:12:53.000Z"},
                "ContentLength": 1_500_000_000,
            }
        ]
    )

    assert "2024-01-17T05:12:53" in line
    assert "1.50 GB" in line
    assert "S1A_IW_GRDH" in line

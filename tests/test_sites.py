from __future__ import annotations

import pytest

from gnn_sar.data.sites import PILOT_SITES, StudySite, format_sites, get_site


def test_the_three_pilot_areas_are_registered():
    assert set(PILOT_SITES) == {"roma", "viterbo", "milano"}


@pytest.mark.parametrize("name", sorted(PILOT_SITES))
def test_a_pilot_area_defines_a_whole_number_of_pixels(name):
    site = PILOT_SITES[name]
    height, width = site.shape

    assert height > 0 and width > 0
    assert site.grid().shape == (height, width)


@pytest.mark.parametrize("name", sorted(PILOT_SITES))
def test_a_pilot_area_is_small_enough_to_be_one_full_batch_graph(name):
    height, width = PILOT_SITES[name].shape

    # A study area is one graph held whole in memory; ~10 Mpx is the budget.
    assert height * width < 10_000_000


def test_the_grid_is_north_up_and_starts_at_the_extent_corner():
    grid = get_site("roma").grid()
    xmin, _, _, ymax = get_site("roma").extent

    assert grid.origin == (xmin, ymax)
    assert grid.pixel_size == (10.0, 10.0)
    assert grid.transform[5] < 0  # north up


def test_the_lazio_and_lombardia_sites_sit_in_different_utm_zones():
    # Mixing them inside one study area is exactly what must not happen.
    assert get_site("roma").crs == get_site("viterbo").crs == "EPSG:32633"
    assert get_site("milano").crs == "EPSG:32632"


def test_the_gdalwarp_options_carry_crs_resolution_and_extent():
    options = get_site("milano").gdalwarp_options()

    assert "-t_srs EPSG:32632" in options
    assert "-tr 10 10" in options
    assert "-te 503900 5025100 527500 5047500" in options


def test_a_site_can_be_looked_up_case_insensitively():
    assert get_site("ROMA") is get_site("roma")


def test_an_unknown_site_lists_the_ones_that_exist():
    with pytest.raises(ValueError, match="milano, roma, viterbo"):
        get_site("firenze")


def test_the_bbox_becomes_a_catalogue_footprint():
    assert get_site("viterbo").bbox_wkt().startswith("POLYGON((11.95 42.35")


def test_an_extent_that_is_not_a_whole_number_of_pixels_is_refused():
    with pytest.raises(ValueError, match="whole number of 10.0 m pixels"):
        StudySite(
            name="storto",
            description="",
            bbox=(11.0, 42.0, 12.0, 43.0),
            crs="EPSG:32633",
            extent=(0.0, 0.0, 1005.0, 1000.0),
        )


def test_an_inverted_bbox_is_refused():
    with pytest.raises(ValueError, match="west, south, east, north"):
        StudySite(
            name="storto",
            description="",
            bbox=(12.0, 42.0, 11.0, 43.0),
            crs="EPSG:32633",
            extent=(0.0, 0.0, 1000.0, 1000.0),
        )


def test_every_site_describes_itself_for_the_cli():
    rendered = format_sites()

    for name in PILOT_SITES:
        assert name in rendered
    assert "gdalwarp" in rendered

from __future__ import annotations

import numpy as np
import pytest

from gnn_sar.data.segmentation import NodeGeometries
from gnn_sar.data.texture import glcm_texture, glcm_texture_stack


@pytest.fixture
def band_and_geometries() -> tuple[np.ndarray, NodeGeometries]:
    """Four 4x4 units: uniform, a gradient, noise, uniform again."""
    geometries = NodeGeometries.regular_grid(8, 8, block=4)
    band = np.zeros((8, 8))
    band[:4, :4] = 0.0
    band[:4, 4:] = np.indices((4, 4))[0] * 3.0
    band[4:, :4] = np.random.default_rng(0).normal(0.0, 5.0, (4, 4))
    band[4:, 4:] = 1.0
    return band, geometries


def test_a_uniform_unit_has_no_contrast_and_full_homogeneity(band_and_geometries):
    band, geometries = band_and_geometries

    texture = glcm_texture(band, geometries, levels=8)

    assert texture[0, 0] == pytest.approx(0.0)
    assert texture[0, 1] == pytest.approx(1.0)


def test_the_noisiest_unit_has_the_highest_contrast(band_and_geometries):
    band, geometries = band_and_geometries

    contrast = glcm_texture(band, geometries, levels=8)[:, 0]

    assert int(np.argmax(contrast)) == 2


def test_energy_is_maximal_for_a_uniform_unit(band_and_geometries):
    band, geometries = band_and_geometries

    energy = glcm_texture(band, geometries, levels=8, properties=("energy",))[:, 0]

    assert energy[0] == pytest.approx(1.0)
    assert energy[2] < energy[0]


def test_texture_ignores_pairs_that_cross_a_segment_border():
    """Two uniform units of different levels must both read as textureless."""
    geometries = NodeGeometries.from_label_image(
        np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    )
    band = np.zeros((4, 4))
    band[:, 2:] = 10.0  # a strong step exactly on the shared border

    texture = glcm_texture(band, geometries, levels=8)

    assert np.allclose(texture[:, 0], 0.0)
    assert np.allclose(texture[:, 1], 1.0)


def test_texture_is_invariant_to_a_shift_of_the_whole_band(band_and_geometries):
    band, geometries = band_and_geometries
    value_range = (float(band.min()), float(band.max()) + 7.0)

    original = glcm_texture(band, geometries, levels=8, value_range=value_range)
    shifted = glcm_texture(band + 7.0, geometries, levels=8, value_range=(
        value_range[0] + 7.0, value_range[1] + 7.0
    ))

    assert np.allclose(original, shifted)


def test_a_shared_range_keeps_two_dates_comparable(band_and_geometries):
    band, geometries = band_and_geometries
    stack = np.stack([band, band * 0.5])[:, None]

    texture = glcm_texture_stack(stack, geometries, bands=(0,), levels=8)

    assert texture.shape == (2, geometries.n_nodes, 2)
    # The second date is genuinely smoother, and a common range can say so.
    assert texture[1, 2, 0] < texture[0, 2, 0]


def test_identical_dates_give_identical_texture(band_and_geometries):
    band, geometries = band_and_geometries
    stack = np.stack([band, band])[:, None]

    texture = glcm_texture_stack(stack, geometries, bands=(0,), levels=8)

    assert np.allclose(texture[0], texture[1])


def test_texture_stack_can_cover_several_bands(band_and_geometries):
    band, geometries = band_and_geometries
    stack = np.stack([np.stack([band, band * 2.0])] * 3)

    texture = glcm_texture_stack(stack, geometries, bands=(0, 1), levels=8)

    assert texture.shape == (3, geometries.n_nodes, 4)


def test_a_constant_band_does_not_divide_by_zero():
    geometries = NodeGeometries.regular_grid(4, 4, block=2)

    texture = glcm_texture(np.full((4, 4), 3.0), geometries, levels=8)

    assert np.all(np.isfinite(texture))
    assert np.allclose(texture[:, 0], 0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"levels": 1}, "levels must be >= 2"),
        ({"properties": ("entropy",)}, "unknown properties"),
    ],
)
def test_invalid_settings_are_rejected(band_and_geometries, kwargs, message):
    band, geometries = band_and_geometries

    with pytest.raises(ValueError, match=message):
        glcm_texture(band, geometries, **kwargs)


def test_texture_rejects_a_raster_that_does_not_match_the_geometries():
    geometries = NodeGeometries.regular_grid(8, 8, block=4)

    with pytest.raises(ValueError, match="does not match geometries"):
        glcm_texture(np.zeros((4, 4)), geometries)

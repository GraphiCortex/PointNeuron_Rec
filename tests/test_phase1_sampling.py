"""Analytic operator/sampling controls; no real images or labels."""
import numpy as np
import pytest

from pn2.inputs.sampling import (normalized_image, partition_tiles, probabilities,
                                 ranked_selection, sato_guide, selection_count, source_indices)


def tube(radius=2., side=33):
    z, y, x = np.indices((side, side, side), dtype=float)
    middle = (side - 1) / 2
    return np.exp(-((z - middle)**2 + (y - middle)**2) / (2 * radius**2)).astype(np.float32)


def test_analytic_gaussian_tube_scale_sign_axes_and_border():
    # For a continuous Gaussian tube, scale-normalized Hessian response at
    # radius=sigma is 1/4. Boundaries along its invariant X axis cannot change it.
    image = tube()
    response = sato_guide(image, [2.])
    assert response[16, 16, 16] == pytest.approx(.25, rel=.02)
    assert response[16, 16, 16] > 100 * response[0, 0, 16]
    np.testing.assert_allclose(response[16, 16, :], response[16, 16, 16], rtol=1e-6)
    permuted = sato_guide(image.transpose(2, 0, 1), [2.]).transpose(1, 2, 0)
    np.testing.assert_allclose(response, permuted, atol=1e-7, rtol=1e-5)
    dark = sato_guide(1 - image, [2.])
    assert dark[16, 16, 16] < response[16, 16, 16] / 100


def test_scale_response_and_affine_intensity_normalization():
    thin, thick = tube(1., 49), tube(4., 49)
    assert sato_guide(thin, [1.])[24, 24, 24] > sato_guide(thin, [4.])[24, 24, 24]
    assert sato_guide(thick, [4.])[24, 24, 24] > sato_guide(thick, [1.])[24, 24, 24]
    normalized, _ = normalized_image(thick)
    changed, _ = normalized_image(thick * 700 + 23)
    np.testing.assert_allclose(normalized, changed, atol=2 * np.finfo(np.float32).eps)
    response = sato_guide(normalized, [1., 2., 4.])
    # Tubeness takes a square root of clipped eigenvalue products: at a zero
    # eigenvalue the sensitivity is sqrt(epsilon), not epsilon. Coordinates
    # and score tolerances are unrelated to this image-field numerical bound.
    numerical_bound = np.sqrt(np.finfo(np.float32).eps) * response.max()
    np.testing.assert_allclose(response, sato_guide(changed, [1., 2., 4.]), atol=numerical_bound)


def test_y_fixture_remains_finite_with_positive_junction_and_arms():
    z, y, x = np.indices((33, 33, 33), dtype=float)
    xyz = np.stack((x, y, z), axis=-1)
    center = np.array([16., 16., 16.])
    endpoints = [np.array(p) for p in ([3., 16., 16.], [29., 3., 16.], [29., 29., 16.])]
    distances = []
    for endpoint in endpoints:
        delta = endpoint - center
        fraction = np.clip(np.einsum("...i,i->...", xyz - center, delta) / np.dot(delta, delta), 0, 1)
        distances.append(np.linalg.norm(xyz - center - fraction[..., None] * delta, axis=-1))
    image = np.exp(-np.min(distances, axis=0)**2 / 8).astype(np.float32)
    guide = sato_guide(image, [1., 2., 4.])
    assert np.isfinite(guide).all() and guide.min() >= 0
    for endpoint in endpoints:
        midpoint = np.rint((center + endpoint) / 2).astype(int)
        assert guide[tuple(midpoint[::-1])] > .05
    assert guide[16, 16, 16] > .05
    assert guide[16, 16, 16] > 10 * guide[0, 0, 0]


def test_constant_image_and_zero_guide_fallback():
    raw = np.full((3, 4, 5), 23, dtype=np.uint16)
    normalized, transform = normalized_image(raw)
    assert transform == {"minimum": 23., "maximum": 23., "constant": True}
    assert not normalized.any() and np.all(raw == 23)
    guide = sato_guide(normalized, [1., 2., 4.])
    assert not guide.any()
    probability, fallback = probabilities(guide, .1)
    assert fallback and np.all(probability == 1 / 60)


def test_distinct_source_locations_nested_counts_and_exact_intensities():
    shape = (5, 7, 11)
    raw = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    recovered = []
    for tile_id, tile in enumerate(partition_tiles(shape, 4)):
        values = raw[tile]
        probability, _ = probabilities(values, .1)
        full = ranked_selection(probability, values.size, 3, tile_id)
        small = ranked_selection(probability, min(3, values.size), 3, tile_id)
        np.testing.assert_array_equal(small, full[:len(small)])
        flat, xyz = source_indices(full, tile, shape)
        np.testing.assert_array_equal(raw[xyz[:, 2], xyz[:, 1], xyz[:, 0]], raw.ravel()[flat])
        recovered.extend(flat)
    np.testing.assert_array_equal(np.sort(recovered), np.arange(raw.size))
    assert selection_count(128**3, 4) == 32768
    assert selection_count(3, 8) == 1


def test_weighted_sampling_first_draw_distribution_and_reproducibility():
    probability = np.array([.1, .2, .7])
    draws = [ranked_selection(probability, 1, seed, 0)[0] for seed in range(4000)]
    np.testing.assert_allclose(np.bincount(draws, minlength=3) / len(draws), probability, atol=.025)
    for seed in range(5):
        full = ranked_selection(probability, 3, seed, 2)
        np.testing.assert_array_equal(np.sort(full), np.arange(3))
        np.testing.assert_array_equal(full, ranked_selection(probability, 3, seed, 2))


@pytest.mark.parametrize("values,mass", [([-1, 2], .1), ([np.nan, 1], .1), ([0, 1], 0), ([0, 1], 1.1)])
def test_invalid_weights_stop(values, mass):
    with pytest.raises(ValueError):
        probabilities(values, mass)

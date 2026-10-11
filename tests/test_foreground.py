import numpy as np
import pytest

from pointneuron.data.foreground import foreground_mask, otsu_threshold


def test_otsu_separates_two_populations_and_constant_has_no_foreground():
    image = np.concatenate((np.full(900, 10), np.full(100, 100))).reshape(10, 10, 10)
    mask, _, _ = foreground_mask(image, 'raw_otsu')
    np.testing.assert_array_equal(mask, image == 100)
    constant, _, metadata = foreground_mask(np.full((3, 4, 5), 100), 'raw_otsu')
    assert not constant.any() and metadata['constant_score']


def test_fixed_mask_matches_existing_rule_and_preserves_raw_image():
    image = np.arange(120, dtype=np.uint8).reshape(3, 4, 10)
    before = image.copy()
    mask, score, _ = foreground_mask(image, 'fixed', fixed_threshold=51)
    np.testing.assert_array_equal(mask, image > 51)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(score, image)


def test_background_rule_retains_thin_bright_line_on_smooth_ramp():
    image = np.broadcast_to(np.linspace(20, 80, 49)[None, None, :], (9, 49, 49)).copy()
    image[:, 24, :] += 80
    mask, score, metadata = foreground_mask(image, 'background_otsu', background_sigma=8,
                                            spacing_xyz=(1, 1, 2))
    assert mask[:, 24, :].all() and mask.mean() < .1
    assert score.min() >= 0 and metadata['sigma_zyx'] == [4, 8, 8]


def test_mask_selection_is_invariant_to_positive_intensity_rescaling():
    rng = np.random.default_rng(3)
    image = rng.uniform(0, 10, (8, 9, 10)); image[:, 4, 4] += 50
    for method in ('raw_otsu', 'background_otsu'):
        np.testing.assert_array_equal(foreground_mask(image, method)[0],
                                      foreground_mask(image * 100, method)[0])


def test_invalid_images_and_settings_are_rejected():
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan]))
    with pytest.raises(ValueError):
        foreground_mask(np.zeros((3, 4)), 'fixed')
    with pytest.raises(ValueError):
        foreground_mask(np.zeros((3, 4, 5)), 'background_otsu', spacing_xyz=(1, 0, 1))
    with pytest.raises(ValueError):
        foreground_mask(np.zeros((3, 4, 5)), 'unknown')


def test_context_halo_matches_full_volume_filtering_inside_core():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(5)
    image = rng.uniform(0, 100, (12, 80, 80)).astype(np.float32)
    context = image[:, 14:66, 14:66]
    core = (slice(3, 9), slice(16, 36), slice(16, 36))
    mask, score, metadata = foreground_mask(context, 'background_otsu', background_sigma=4, core_slices=core)
    full = np.maximum(image - gaussian_filter(image, sigma=4, mode='reflect'), 0)[3:9, 30:50, 30:50]
    np.testing.assert_allclose(score, full, atol=1e-5)
    np.testing.assert_array_equal(mask, full > metadata['threshold'])


def test_compact_patch_selection_matches_existing_image_coverage_rule():
    from scripts.aggregate_proposals import build_patch_specs
    from scripts.pointneuron2.compare_foreground import image_patch_boxes
    rng = np.random.default_rng(6)
    image = rng.integers(0, 100, (16, 24, 32), dtype=np.uint8)
    config = {'patch_radius': 6, 'patch_stride': 8, 'max_patches_per_neuron': 4, 'min_foreground_points': 1}
    original = build_patch_specs(np.flatnonzero(image.ravel() > 51), image.ravel(), 32, 24, 16,
                                  6, 8, 51, 1, 4, 'coverage')
    compact = image_patch_boxes(image, 51, config)
    np.testing.assert_array_equal([p['center'] for _, p in original], [p['center'] for p in compact])
    assert [p['foreground'] for _, p in original] == [p['foreground'] for p in compact]

import numpy as np
import pytest

from pointneuron.data.foreground import (difference_noise_scale, foreground_mask, noise_hysteresis_mask,
                                        guarded_noise_hysteresis_mask, otsu_threshold)
from scripts.pointneuron2.compare_foreground import frozen_patch_boxes
from scripts.pointneuron2.inspect_foreground_slices import clip_segments_to_slab


def test_difference_estimator_recovers_noise_despite_background_offset():
    raw = np.random.default_rng(4).normal(100, 5, (40, 40, 40))
    sigma, _ = difference_noise_scale(raw)
    assert sigma == pytest.approx(5, rel=.04)
    assert difference_noise_scale(raw + 1000)[0] == pytest.approx(sigma)


def test_noise_only_patch_does_not_become_dense_foreground():
    raw = np.random.default_rng(8).normal(80, 4, (48, 48, 48))
    mask, _, _ = foreground_mask(raw, 'noise_hysteresis')
    assert mask.mean() < .002


def test_repeated_textured_intensities_do_not_trigger_noise_floor_flood():
    raw = np.repeat(np.random.default_rng(5).integers(50, 100, (12, 12, 12), dtype=np.uint8), 4, axis=2)
    raw = np.repeat(raw, 4, axis=1)
    # Majority adjacent values are identical along two axes, although texture exists.
    score = np.random.default_rng(7).uniform(0, 20, raw.shape)
    old, _ = noise_hysteresis_mask(score, raw)
    guarded, info = guarded_noise_hysteresis_mask(score, raw)
    assert not info['noise_estimate_above_floor']
    np.testing.assert_array_equal(guarded, score > otsu_threshold(score))
    assert old.mean() > .95 and guarded.mean() < .6


def test_reliable_noise_guard_keeps_connected_signal_within_otsu_bound():
    raw = np.random.default_rng(5).normal(80, 2, (20, 20, 20))
    score = np.zeros_like(raw); score[5, 5, 2] = 12; score[5, 5, 3:10] = 5
    mask, info = guarded_noise_hysteresis_mask(score, raw)
    assert info['noise_estimate_above_floor'] and info['fallback'] == 'none'
    assert mask[5, 5, 2:10].all()
    assert np.all(~mask | (score > otsu_threshold(score)))


def test_weak_voxels_need_connected_strong_seed_without_gap_filling():
    raw = np.random.default_rng(7).normal(80, 1, (20, 20, 20))
    score = np.zeros_like(raw)
    score[5, 5, 2] = 8
    score[5, 5, 3:10] = 3
    score[5, 5, 11:15] = 3  # One missing voxel prevents propagation.
    score[15, 15, 2:10] = 3
    mask, _ = noise_hysteresis_mask(score, raw)
    assert mask[5, 5, 2:10].all()
    assert not mask[5, 5, 10:].any() and not mask[15, 15].any()


def test_constant_and_sparse_quantized_images_have_finite_noise_floor():
    raw = np.full((8, 8, 8), 100, dtype=np.uint8)
    mask, _, metadata = foreground_mask(raw, 'noise_hysteresis')
    assert not mask.any() and metadata['noise_sigma'] > 0
    raw[:] = 0; raw[3, 3, 3] = 255
    assert difference_noise_scale(raw)[0] == pytest.approx(1 / np.sqrt(12))


def test_noise_method_preserves_image_and_scales_with_float_intensity():
    raw = np.random.default_rng(3).normal(40, 2, (20, 30, 30)).astype(np.float32)
    raw[:, 15, 15] += 40
    before = raw.copy()
    mask, score, _ = foreground_mask(raw, 'noise_hysteresis')
    scaled, scaled_score, _ = foreground_mask(raw * 100, 'noise_hysteresis')
    np.testing.assert_array_equal(mask, scaled)
    np.testing.assert_allclose(scaled_score, score * 100, atol=.001)
    np.testing.assert_array_equal(raw, before)
    assert mask[:, 15, 15].all()


@pytest.mark.parametrize('high,low', [(1, 2), (4, 0), (np.nan, 2), (4, np.inf)])
def test_invalid_noise_thresholds_rejected(high, low):
    with pytest.raises(ValueError):
        foreground_mask(np.zeros((3, 3, 3)), 'noise_hysteresis', noise_high=high, noise_low=low)


def test_reference_edges_are_clipped_to_actual_slice():
    segments = np.array([[[0, 0, 0], [10, 10, 10]], [[0, 0, 7], [10, 10, 7]]])
    clipped = clip_segments_to_slab(segments, 5)
    assert len(clipped) == 1
    np.testing.assert_allclose(clipped[0], [[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])
    assert len(clip_segments_to_slab(segments, 7)) == 2


def test_frozen_boxes_preserve_indices_and_reject_out_of_bounds_or_validation():
    row = {'sample_index': 0, 'sample_id': 'a', 'partition': 'train', 'patch_index': 7,
           'lower_xyz': [1, 2, 3], 'upper_xyz': [4, 5, 6]}
    boxes = frozen_patch_boxes({'patches': [row]}, 0, 'a', (10, 10, 10, 1))
    assert boxes[0][0] == 7
    np.testing.assert_array_equal(boxes[0][1]['lower'], [1, 2, 3])
    for altered in ({**row, 'partition': 'val'}, {**row, 'upper_xyz': [12, 5, 6]}):
        with pytest.raises(ValueError):
            frozen_patch_boxes({'patches': [altered]}, 0, 'a', (10, 10, 10, 1))

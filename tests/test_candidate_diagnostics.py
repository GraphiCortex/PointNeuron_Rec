from unittest.mock import patch

import numpy as np
import pytest

from pointneuron.data.candidate_diagnostics import image_candidate_indices, select_candidate_rows
from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.sampling_cache import SamplingPoolConfig, build_image_pools
from pointneuron.data.training_cache import indices_to_point_array
from scripts.pointneuron2.compare_foreground import process_mask
from scripts.pointneuron2.compare_candidates import summaries, validate_config
from scripts.pointneuron2.reference_gaps import segment_probe, segment_support


def test_uniform_pool_matches_actual_cache_with_identical_seed_and_raw_values():
    data = np.arange(200, dtype=np.uint16)
    indices = np.arange(0, 200, 2)
    with patch('pointneuron.data.sampling_cache.image_medial_axis', return_value=np.empty((0, 3))):
        actual = build_image_pools(indices, data, 20, 10, SamplingPoolConfig(32), MedialSamplingConfig(), 7)
    chosen = image_candidate_indices(indices, data, 20, 10, 32, 'uniform', 7)
    np.testing.assert_array_equal(chosen, actual['candidate_indices'])
    np.testing.assert_array_equal(indices_to_point_array(chosen, data, 20, 10), actual['points'])


@pytest.mark.parametrize('method', ['uniform', 'spatial'])
def test_sparse_pools_preserve_all_original_voxels_and_reportable_padding(method):
    indices = np.array([2, 7, 99]); data = np.arange(100)
    chosen = image_candidate_indices(indices, data, 100, 1, 16, method, 3)
    assert len(chosen) == 16 and set(chosen) == set(indices)
    assert np.array_equal(chosen, image_candidate_indices(indices, data, 100, 1, 16, method, 3))
    assert not len(image_candidate_indices(np.array([], dtype=int), data, 100, 1, 16, method, 3))


def test_spatial_pool_covers_separated_image_regions_and_fills_without_duplicates():
    indices = np.r_[np.arange(30), np.arange(270, 300)]
    data = np.ones(300); data[29] = 100; data[299] = 100
    chosen = image_candidate_indices(indices, data, 300, 1, 8, 'spatial', 4, cell_size=10)
    assert len(chosen) == len(np.unique(chosen)) == 8
    assert 29 in chosen and 299 in chosen and set(chosen) <= set(indices)


def test_point_selection_keeps_unique_pool_rows_and_empty_guide_is_uniform():
    points = np.column_stack((np.arange(100), np.zeros((100, 2)), np.arange(100)))
    guide = np.array([[0., 0, 0]])
    for method in ('random', 'fixed_medial', 'farthest'):
        rows = select_candidate_rows(points, guide, 24, method, 5)
        assert len(rows) == len(np.unique(rows)) == 24 and np.all((rows >= 0) & (rows < 100))
        np.testing.assert_array_equal(rows, select_candidate_rows(points, guide, 24, method, 5))
    np.testing.assert_array_equal(select_candidate_rows(points, np.empty((0, 3)), 24, 'fixed_medial', 5),
                                  select_candidate_rows(points, guide, 24, 'random', 5))
    assert not len(select_candidate_rows(np.empty((0, 4)), guide, 24, 'fixed_medial', 5))
    guided_x = points[select_candidate_rows(points, guide, 24, 'fixed_medial', 5), 0].mean()
    random_x = points[select_candidate_rows(points, guide, 24, 'random', 5), 0].mean()
    assert guided_x < random_x


def test_invalid_or_duplicate_voxels_and_unknown_selector_are_rejected():
    with pytest.raises(ValueError):
        image_candidate_indices(np.array([1, 1]), np.ones(10), 10, 1, 3, 'uniform', 0)
    with pytest.raises(ValueError):
        image_candidate_indices(np.array([10]), np.ones(10), 10, 1, 3, 'uniform', 0)
    with pytest.raises(ValueError):
        image_candidate_indices(np.arange(10), np.ones(10), 10, 1, 3, 'oracle', 0)
    with pytest.raises(ValueError):
        select_candidate_rows(np.ones((10, 4)), np.ones((1, 3)), 3, 'oracle', 0)
    with pytest.raises(ValueError):
        select_candidate_rows(np.ones((10, 4)), np.ones((1, 3)), 11, 'random', 0)


def test_expanded_mask_path_preserves_every_guide_vertex():
    image = np.full((3, 4, 5), 100, dtype=np.uint8)
    guide = np.arange(1800, dtype=float).reshape(600, 3)
    with patch('scripts.pointneuron2.compare_foreground.image_medial_skeleton', return_value=(guide.copy(), np.empty((0, 2), dtype=int))):
        outputs = process_mask(image, image > 51, np.array([10, 20, 30]), MedialSamplingConfig(),
                               {'candidate_points': 32, 'guide_points': None}, 2)
    np.testing.assert_array_equal(outputs['image_guide_xyz'], guide + [10, 20, 30])
    assert outputs['points'].shape == (32, 4) and np.all(outputs['points'][:, 3] == 100)


def test_reference_gap_uses_arc_length_and_keeps_segments_separate():
    segments = np.array([[[0, 0, 0], [10, 0, 0]], [[0, 5, 0], [2, 5, 0]]], dtype=float)
    probe = segment_probe(segments, step=1.)
    xyz = np.array([[.5, 0, 0], [9.5, 0, 0], [.5, 5, 0], [1.5, 5, 0]])
    output = segment_support(probe, xyz, distance=.6)
    assert output['reference_length'] == 12 and output['supported_length'] == 4
    assert output['supported_length_fraction'] == pytest.approx(1 / 3)
    assert output['max_unsupported_run_length'] == 8 and output['fully_supported_segments'] == 1
    empty = segment_support(probe, np.empty((0, 3)), distance=.6)
    assert empty['supported_length'] == 0 and empty['max_unsupported_run_length'] == 10


def test_reference_gap_weights_long_and_short_edges_and_applies_spacing_once():
    probe = segment_probe(np.array([[[0, 0, 0], [1, 0, 0]], [[0, 10, 0], [1, 10, 0]]]), spacing=(2, 1, 1))
    output = segment_support(probe, np.array([[.25, 0, 0], [.75, 0, 0]]), distance=.1)
    assert output['reference_length'] == 4 and output['supported_length'] == 2
    assert output['max_unsupported_run_length'] == 2
    assert segment_support(segment_probe(np.empty((0, 2, 3))), np.empty((0, 3))) is None


def test_summary_averages_seeds_within_patch_and_excludes_singleton_references():
    def variant(support, seed):
        return {'candidate_method': 'uniform', 'candidate_budget': 1024, 'seed_offset': seed,
                'selectors': {'random': {'reference_support': {'within_3': support}, 'reference_segment_support': None}}}
    def row(sample, refs, variants):
        return {'sample_index': sample, 'source_group': 'source', 'reference_curve_points': refs,
                'methods': {'fixed': {'variants': variants}}}
    output = summaries([row(0, 3, [variant(0., 0), variant(1., 1)]), row(1, 4, [variant(1., 0)]),
                        row(2, 1, [variant(0., 0)])])
    assert output[0]['mean_over_patches_after_seed_average']['within_3'] == .75
    assert output[0]['reference_patches'] == output[0]['reference_neurons'] == 2


def test_comparison_config_rejects_underbudget_and_invalid_seeds():
    base = {'candidate_budgets': [1024], 'selected_points': 512, 'candidate_methods': ['uniform'],
            'selectors': ['random'], 'seed_offsets': [0, 1], 'cell_size': 8, 'reference_distance': 3., 'reference_step': 1.}
    validate_config(base)
    for changed in ({'candidate_budgets': [256]}, {'seed_offsets': [True]}, {'selectors': ['oracle']},
                    {'reference_step': 0}, {'candidate_methods': []}):
        with pytest.raises(ValueError):
            validate_config({**base, **changed})

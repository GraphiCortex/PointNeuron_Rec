import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from pointneuron.data.manual_review import (build_review_candidate, farthest_indices,
                                         review_settings, spacing_metadata, write_review_decision)
from pointneuron.data.medial_sampling import MedialSamplingConfig, image_medial_skeleton
from scripts.pointneuron2.review_sampling import ReviewSession


def settings():
    return {'threshold_fraction': .2, 'spacing_xyz': [1, 1, 1], 'teasar_scale': 1.5,
            'teasar_const': 2., 'dust_threshold': 8, 'candidate_method': 'uniform',
            'guide_method': 'uniform', 'cell_size': 8}


def test_source_spacing_is_explicit_relative_and_missing_is_unknown(tmp_path):
    path = tmp_path / 'spacing.txt'; path.write_text('0.103x0.103x0.3um')
    metadata = spacing_metadata(path)
    np.testing.assert_allclose(metadata['relative_xyz'], [1, 1, .3 / .103])
    assert metadata['physical_xyz_um'] == [.103, .103, .3]
    assert spacing_metadata(None)['relative_xyz'] is None
    path.write_text('unknown resolution')
    assert spacing_metadata(path)['status'] == 'unrecognized'


def test_farthest_pool_preserves_extent_disconnected_regions_and_sparse_rows():
    points = np.column_stack((np.r_[np.arange(10), np.arange(90, 100)], np.zeros(20), np.zeros(20)))
    selected = farthest_indices(points, 4)
    assert len(set(selected)) == 4 and 0 in selected and 19 in selected
    np.testing.assert_array_equal(farthest_indices(points, 30), np.arange(20))
    np.testing.assert_array_equal(selected, farthest_indices(points, 4))


def test_review_candidates_are_image_only_and_preserve_raw_channels():
    image = np.zeros((2, 4, 12), dtype=np.uint8); image[:, 1, :] = 100
    guide = np.array([[0, 1, 0], [11, 1, 0]], dtype=float); edges = np.array([[0, 1]])
    with patch('pointneuron.data.manual_review.image_medial_skeleton', return_value=(guide.copy(), edges)):
        candidate = build_review_candidate(image, [20, 30, 40], settings(), MedialSamplingConfig(),
                                           candidate_count=8, guide_count=2, seed=3)
    assert candidate['points'].shape == (8, 4)
    assert set(candidate['points'][:, 3]) == {100}
    np.testing.assert_allclose(candidate['full_guide_xyz'], [[20, 31, 40], [31, 31, 40]])
    changed = {**settings(), 'candidate_method': 'spatial', 'guide_method': 'farthest', 'cell_size': 2}
    alternative = build_review_candidate(image, [20, 30, 40], changed, MedialSamplingConfig(), 8, 2, 3,
                        existing_skeleton=(candidate['full_guide_xyz'], candidate['guide_edges']))
    assert len(np.unique(alternative['points'][:, :3], axis=0)) == 8
    assert np.all(alternative['points'][:, 1] == 31)
    np.testing.assert_array_equal(alternative['full_guide_xyz'], candidate['full_guide_xyz'])
    uncapped = build_review_candidate(image, [20, 30, 40], changed, MedialSamplingConfig(),
                       candidate_count=8, guide_count=None, seed=3,
                       existing_skeleton=(candidate['full_guide_xyz'], candidate['guide_edges']))
    np.testing.assert_array_equal(uncapped['image_guide_xyz'], candidate['full_guide_xyz'])


def test_review_settings_reject_nan_unknown_fields_and_ambiguous_units():
    base = MedialSamplingConfig()
    for changed in ({'threshold_fraction': float('nan')}, {'spacing_xyz': [.103, .103, .3]},
                    {'cell_size': 0}, {'candidate_method': 'oracle'}, {'unexpected': 3}):
        with pytest.raises(ValueError):
            review_settings({**settings(), **changed}, base)


def test_component_edge_indices_are_offset_without_bridging():
    skeletons = {1: SimpleNamespace(vertices=np.array([[0, 0, 0], [1, 0, 0]]), edges=np.array([[0, 1]])),
                 2: SimpleNamespace(vertices=np.array([[20, 0, 0], [21, 0, 0]]), edges=np.array([[0, 1]]))}
    with patch('kimimaro.skeletonize', return_value=skeletons):
        vertices, edges = image_medial_skeleton(np.array([2, 3, 22, 23]), 30, 1)
    assert len(vertices) == 4
    np.testing.assert_array_equal(edges, [[0, 1], [2, 3]])


def review_fixture(tmp_path):
    asset = tmp_path / 'case.npz'
    np.savez(asset, image=np.zeros((2, 3, 4), dtype=np.uint8), reference_curve=np.ones((1, 3)),
             reference_segments=np.empty((0, 2, 3)))
    candidate = tmp_path / 'candidate.npz'
    np.savez(candidate, full_guide_xyz=np.ones((1, 3)))
    state = {'revision': 1, 'full_guide_vertices': 1, 'settings': settings(), 'artifact': candidate.name,
             'artifact_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()}
    case = {'case_id': 'case1', 'sample_id': 'neuron1', 'partition': 'val', 'source_volume_sha256': 'image',
            'source_swc_sha256': 'swc', 'current_state': state, 'case_asset': asset.name,
            'case_asset_sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}
    (tmp_path / 'session.json').write_text(json.dumps({'status': 'ready_for_manual_review', 'cases': [case]}))
    return case, state


def test_human_decision_binds_candidate_hash_and_does_not_promote_training(tmp_path):
    case, state = review_fixture(tmp_path)
    path, decision = write_review_decision(tmp_path, case, state, 'accepted', 'Synthetic test reviewer', reference_exposed=True)
    assert path.is_file() and not decision['training_ready']
    assert not decision['eligible_for_image_only_validation']
    np.savez(tmp_path / state['artifact'], full_guide_xyz=np.zeros((1, 3)))
    with pytest.raises(ValueError, match='changed'):
        write_review_decision(tmp_path, case, state, 'accepted', 'Synthetic test reviewer')


def test_stale_empty_and_unnamed_reviews_are_rejected(tmp_path):
    case, state = review_fixture(tmp_path)
    session = ReviewSession(tmp_path)
    with pytest.raises(ValueError, match='stale'):
        session.decide({'case_id': case['case_id'], 'revision': 0, 'decision': 'accepted', 'reviewer': 'Test'})
    with pytest.raises(ValueError, match='reviewer'):
        write_review_decision(tmp_path, case, state, 'accepted', '')
    with pytest.raises(ValueError, match='empty'):
        write_review_decision(tmp_path, case, {**state, 'full_guide_vertices': 0}, 'accepted', 'Test')
    with pytest.raises(ValueError, match='why'):
        write_review_decision(tmp_path, case, state, 'rejected', 'Test')

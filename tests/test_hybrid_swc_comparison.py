"""Guard the paired experiment against pool drift and fabricated features."""
import json

import numpy as np
import pytest

from scripts.pointneuron2.compare_hybrid_swc import check_snapshot, make_hybrid, validate_selection


def fixture_arrays():
    n = 17000
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 0] = np.arange(n)
    features = np.column_stack((np.arange(n), np.arange(n) + 1)).astype(np.float32)
    radii = np.ones(n, dtype=np.float32)
    scores = np.full(n, .9, dtype=np.float32)
    patches = np.arange(n, dtype=np.int32) // 512
    full = dict(all_centers=xyz, all_radii=radii, all_scores=scores,
        all_features=features, all_patch_indices=patches,
        centers=xyz[:4096], radii=radii[:4096], scores=scores[:4096],
        features=features[:4096], patch_indices=patches[:4096],
        metadata=np.array(json.dumps({'sample_index': 79, 'metrics': {'not_hybrid': True}})))
    old = {k: v for k, v in full.items() if k not in ('all_features', 'all_patch_indices')}
    indices = np.concatenate((np.arange(2048), np.arange(5000, 7048))).astype(np.int64)
    selection = dict(source_indices=indices, anchor_indices=np.arange(2048, dtype=np.int64),
        candidate_order=np.arange(16384, dtype=np.int64), centers=xyz[indices],
        radii=radii[indices], scores=scores[indices])
    return full, old, selection


def test_snapshot_calculations_unchanged():
    check_snapshot()


def test_hybrid_keeps_real_source_features_and_scores(tmp_path):
    full, old, selection = fixture_arrays()
    for name, payload in [('full', full), ('old', old), ('selection', selection)]:
        np.savez_compressed(tmp_path / f'{name}.npz', **payload)
    output = tmp_path / 'hybrid.npz'
    with np.load(tmp_path / 'full.npz') as f, np.load(tmp_path / 'old.npz') as o, np.load(tmp_path / 'selection.npz') as s:
        make_hybrid(f, o, s, output)
    with np.load(output) as result:
        idx = selection['source_indices']
        for key, source in [('features', 'all_features'), ('scores', 'all_scores'), ('patch_indices', 'all_patch_indices')]:
            np.testing.assert_array_equal(result[key], full[source][idx])
        assert 'metrics' not in json.loads(str(result['metadata']))


@pytest.mark.parametrize('key', ['all_centers', 'all_scores', 'features'])
def test_pool_or_accepted_feature_drift_stops_before_tracing(tmp_path, key):
    full, old, selection = fixture_arrays()
    full[key] = full[key].copy()
    full[key].flat[0] += .01
    np.savez_compressed(tmp_path / 'full.npz', **full)
    with np.load(tmp_path / 'full.npz') as f:
        with pytest.raises(ValueError, match='differs from saved control'):
            make_hybrid(f, old, selection, tmp_path / 'bad.npz')
    assert not (tmp_path / 'bad.npz').exists()


def test_misaligned_complete_features_are_rejected(tmp_path):
    full, old, selection = fixture_arrays()
    full['all_features'] = np.roll(full['all_features'], 1, axis=0)
    np.savez_compressed(tmp_path / 'full.npz', **full)
    with np.load(tmp_path / 'full.npz') as f:
        with pytest.raises(ValueError, match='Feature or patch identities'):
            make_hybrid(f, old, selection, tmp_path / 'bad.npz')


def test_invalid_anchor_prefix_rejected():
    _, old, selection = fixture_arrays()
    selection['source_indices'] = selection['source_indices'][::-1]
    with pytest.raises(ValueError, match='budgets, source identities or anchors'):
        validate_selection(old, selection)

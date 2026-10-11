import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.data.sampling_cache import SamplingPoolConfig, build_image_pools, load_sampling_manifest, SamplingCacheDataset, collate_sampling_records
from pointneuron.models.guide_distances import nearest_distances
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler, masked_chamfer, symmetric_chamfer
from pointneuron.models.sampling_pipeline import sampler_from_checkpoint, inference_sample
from test_sampling_training import fixture_manifest


@pytest.mark.parametrize('squared', [False, True])
def test_blocked_values_and_both_coordinate_gradients_match_dense(squared):
    torch.manual_seed(12)
    q = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
    r = torch.randn(2, 11, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.rand(2, 11) > .3
    dense = torch.cdist(q, r, compute_mode='donot_use_mm_for_euclid_dist').masked_fill(~mask[:, None], torch.inf).amin(-1)
    if squared:
        dense = dense.square()
    gradient = torch.randn_like(dense)
    expected = torch.autograd.grad(dense, (q, r), gradient)
    blocked = nearest_distances(q, r, mask, chunk_size=3, squared=squared)
    actual = torch.autograd.grad(blocked, (q, r), gradient)
    torch.testing.assert_close(blocked, dense)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


def test_ties_and_coincident_points_match_amin_gradient_sharing():
    q = torch.tensor([[[0., 0, 0], [1, 0, 0]]], requires_grad=True)
    r = torch.tensor([[[-1., 0, 0], [1, 0, 0], [1, 0, 0]]], requires_grad=True)
    for squared in (False, True):
        dense = torch.cdist(q, r, compute_mode='donot_use_mm_for_euclid_dist')
        if squared:
            dense = dense.square()
        expected = torch.autograd.grad(dense.amin(-1).sum(), (q, r))
        result = nearest_distances(q, r, chunk_size=1, squared=squared)
        actual = torch.autograd.grad(result.sum(), (q, r))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
            assert torch.isfinite(a).all()


def test_nearest_gradcheck_and_zero_reference_mask():
    torch.manual_seed(4)
    q = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
    r = torch.rand(1, 5, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: nearest_distances(a, b, chunk_size=2, squared=True), (q, r))
    assert torch.isinf(nearest_distances(q, r, torch.zeros(1, 5, dtype=torch.bool), 2)).all()


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_sampler_and_chamfer_preserve_gradients_with_padding_and_empty_guide(device):
    torch.manual_seed(6)
    points = torch.rand(2, 35, 4, device=device)
    guide = torch.rand(2, 47, 3, device=device)
    mask = torch.ones(2, 47, dtype=torch.bool, device=device); mask[0, 39:] = False; mask[1] = False
    gradients, probabilities, losses = [], [], []
    for chunk in (None, 9):
        sampler = AdaptiveMedialSampler(distance_chunk_size=chunk).to(device)
        torch.manual_seed(8)
        output = sampler(points, guide, 12, mask)
        loss = masked_chamfer(output.points[..., :3], guide, mask, chunk)
        loss.backward()
        probabilities.append(output.probabilities); gradients.append(sampler.log_alpha.grad); losses.append(loss)
        assert torch.isfinite(sampler.log_alpha.grad)
        torch.testing.assert_close(output.probabilities[1], torch.full((35,), 1 / 35, device=device))
    torch.testing.assert_close(probabilities[0], probabilities[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(losses[0], losses[1])


def test_full_guide_preserves_every_vertex_and_candidate_budget():
    guide = np.arange(1800, dtype=np.float32).reshape(600, 3)
    with patch('pointneuron.data.sampling_cache.image_medial_axis', return_value=guide):
        full = build_image_pools(np.arange(100), np.arange(100), 100, 1, SamplingPoolConfig(16), MedialSamplingConfig(), 2)
        capped = build_image_pools(np.arange(100), np.arange(100), 100, 1, SamplingPoolConfig(16, 8), MedialSamplingConfig(), 2)
    np.testing.assert_array_equal(full['image_guide_xyz'], guide)
    np.testing.assert_array_equal(full['points'], capped['points'])
    assert full['full_guide_count'] == 600 and len(capped['image_guide_xyz']) == 8
    for value in (0, -1, True, 'full'):
        with pytest.raises(ValueError):
            SamplingPoolConfig(16, value)


def test_full_cache_contract_rejects_discarded_vertices_and_loads_variable_guides(tmp_path):
    path, partition, manifest, record = fixture_manifest(tmp_path)
    manifest['config']['guide_points'] = None
    with np.load(record) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = json.loads(str(arrays['metadata'])); metadata['config'] = manifest['config']
    metadata.update(sampling_cache_schema=2, full_guide_count=2)
    arrays['metadata'] = np.array(json.dumps(metadata))
    np.savez(record, **arrays)
    manifest['records'][0]['sha256'] = hashlib.sha256(record.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    payload = load_sampling_manifest(path, partition)
    data = SamplingCacheDataset([payload['records'][0]['resolved_path']])
    batch = collate_sampling_records([data[0]])
    assert batch['image_guide_mask'].sum() == 2
    metadata['full_guide_count'] = 3; arrays['metadata'] = np.array(json.dumps(metadata)); np.savez(record, **arrays)
    manifest['records'][0]['sha256'] = hashlib.sha256(record.read_bytes()).hexdigest(); path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='discarded'):
        load_sampling_manifest(path, partition)


def test_new_checkpoint_preserves_distance_config_and_legacy_remains_dense():
    sampler = AdaptiveMedialSampler(distance_chunk_size=8)
    config = {'initial_alpha': .25, 'uniform_fraction': .25, 'distance_chunk_size': 8}
    checkpoint = {'sampler': sampler.state_dict(), 'sampling_pipeline': {'schema_version': 1, 'strategy': 'learned', 'sampler_config': config}}
    assert sampler_from_checkpoint(checkpoint).distance_chunk_size == 8
    checkpoint['sampling_pipeline'].update(volume_reader_version=VOLUME_READER_VERSION,
                       pool_config={'candidate_points': 16, 'guide_points': None},
                       medial_config=MedialSamplingConfig().to_dict(), selected_points=8, evaluation_seed=3)
    guide = np.arange(1800, dtype=np.float32).reshape(600, 3)
    restored = sampler_from_checkpoint(checkpoint)
    with patch('pointneuron.data.sampling_cache.image_medial_axis', return_value=guide):
        inputs = build_image_pools(np.arange(100), np.arange(100), 100, 1, SamplingPoolConfig(16), MedialSamplingConfig(), 2)
        expected = restored(torch.tensor(inputs['points'])[None], torch.tensor(guide)[None], 8, seed=5).points
        actual = inference_sample(checkpoint, restored, np.arange(100), np.arange(100), 100, 1, 2)
    np.testing.assert_array_equal(actual, expected[0].detach().numpy())
    del config['distance_chunk_size']
    assert sampler_from_checkpoint(checkpoint).distance_chunk_size is None


def test_large_coordinate_distances_and_zero_chamfer_are_stable():
    guide = torch.tensor([[[10000., 10000, 10000], [10001, 10000, 10000]]], requires_grad=True)
    loss = symmetric_chamfer(guide, guide, distance_chunk_size=1)
    loss.backward()
    assert loss == 0 and torch.isfinite(guide.grad).all() and (guide.grad == 0).all()

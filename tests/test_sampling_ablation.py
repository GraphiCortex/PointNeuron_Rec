import copy
import hashlib
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from scripts.pointneuron2.sampling_ablation import check_matches, live
from scripts.pointneuron2.train_sampling import (REPO_ROOT, isolated_sampling_rng, sampling_objective,
                                                main as train_main)


def test_sampler_draw_count_does_not_change_model_random_stream():
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    with isolated_sampling_rng(42, 'cpu'):
        first = torch.rand(12)
        torch.rand(1000)
    assert torch.equal(torch.get_rng_state(), before)
    with isolated_sampling_rng(42, 'cpu'):
        torch.testing.assert_close(torch.rand(12), first)


@pytest.mark.parametrize('strategy,weight', [('random', .05), ('learned', 0.)])
def test_disabled_auxiliary_has_no_chamfer_computation(strategy, weight):
    supervision = torch.tensor(2., requires_grad=True)
    with patch('scripts.pointneuron2.train_sampling.masked_chamfer', side_effect=AssertionError('Must skip')):
        total, chamfer = sampling_objective(supervision, None, None, None, None,
                                            {'chamfer_weight': weight}, strategy)
    assert total is supervision and chamfer is None


def test_live_runner_preserves_failed_child_log_and_raises(tmp_path):
    path = tmp_path / 'failed.log'
    with pytest.raises(subprocess.CalledProcessError):
        live([sys.executable, '-u', '-c', 'print("visible failure"); raise SystemExit(7)'], path)
    assert 'visible failure' in path.read_text()


def synthetic_manifest(tmp_path):
    config = json.loads((REPO_ROOT / 'configs/pointneuron2/sampling_training.json').read_text())
    config.update(candidate_points=64, selected_points=32, epochs=1)
    medial = MedialSamplingConfig().to_dict()
    partition = {'partitions': {'train': [{'sample_index': 0, 'sample_id': 'train-neuron'}],
                               'val': [{'sample_index': 1, 'sample_id': 'val-neuron'}], 'test': []}}
    part_path = tmp_path / 'partition.json'; part_path.write_text(json.dumps(partition))
    rows = []
    for index, role in enumerate(('train', 'train', 'val')):
        sample_index = 0 if role == 'train' else 1
        identity = f'{role}-neuron'
        points = np.random.default_rng(index).uniform(0, 10, (64, 4)).astype(np.float32)
        metadata = {'sampling_cache_schema': 2, 'volume_reader_version': VOLUME_READER_VERSION,
                    'sample_index': sample_index, 'sample_id': identity,
                    'partition': role, 'config': config, 'medial_config': medial, 'seed': index,
                    'full_guide_count': 2}
        path = tmp_path / f'{index}.npz'
        np.savez(path, points=points, image_guide_xyz=points[:2, :3],
                 skeleton_nodes=np.array([[1, 5, 5, 5, 1, -1]], dtype=np.float32),
                 edge_index=np.empty((0, 2), dtype=np.int64), metadata=np.array(json.dumps(metadata)))
        rows.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'sample_index': sample_index, 'sample_id': identity, 'partition': role})
    manifest = {'status': 'completed', 'volume_reader_version': VOLUME_READER_VERSION,
                'failures': [], 'config': config, 'medial_config': medial,
                'partition_sha256': hashlib.sha256(part_path.read_bytes()).hexdigest(), 'records': rows}
    manifest_path = tmp_path / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    return manifest_path, part_path


def test_three_cpu_arms_share_initialization_batches_and_augmentation(tmp_path, monkeypatch):
    manifest, partition = synthetic_manifest(tmp_path)
    runs = []
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for name, strategy, weight in [('random', 'random', 0), ('no_aux', 'learned', 0), ('aux', 'learned', .05)]:
            output = tmp_path / name
            monkeypatch.setattr(sys, 'argv', ['train_sampling.py', '--manifest', str(manifest),
                                '--partition', str(partition), '--output-dir', str(output),
                                '--device', 'cpu', '--strategy', strategy, '--chamfer-weight', str(weight),
                                '--proposal-coordinate-mode', 'local'])
            assert train_main() == 0
            checkpoint = torch.load(output / 'proposal.pt', map_location='cpu', weights_only=False)
            assert checkpoint['args']['proposal_coordinate_mode'] == 'local'
            runs.append(json.loads((output / 'run.json').read_text()))
            checkpoint = torch.load(output / 'proposal.pt', weights_only=False)
            assert checkpoint['sampling_pipeline']['strategy'] == strategy
            assert (checkpoint['sampler'] is not None) == (strategy == 'learned')
    finally:
        torch.set_num_threads(threads)
    check_matches(runs)
    assert runs[0]['epochs'][0]['train']['normalized_chamfer'] is None
    assert runs[1]['epochs'][0]['train']['normalized_chamfer'] is None
    assert runs[2]['epochs'][0]['train']['normalized_chamfer'] > 0
    assert runs[1]['epochs'][0]['train']['nonzero_sampler_gradient_batches'] == 1
    altered = copy.deepcopy(runs)
    altered[1]['epochs'][0]['train']['augmented_candidates_sha256'] = 'different'
    with pytest.raises(ValueError, match='Unmatched train'):
        check_matches(altered)


def test_coordinate_ablation_preserves_initial_weights_and_image_streams(tmp_path, monkeypatch):
    manifest, partition = synthetic_manifest(tmp_path)
    runs = []
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for mode in ('raw', 'local'):
            output = tmp_path / mode
            monkeypatch.setattr(sys, 'argv', ['train_sampling.py', '--manifest', str(manifest),
                                '--partition', str(partition), '--output-dir', str(output), '--device', 'cpu',
                                '--strategy', 'random', '--chamfer-weight', '0', '--proposal-coordinate-mode', mode])
            assert train_main() == 0
            run = json.loads((output / 'run.json').read_text())
            assert run['recipe']['proposal_coordinate_mode'] == mode
            runs.append(run)
        check_matches(runs)
    finally:
        torch.set_num_threads(threads)

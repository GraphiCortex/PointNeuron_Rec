import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pointneuron.models.query_pipeline import build_models, comparison_checkpoint, models_from_checkpoint, forward_proposals
from scripts.pointneuron2.compare_query_normalization import initialization_check, verify_baseline, check_control
from scripts.pointneuron2.train_query_comparison import run_arm, observe_fit, state_digest, check_pairs


@pytest.fixture(autouse=True, scope='module')
def few_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config():
    value = json.loads(Path('configs/pointneuron2/query_comparison.json').read_text())
    value.update(encoder_k=4, selected_points=8, target_capacity=8)
    value['decoder'].update(query_count=8, model_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0.)
    return value


def test_only_pooled_normalization_changes_and_old_checkpoints_keep_batchnorm():
    base = config()
    control, checked = initialization_check(base)
    assert 'pooled_normalization' not in base
    assert control == {**base, 'pooled_normalization': 'layernorm'}
    assert checked['unchanged_encoder_tensors'] == checked['query_head_initialization'] == 'exact'
    old_encoder, old_head = build_models(base, 'query', 0)
    encoder, head = build_models(control, 'query', 0)
    assert isinstance(old_encoder.global_mlp[1], nn.BatchNorm1d)
    assert isinstance(encoder.global_mlp[1], nn.LayerNorm)
    assert sum(isinstance(m, nn.LayerNorm) for m in encoder.modules()) == 1
    assert sum(p.numel() for p in encoder.parameters()) == sum(p.numel() for p in old_encoder.parameters())
    original, _ = models_from_checkpoint(comparison_checkpoint(old_encoder, old_head, base, 'query', 0))
    assert isinstance(original.global_mlp[1], nn.BatchNorm1d)
    restored, restored_head = models_from_checkpoint(comparison_checkpoint(encoder, head, control, 'query', 0))
    assert isinstance(restored.global_mlp[1], nn.LayerNorm)
    points, centers = torch.rand(2, 12, 4), torch.zeros(2, 3)
    encoder.eval(); head.eval()
    with torch.no_grad():
        expected = forward_proposals(head, 'query', points, encoder(points), centers, 48)
        actual = forward_proposals(restored_head, 'query', points, restored(points), centers, 48)
    for field in ('center_proposals', 'objectness_logits', 'radius'):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    with pytest.raises(ValueError, match='Pooled normalization'):
        build_models({**base, 'pooled_normalization': 'typo'}, 'query', 0)


def test_pooled_layernorm_is_independent_of_companion_crop_and_training_mode():
    control, _ = initialization_check(config())
    encoder, _ = build_models(control, 'query', 0)
    vectors = torch.rand(2, 192)
    with torch.no_grad():
        encoder.global_mlp.train(); training = encoder.global_mlp(vectors)
        encoder.global_mlp.eval(); paired = encoder.global_mlp(vectors)
        single = encoder.global_mlp(vectors[:1])
        modified = vectors.clone(); modified[1] += 100
        other_companion = encoder.global_mlp(modified)
    torch.testing.assert_close(training, paired, rtol=0, atol=0)
    torch.testing.assert_close(single[0], paired[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(other_companion[0], paired[0], rtol=0, atol=0)


def test_single_crop_observation_preserves_model_buffers_rng_and_modes():
    control, _ = initialization_check(config())
    encoder, head = build_models(control, 'query', 0)
    encoder.train(); head.train()
    points, centers = torch.rand(2, 12, 4), torch.zeros(2, 3)
    nodes = torch.tensor([[[1., 1, 1, 1, 1, -1]], [[1., 2, 1, 1, 1, -1]]])
    mask = torch.ones(2, 1, dtype=torch.bool)
    before = (state_digest(encoder), state_digest(head), torch.get_rng_state().clone())
    _, result = observe_fit(encoder, head, 'query', points, nodes, mask, centers, control, {}, inspect_single=True)
    assert encoder.training and head.training and all(m.training for m in encoder.modules())
    assert before[:2] == (state_digest(encoder), state_digest(head))
    torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)
    assert result['single_crop']['max_center_difference_from_pair'] < 1e-4
    assert len(result['matching']) == len(result['single_crop']['matching']) == 2


@pytest.fixture
def saved_baseline(tmp_path):
    cfg, rows = config(), []
    generator = np.random.default_rng(5)
    for i in range(2):
        path = tmp_path / f'train_{i}.npz'
        nodes = np.array([[1, 1+i, 1, 1, 1, -1]], np.float32)
        meta = {'seed': i, 'patch_center': [0, 0, 0]}
        np.savez(path, points=generator.random((12, 4)).astype(np.float32)*5, target_nodes=nodes,
                 reference_curve=nodes[:, 1:4], metadata=np.array(json.dumps(meta)))
        rows.append({'path': path.name, 'resolved_path': str(path), 'partition': 'train', 'target_count': 1})
    manifest = {'partition_sha256': 'synthetic', 'input_config': {'selected_points': 8}, 'medial_config': {}, 'records': rows}
    mp = tmp_path / 'manifest.json'; mp.write_text(json.dumps(manifest))
    args = SimpleNamespace(device='cpu', smoke=True, steps=2, eval_steps=[1, 2], manifest=mp, log_every=2)
    recipe = json.loads(Path(cfg['pointwise_recipe']).read_text())['training']
    folder = tmp_path / 'baseline'; folder.mkdir()
    reports = [run_arm(arm, args, manifest, cfg, recipe, folder/arm) for arm in ('pointwise', 'query')]
    check_pairs(reports)
    (folder/'comparison.json').write_text(json.dumps({'status': 'completed', 'paired_checks': 'passed', 'arms': reports}))
    return folder, mp, manifest, args, recipe


def test_actual_saved_baseline_and_layernorm_fit_are_matched(saved_baseline, tmp_path):
    folder, mp, manifest, args, recipe = saved_baseline
    baseline, control, milestones, checked = verify_baseline(folder, mp, manifest, 2, [1, 2], 'cpu')
    assert checked['baseline_encoder_sha256'] == baseline['initial_encoder_sha256']
    args.inspect_single = True
    report = run_arm('query', args, manifest, control, recipe, tmp_path/'layernorm')
    check_control(baseline, report)
    assert report['roundtrip'] == 'exact' and report['updates'] == 2
    changed = copy.deepcopy(report); changed['inputs_sha256'] = 'different'
    with pytest.raises(ValueError, match='inputs'):
        check_control(baseline, changed)
    changed = copy.deepcopy(report); del changed['evaluations'][0]['single_crop']
    with pytest.raises(ValueError, match='inference checks'):
        check_control(baseline, changed)
    changed = copy.deepcopy(report); changed['config']['lr'] *= 2
    with pytest.raises(ValueError, match='beyond pooled'):
        check_control(baseline, changed)


def test_baseline_preflight_rejects_wrong_budget_device_and_tampered_snapshot(saved_baseline):
    folder, mp, manifest, _, _ = saved_baseline
    for steps, device in ((3, 'cpu'), (2, 'cuda')):
        with pytest.raises(ValueError, match='budgets and devices'):
            verify_baseline(folder, mp, manifest, steps, [], device)
    with pytest.raises(ValueError, match='milestones'):
        verify_baseline(folder, mp, manifest, 2, [2], 'cpu')
    snapshot = folder/'query'/'step_000001.pt'
    snapshot.write_bytes(snapshot.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_baseline(folder, mp, manifest, 2, [], 'cpu')

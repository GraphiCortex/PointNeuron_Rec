"""Continuation scope, data order, full-state recovery and scientific verdicts."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import fit_consistent_node_objective as fit
from pn2.training import node_fit_state as state


def fake_dataset():
    ids = [f'train{i:02d}' for i in range(33)]
    eligible = [(i % 33, i) for i in range(460)]
    boxes = {(ids[ci], tile): {'target': {'gradient_training_allowed': True}} for ci, tile in eligible}
    boxes[('dev', 1)] = {'target': {'gradient_training_allowed': False}}
    boxes[(ids[0], 999)] = {'target': {'gradient_training_allowed': False}}
    schedule = np.array([[epoch, ci, tile] for epoch in range(20) for ci, tile in eligible])
    return SimpleNamespace(boxes=boxes, schedule=lambda: (ids, schedule)), schedule


def test_five_epochs_cover_all_supervised_boxes_without_unknowns_or_development():
    data, original = fake_dataset(); ids, plan = fit.make_schedule(data)
    assert len(ids) == 33 and np.array_equal(plan, original[:2300])
    for epoch in range(5):
        rows = plan[epoch*460:(epoch+1)*460]
        assert len(set(map(tuple, rows[:, 1:]))) == 460
        assert np.all(rows[:, 2] != 999)


def test_schedule_with_missing_box_or_corrupt_epoch_is_rejected():
    data, original = fake_dataset();original[459] = original[458]
    with pytest.raises(ValueError, match='exactly once'):fit.make_schedule(data)
    data, original = fake_dataset();original[0, 0] = 7
    with pytest.raises(ValueError, match='exactly once'):fit.make_schedule(data)


def checkpoint(tmp_path, method='baseline', stage=460, clock=9660):
    model = torch.nn.Linear(2, 1); opt = torch.optim.AdamW(model.parameters(), lr=.0001)
    model(torch.ones(1, 2)).sum().backward();opt.step()
    for value in opt.state.values():value['step'].fill_(clock)
    c = {'methods': ['baseline', 'consistent']};oc = {'checkpoint_sha256': 'origin'}
    binding = fit.bindings(c, oc, method)
    path = tmp_path/f'{method}_{stage}.pt';state.save(path, model, opt, binding, stage)
    return path, binding


def test_recovery_requires_correct_method_origin_cursor_and_adam_clocks(tmp_path):
    path, binding = checkpoint(tmp_path)
    restored = fit.resume_payload(path, binding)
    assert restored['step'] == 460
    with pytest.raises(ValueError, match='method/config'):
        fit.resume_payload(path, dict(binding, method='consistent'))
    bad, bad_binding = checkpoint(tmp_path, method='consistent', clock=9200)
    with pytest.raises(ValueError, match='Adam'):
        fit.resume_payload(bad, bad_binding)


def test_complete_stage_recovery_and_bad_hash(tmp_path):
    path, binding = checkpoint(tmp_path, stage=2300, clock=11500)
    assert fit.resume_payload(path, binding)['step'] == 2300
    with path.open('ab') as stream:stream.write(b'corrupted')
    with pytest.raises(ValueError, match='changed'):fit.resume_payload(path, binding)


def metric(p, r, f, curve):
    return {'precision': p, 'recall': r, 'f1': f, 'curve_support': curve}


def evaluation(values):
    p, r, f, curve = values.values()
    nodes = {'precision': p, 'recall': r, 'f1': f}
    return {'summary': {'neurons': [{'tolerances': {'2.0': {
        'full_grid': nodes, 'reference_present_only': nodes, 'curve_to_accepted_node_support': curve}}}]}}


def arms(a, b):
    return [{'method': m, 'evaluations': {p: evaluation(v) for p in ['training', 'development']}}
            for m, v in [('baseline', a), ('consistent', b)]]


def test_fewer_predictions_or_precision_gain_with_recall_loss_cannot_pass():
    origin = {p: metric(.1, .2, .13, .3) for p in ['training', 'development']}
    result = fit.comparison(arms(metric(.1, .2, .13, .3), metric(.8, .1, .18, .3)), origin)
    assert not result['candidate_screening_passed']
    assert result['consistent_minus_baseline']['development']['recall'] < 0


def test_relative_screening_requires_geometry_support_and_is_distinct_from_absolute_readiness():
    origin = {p: metric(.01, .01, .01, .1) for p in ['training', 'development']}
    paired = arms(metric(.1, .2, .13, .3), metric(.2, .3, .24, .4))
    result = fit.comparison(paired, origin)
    assert result['candidate_screening_passed']
    assert result['terminal_minus_origin']['development']['consistent']['recall'] == pytest.approx(.29)
    paired[1]['evaluations']['development'] = evaluation(metric(.2, .3, .24, .2))
    assert not fit.comparison(paired, origin)['candidate_screening_passed']


def test_count_strata_keeps_sparse_extras_in_precision_denominator():
    rows = [{'target_count': n, 'accepted_count': q,
             'tolerances': {'2.0': {'nodes': {'matched': m}}}} for n, q, m in [(8, 100, 4), (180, 200, 20), (0, 512, 0)]]
    result = fit.count_strata(rows)
    assert result[0]['pooled_precision_2'] == .04 and result[0]['pooled_recall_2'] == .5
    assert result[3]['boxes'] == 1 and result[3]['pooled_recall_2'] == pytest.approx(20/180)


def test_scientific_failure_does_not_skip_second_arm_but_execution_failure_does(monkeypatch, tmp_path):
    c = {'methods': ['baseline', 'consistent'], 'worker_timeout_seconds': 1}
    oc = {'checkpoint_sha256': 'origin', 'completed_fit': 'prior.json'}
    context = (c, oc, None, None, None, None, None, None, None)
    monkeypatch.setattr(fit, 'require_cpu', lambda c: None)
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    initial = {'model': 'same', 'adam': 'same', 'rng': 'same'}
    values = metric(.1, .2, .13, .3)
    prior = {'arms': [{'budget': 2048, 'evaluations': [
        {'step': 9200, 'summary': {'partitions': {p: {}}, **evaluation(values)['summary']}}
        for p in ('training', 'development')]}]}
    (tmp_path/'prior.json').write_text(json.dumps(prior))
    calls = []
    def launch(command, folder, timeout):
        method = folder.name;calls.append(method)
        report = {'status': 'completed', 'stage': 2300, 'method': method, 'initial_state': initial,
            'bindings': fit.bindings(c, oc, method), 'absolute_screening': {'broader_training_gate': False},
            'evaluations': {p: evaluation(values) for p in ('training', 'development')}}
        (folder/'worker_report.json').write_text(json.dumps(report));return 0, 'completed'
    monkeypatch.setattr(fit, 'launch_worker', launch)
    folder = tmp_path/'paired';folder.mkdir()
    args = SimpleNamespace(action='train', output=folder, config='fixture', checkpoint=None)
    assert fit.controller(context, args) == 0 and calls == ['baseline', 'consistent']
    monkeypatch.setattr(fit, 'launch_worker', lambda *args: (1, 'timeout'))
    folder = tmp_path/'stopped';folder.mkdir();args.output = folder
    assert fit.controller(context, args) == 1 and not (folder/'consistent').exists()


def test_completed_stage_recovery_never_calls_an_optimizer_update(monkeypatch, tmp_path):
    c = {'methods': ['baseline', 'consistent']}; oc = {'checkpoint_sha256': 'origin'}
    model = torch.nn.Linear(2, 1);opt = torch.optim.AdamW(model.parameters())
    payload = {'step': 2300}
    monkeypatch.setattr(fit, 'resume_payload', lambda *a: payload)
    monkeypatch.setattr(fit, 'validate_adam', lambda *a: None)
    monkeypatch.setattr(fit.candidate.common, 'runtime', lambda *a: None)
    monkeypatch.setattr(fit.candidate, 'build', lambda *a: (model, opt))
    monkeypatch.setattr(fit.candidate, 'step', lambda *a: pytest.fail('Completed stage must not update'))
    values = metric(.1, .2, .13, .3)
    monkeypatch.setattr(fit, 'evaluate_terminal', lambda *a: {p: evaluation(values) for p in ('training', 'development')})
    monkeypatch.setattr(fit.candidate.baseline, 'screening', lambda *a: {'fixture': True})
    context = (c, oc, {'optimizer': {'fixture': True}}, None, None, None, [], np.zeros((2300, 3)), None)
    args = SimpleNamespace(worker='baseline', output=tmp_path, checkpoint='complete.pt')
    result = fit.worker(context, args)
    assert result['new_updates_this_run'] == 0 and result['history_updates'] == 11500


@pytest.mark.parametrize('field,value', [('new_epochs', 6), ('new_updates', 2301),
    ('origin_updates', 9201), ('budget', 8192), ('automatic_extension', True)])
def test_config_cannot_silently_extend_or_change_the_agreed_comparison(tmp_path, field, value):
    c = {'schema_version': 1, 'methods': ['baseline', 'consistent'], 'new_epochs': 5,
         'updates_per_epoch': 460, 'new_updates': 2300, 'origin_updates': 9200,
         'budget': 2048, 'worker_timeout_seconds': 14400, 'automatic_extension': False,
         'source_sha256': {}}
    c[field] = value
    path = tmp_path/'config.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='agreed five-epoch'):
        fit.load_control(path)


def test_cpu_action_cannot_be_redirected_to_a_gpu_worker(monkeypatch):
    monkeypatch.setattr(fit, 'load_control', lambda *a: pytest.fail('Must stop before setup'))
    with pytest.raises(ValueError, match='CPU checks cannot'):
        fit.main(SimpleNamespace(action='cpucheck', worker='consistent'))

"""Protect denominator/failure handling and reuse of the completed neuron-21 pair."""
import math
from pathlib import Path

import pytest

from scripts.pointneuron2 import compare_validation_cutoffs as comparison

INDICES = [19, 21, 35, 44, 79, 94, 135]


def pair(index, position):
    result = {'sample_index': index, 'status': 'completed', 'results': []}
    for cutoff in [.85, .5]:
        precision = .1 + .1 * position
        recall = .9 - .1 * position + (.05 if cutoff == .5 else 0)
        result['results'].append({
            'cutoff': cutoff, 'metrics': {'spatial': {'precision': precision, 'recall': recall,
                         'f1': 2 * precision * recall / (precision + recall)}, 'topology': {'edge_f1': .5}},
            'reconstruction': {'bridge_edges': 1, 'foreground_cap_satisfied': index != 94}})
    return result


@pytest.mark.parametrize('failure_mode', ['missing', 'failed'])
def test_no_means_over_surviving_neurons(failure_mode):
    cases = [{'sample_index': i, 'status': 'completed'} for i in INDICES]
    results = {i: pair(i, n) for n, i in enumerate(INDICES)}
    if failure_mode == 'missing':
        results.pop(94)
    else:
        next(c for c in cases if c['sample_index'] == 94)['status'] = 'failed'
    assert comparison.complete_means(cases, results) is None


def test_cap_failure_remains_in_equal_neuron_mean():
    cases = [{'sample_index': i, 'status': 'completed'} for i in INDICES]
    results = {i: pair(i, n) for n, i in enumerate(INDICES)}
    mean = comparison.complete_means(cases, results)['0.85']
    expected = sum(r['results'][0]['metrics']['spatial']['f1'] for r in results.values()) / 7
    assert math.isclose(mean['f1'], expected)
    assert mean['neurons'] == 7 and mean['foreground_cap_failed_neurons'] == [94]
    assert not math.isclose(mean['f1'], 2 * mean['precision'] * mean['recall'] / (mean['precision'] + mean['recall']))


@pytest.mark.parametrize('fail_index', [None, 94])
def test_run_reuses_21_and_preserves_failed_case(tmp_path, monkeypatch, fail_index):
    cases = []
    for n, index in enumerate(INDICES):
        directory = tmp_path / f'neuron_{index:04d}'; directory.mkdir()
        path = directory / 'comparison.json'
        child = pair(index, n)
        if index != 21:
            child['status'] = 'prepared'
        comparison.write(path, child)
        case = {'sample_index': index, 'reused': index == 21, 'comparison': str(path),
                'status': 'completed' if index == 21 else 'prepared'}
        if index == 21:
            case['comparison_sha256'] = comparison.digest(path)
        cases.append(case)
    parent = {'status': 'prepared', 'cases': cases, 'checkpoint_sha256': 'shared-model',
              'source_sha256': {}, 'failures': [], 'means': None}
    comparison.write(tmp_path / 'validation_comparison.json', parent)
    monkeypatch.setattr(comparison, 'verify', lambda plan: None)
    monkeypatch.setattr(comparison, 'completed_pair', lambda path, checkpoint: comparison.read(path))
    attempted = []
    def fake_pair(directory, device):
        path = directory / 'comparison.json'; child = comparison.read(path)
        attempted.append(child['sample_index'])
        if child['sample_index'] == fail_index:
            raise RuntimeError('injected tracing failure')
        child['status'] = 'completed'; comparison.write(path, child)
    monkeypatch.setattr(comparison, 'run_pair', fake_pair)
    code = comparison.run(tmp_path, 'cpu')
    saved = comparison.read(tmp_path / 'validation_comparison.json')
    assert attempted == [19, 35, 44, 79, 94, 135]  # no rerun of neuron 21
    if fail_index is None:
        assert code == 0 and saved['status'] == 'completed' and saved['means']['0.5']['neurons'] == 7
    else:
        assert code == 1 and saved['status'] == 'failed_or_incomplete' and saved['means'] is None
        assert saved['failures'][0]['sample_index'] == 94
        assert next(c for c in saved['cases'] if c['sample_index'] == 135)['status'] == 'completed'

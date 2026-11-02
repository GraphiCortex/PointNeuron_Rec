"""Execution gates and unpadded short-box semantics; no GPU or real training."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import node_budget_timing as timing


def test_distinct_count_uses_actual_short_box_without_padding():
    assert timing.point_count(8192, 7650) == 7650
    assert timing.point_count(2048, 7650) == 2048
    assert timing.point_count(8192, 2000000) == 8192
    with pytest.raises(ValueError): timing.point_count(8192, 255)


@pytest.mark.parametrize('worker_state', ['timeout', 'completed'])
def test_failed_or_missing_worker_report_stops_without_launching_second_arm(monkeypatch, tmp_path, worker_state):
    monkeypatch.setattr(timing, 'require_cpu', lambda *a: None)
    monkeypatch.setattr(timing, 'load_records', lambda *a: [])
    monkeypatch.setattr(timing, 'identity', lambda *a: {'fixture': True})
    monkeypatch.setattr(timing, 'data_identity', lambda *a: 'fixture')
    calls = []
    def launch(*a): calls.append(a); return -1 if worker_state == 'timeout' else 0, worker_state
    monkeypatch.setattr(timing, 'launch_worker', launch)
    args = SimpleNamespace(output=str(tmp_path / 'run'), config='fixture.json')
    assert timing.controller({'budgets': [2048, 8192], 'gpu_worker_timeout_seconds': 600}, {}, args) == 1
    assert len(calls) == 1
    report = json.loads((Path(args.output) / 'report.json').read_text())
    assert report['automatic_retry'] is False and report['training_action_exists'] is False


def test_changed_input_manifest_cannot_reuse_cpu_gate(monkeypatch, tmp_path):
    proof = {'status': 'passed', 'reload_passed': True, 'packet_manifest_sha256': 'old'}
    (tmp_path / 'cpu.json').write_text(json.dumps(proof))
    monkeypatch.setattr(timing, 'ROOT', tmp_path)
    monkeypatch.setattr(timing, 'data_identity', lambda *a: 'changed')
    with pytest.raises(ValueError, match='CPU verification'):
        timing.require_cpu({'cpu_preflight': 'cpu.json'}, {})


def test_different_initial_parameters_fail_even_when_both_workers_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(timing, 'require_cpu', lambda *a: None)
    monkeypatch.setattr(timing, 'load_records', lambda *a: [])
    monkeypatch.setattr(timing, 'identity', lambda *a: {'fixture': True})
    monkeypatch.setattr(timing, 'data_identity', lambda *a: 'fixture')
    def launch(command, folder, timeout):
        budget = int(folder.name.removeprefix('points'))
        row = {'status': 'completed', 'budget': budget, 'updates': 8,
            'mode_consistency_passed': True, 'model_adam_rng_reload_passed': True,
            'packet_manifest_sha256': 'fixture', 'fixture': True,
            'initial_state_sha256': str(budget)}
        (folder / 'worker_report.json').write_text(json.dumps(row))
        return 0, 'completed'
    monkeypatch.setattr(timing, 'launch_worker', launch)
    args = SimpleNamespace(output=str(tmp_path / 'run'), config='fixture.json')
    assert timing.controller({'budgets': [2048, 8192], 'gpu_worker_timeout_seconds': 600}, {}, args) == 1
    report = json.loads((Path(args.output) / 'report.json').read_text())
    assert report['status'] == 'failed_initialization_mismatch'

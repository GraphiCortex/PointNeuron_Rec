"""External timeout/controller checks only; all worker launches are mocked."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import geometric_query_train_controller as launch


def setup_controller(monkeypatch, tmp_path):
    recipe = {"updates_per_arm": 2000, "train_worker_timeout_seconds": 3600}
    timing = tmp_path/"timing"; timing.mkdir()
    (timing/"report.json").write_text('{"status":"completed"}')
    args = SimpleNamespace(config="qualified.json", output=str(tmp_path/"train"), timing_run=str(timing), cpu_preflight=None)
    monkeypatch.setattr(launch.experiment, "load_recipe", lambda p: recipe)
    monkeypatch.setattr(launch.experiment, "require_cpu", lambda *a: None)
    monkeypatch.setattr(launch.experiment, "require_timing", lambda *a: {})
    monkeypatch.setattr(launch.experiment, "identity", lambda r: {"recipe_fingerprint": launch.common.fingerprint(r)})
    monkeypatch.setattr(launch.common, "load_tasks", lambda *a: None)
    return recipe, args


def test_larger_external_wall_limit_keeps_worker_recipe_and_updates_unchanged(monkeypatch, tmp_path):
    recipe, args = setup_controller(monkeypatch, tmp_path)
    before = copy.deepcopy(recipe)
    calls = []
    def worker(command, folder, timeout):
        calls.append((command, timeout))
        launch.common.write(folder/"worker_report.json", {"status": "completed", "stage": "train", "updates": 2000,
            "normalization_consistency_passed": True, "learnability_gate": False, **launch.experiment.identity(recipe)})
        return 0, "completed"
    monkeypatch.setattr(launch, "launch_worker", worker)
    assert launch.controller(args) == 0
    assert recipe == before
    command, timeout = calls[0]
    assert timeout == 5400
    assert command[2].endswith('geometric_query_experiment.py')
    assert command[command.index('--action')+1] == '_worker'
    assert command[command.index('--stage')+1] == 'train'
    saved = json.loads((Path(args.output)/"report.json").read_text())
    assert saved['learnability_gate'] is False
    assert saved['launch']['update_cap'] == 2000
    assert saved['launch']['legacy_controller_wall_timeout_seconds'] == 3600


def test_incompatible_timing_stops_before_creating_output_or_launching(monkeypatch, tmp_path):
    _, args = setup_controller(monkeypatch, tmp_path)
    def reject(*a):
        raise ValueError('stale timing')
    monkeypatch.setattr(launch.experiment, "require_timing", reject)
    monkeypatch.setattr(launch, "launch_worker", lambda *a: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='stale'):
        launch.controller(args)
    assert not Path(args.output).exists()


def test_worker_timeout_is_not_retried_or_reported_as_success(monkeypatch, tmp_path):
    _, args = setup_controller(monkeypatch, tmp_path)
    calls = []
    def timeout(*a):
        calls.append(a); return -1, 'timeout'
    monkeypatch.setattr(launch, "launch_worker", timeout)
    assert launch.controller(args) == 1
    assert len(calls) == 1
    report = json.loads((Path(args.output)/"report.json").read_text())
    assert report['status'] == 'timeout'


def test_zero_exit_without_complete_worker_report_is_failure(monkeypatch, tmp_path):
    _, args = setup_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(launch, "launch_worker", lambda *a: (0, 'completed'))
    assert launch.controller(args) == 1
    assert json.loads((Path(args.output)/"report.json").read_text())['status'] == 'failed_incomplete_control'

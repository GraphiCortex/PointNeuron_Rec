"""Controller failure handling without importing Torch or starting a GPU worker."""
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sampler_node_experiment as runner


def fixture(monkeypatch, tmp_path):
    recipe = {"methods": ["pn1_range_random", "pn2_sato"], "timing_updates_per_arm": 8,
              "updates_per_arm": 2000, "timing_worker_timeout_seconds": 1}
    args = SimpleNamespace(action="timing", config=str(tmp_path / "recipe.json"),
                           output=str(tmp_path / "out"), timing_run=None, cpu_preflight=None)
    monkeypatch.setattr(runner, "require_cpu_preflight", lambda *a: None)
    monkeypatch.setattr(runner.common, "load_tasks", lambda *a: None)
    monkeypatch.setattr(runner, "identity", lambda *a: {"fixture": True})
    return recipe, args


def test_missing_report_stops_and_preserves_declared_second_arm(monkeypatch, tmp_path):
    recipe, args = fixture(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(runner, "launch_worker", lambda *a: (calls.append(a) or (0, "completed")))
    assert runner.controller(recipe, args) == 1
    report = json.loads((Path(args.output) / "report.json").read_text())
    assert len(calls) == 1
    assert report["records"][0]["status"] == "failed_missing_report"
    assert report["records"][1]["status"] == "not_run_after_execution_failure"


def test_incomplete_success_report_is_rejected(monkeypatch, tmp_path):
    recipe, args = fixture(monkeypatch, tmp_path)
    def launch(command, folder, timeout):
        runner.common.write(folder / "report.json", {"method": folder.name, "status": "completed", "updates": 4})
        return 0, "completed"
    monkeypatch.setattr(runner, "launch_worker", launch)
    assert runner.controller(recipe, args) == 1
    report = json.loads((Path(args.output) / "report.json").read_text())
    assert report["records"][0]["status"] == "failed_incomplete_updates"


def test_completed_paired_timing_preserves_update_estimates(monkeypatch, tmp_path):
    recipe, args = fixture(monkeypatch, tmp_path)
    def launch(command, folder, timeout):
        runner.common.write(folder / "report.json", {"method": folder.name, "status": "completed", "updates": 8,
            "checkpoint_reload_passed": True, "estimated_update_minutes_for_one_arm": [10, 15]})
        return 0, "completed"
    monkeypatch.setattr(runner, "launch_worker", launch)
    assert runner.controller(recipe, args) == 0
    report = json.loads((Path(args.output) / "report.json").read_text())
    assert report["estimated_paired_update_minutes"] == [20, 30]


def test_timeout_kills_owned_worker_and_keeps_output(monkeypatch, tmp_path):
    class Process:
        stdout = io.StringIO("fixture output\n")
        returncode = -9
        killed = False
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return self.returncode
        def kill(self):
            self.killed = True
    process = Process()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: process)
    assert runner.launch_worker(["never executed"], tmp_path, 1) == (-9, "timeout")
    assert process.killed and (tmp_path / "console.log").read_text() == "fixture output\n"


def test_training_cannot_start_without_passed_timing():
    with pytest.raises(ValueError, match="timing"):
        runner.require_timing({}, None)

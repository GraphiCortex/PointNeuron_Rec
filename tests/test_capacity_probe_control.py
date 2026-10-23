"""CPU-only failure-path checks; never import Torch or start a GPU process."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/probe_pointnext_capacity.py"
spec = importlib.util.spec_from_file_location("capacity_probe", MODULE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def setup_args(tmp_path):
    return SimpleNamespace(output=str(tmp_path / "run"), config=str(tmp_path / "config.json"))


def test_missing_report_is_failure_and_larger_budgets_not_run(tmp_path):
    class Process:
        stdout = io.StringIO("fake worker\n")
        def wait(self, **kwargs):
            return 0
    config = {"budgets": [8192, 16384], "worker_timeout_seconds": 1}
    args = setup_args(tmp_path)
    with patch.object(probe.subprocess, "Popen", return_value=Process()) as launch:
        assert probe.sweep(args, config) == 1
    report = json.loads((Path(args.output) / "report.json").read_text())
    assert launch.call_count == 1
    assert report["rows"][0]["status"] == "failed_missing_report"
    assert report["rows"][1]["status"] == "not_run_after_failed_missing_report"


def test_timeout_kills_only_worker_and_keeps_console(tmp_path):
    class Process:
        stdout = io.StringIO("before timeout\n")
        returncode = -9
        killed = False
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake worker", timeout)
            return self.returncode
        def kill(self):
            self.killed = True
    process = Process()
    args = setup_args(tmp_path)
    with patch.object(probe.subprocess, "Popen", return_value=process):
        assert probe.sweep(args, {"budgets": [8192, 16384], "worker_timeout_seconds": 1}) == 1
    assert process.killed
    root = Path(args.output)
    assert (root / "8192/console.log").read_text() == "before timeout\n"
    assert json.loads((root / "report.json").read_text())["rows"][0]["status"] == "timeout"


def test_success_preserves_worker_measurements_for_each_budget(tmp_path):
    class Process:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO("completed worker\n")
            target = Path(command[command.index("--output")+1])
            probe.write(target / "report.json", {"points": int(target.name), "status": "completed",
                                                "modes": {"eval_forward": {"mean_seconds": .125}}})
        def wait(self, **kwargs):
            return 0
    args = setup_args(tmp_path)
    with patch.object(probe.subprocess, "Popen", side_effect=Process):
        assert probe.sweep(args, {"budgets": [8192, 16384], "worker_timeout_seconds": 1}) == 0
    report = json.loads((Path(args.output) / "report.json").read_text())
    assert [row["points"] for row in report["rows"]] == [8192, 16384]
    assert all(row["modes"]["eval_forward"]["mean_seconds"] == .125 for row in report["rows"])

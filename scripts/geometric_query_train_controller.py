"""Launch the unchanged, timing-qualified worker with a measured wall allowance.

The original worker/recipe, CPU evidence and CUDA prefix stay byte-identical.
Only the parent process's external wall limit changes from 60 to 90 minutes.
This controller cannot change model settings, update count, resume or retry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import sampler_node_common as common
import geometric_query_experiment as experiment
from sampler_node_experiment import launch_worker

TRAIN_WALL_TIMEOUT_SECONDS = 5400


def controller(args):
    recipe = experiment.load_recipe(args.config)
    experiment.require_cpu(recipe, args.cpu_preflight)
    experiment.require_timing(recipe, args.timing_run)
    common.load_tasks(recipe)
    if recipe["updates_per_arm"] != 2000:
        raise ValueError("This launcher cannot change the qualified 2000-update cap")
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    common.write(folder/"recipe.json", recipe)
    command = [sys.executable, "-B", str(Path(experiment.__file__).resolve()),
        "--action", "_worker", "--stage", "train", "--config", str(Path(args.config).resolve()),
        "--output", str(folder), "--timing-run", str(Path(args.timing_run).resolve())]
    if args.cpu_preflight:
        command += ["--cpu-preflight", str(Path(args.cpu_preflight).resolve())]
    launch = {"controller_sha256": common.digest(__file__),
        "powershell_launcher_sha256": common.digest(common.ROOT/"scripts/run_geometric_query_fit.ps1"),
        "recipe_fingerprint": common.fingerprint(recipe), "update_cap": 2000,
        "worker_command": command, "external_wall_timeout_seconds": TRAIN_WALL_TIMEOUT_SECONDS,
        "legacy_controller_wall_timeout_seconds": recipe["train_worker_timeout_seconds"],
        "timing_report_sha256": common.digest(Path(args.timing_run)/"report.json"),
        "reason": "Measured 41.6-62.4 minutes of updates plus evaluation/IO; original 60-minute wall limit is too close"}
    common.write(folder/"launch.json", launch)
    print("TRAIN LAUNCH: unchanged qualified worker/recipe; cap=2000 updates; external wall limit=90 minutes; no retry/resume", flush=True)
    code, status = launch_worker(command, folder, TRAIN_WALL_TIMEOUT_SECONDS)
    path = folder/"worker_report.json"
    report = json.loads(path.read_text()) if path.exists() else {"status": "failed_missing_report", "stage": "train", **experiment.identity(recipe)}
    if status != "completed":
        report["status"] = status
    elif (report.get("status") != "completed" or report.get("stage") != "train"
            or report.get("updates") != 2000 or not report.get("normalization_consistency_passed")
            or any(report.get(k) != v for k, v in experiment.identity(recipe).items())):
        report["status"] = "failed_incomplete_control"
    report.update(exit_code=code, launch=launch)
    common.write(folder/"report.json", report)
    print(f"FINISHED geometric train status={report['status']} learnability={report.get('learnability_gate')} report={folder/'report.json'}", flush=True)
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("train",), default="train")
    parser.add_argument("--config", default=str(experiment.CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-run", required=True)
    parser.add_argument("--cpu-preflight")
    return controller(parser.parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

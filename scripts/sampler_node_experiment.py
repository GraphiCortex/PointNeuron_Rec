"""Bounded node learnability comparison: CPU preflight, manual GPU timing/fit.

No automatic extension, retry, sampler adjustment, normalization switch or resume.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

import sampler_node_common as common

ROOT = common.ROOT


def verify_upstream(recipe):
    import probe_pointnext_capacity as probe
    config, _ = probe.load(ROOT / recipe["encoder_config"])
    if common.digest(ROOT / recipe["head_source"]) != recipe["head_source_sha256"]:
        raise ValueError("Historical head changed")
    return {"encoder_config": common.fingerprint(config), "head": recipe["head_source_sha256"]}


def identity(recipe):
    return {"recipe_fingerprint": common.fingerprint(recipe), "code_identity": common.code_identity(),
            "upstream_identity": verify_upstream(recipe)}


def require_cpu_preflight(recipe, path=None):
    report = json.loads((Path(path) if path else ROOT / recipe["cpu_preflight"]).read_text())
    if report["status"] != "passed" or any(report.get(k) != v for k, v in identity(recipe).items()):
        raise ValueError("CPU preflight missing, failed or stale; run -Action check with current source")


def require_timing(recipe, directory):
    if directory is None:
        raise ValueError("Training needs --timing-run pointing to a completed paired timing run")
    report = json.loads((Path(directory) / "report.json").read_text())
    if (report["stage"] != "timing" or report["status"] != "completed"
            or any(report.get(k) != v for k, v in identity(recipe).items())
            or [r["method"] for r in report["records"]] != recipe["methods"]
            or not all(r.get("checkpoint_reload_passed") and r["status"] == "completed" for r in report["records"])):
        raise ValueError("Timing report is not a passed current-recipe two-arm run")
    return report


def cpu_check(recipe, output):
    torch = common.runtime(recipe, "cpu")
    from pn2.models.node_objective import node_set_loss
    groups = common.load_tasks(recipe)
    model = common.make_model(recipe, "cpu")
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    rows = []
    for method, records in groups.items():
        for record in records:
            record = common.tensors(record, torch, "cpu")
            model.load_state_dict(initial, strict=True); model.train(); model.zero_grad(set_to_none=True)
            output_nodes = model(record)
            result = node_set_loss(output_nodes, record["target"], **recipe["loss"])
            result.total.backward()
            if not result.supervised or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Supervised CPU gradients failed")
            final = model.head.prediction[-1]
            if bool((final.weight.grad[2] != 0).any()) or bool(final.bias.grad[2] != 0):
                raise ValueError("Unsupervised radius output received direct loss gradients")
            if not bool((final.weight.grad[:2] != 0).any()) or not bool((final.weight.grad[3:] != 0).any()):
                raise ValueError("Confidence or position output has no supervised gradient")
            if any(not torch.equal(value, initial[key]) for key, value in model.named_parameters()):
                raise ValueError("CPU preflight changed parameters")
            row = {"method": method, "alias": record["metadata"]["alias"], "tile_id": record["metadata"]["tile_id"],
                   "targets": len(record["target"]), "distinct_points": record["metadata"]["distinct_points"],
                   "loss": float(result.total.detach()), "classification": float(result.classification),
                   "center": float(result.center), "all_gradients_finite": True,
                   "radius_row_gradient_zero": True, "parameters_unchanged": True}
            rows.append(row)
            print(f"CPU {method} {row['alias']}/{row['tile_id']} targets={row['targets']} loss={row['loss']:.5f} gradients=finite radius_row=zero", flush=True)
    common.write(output / "report.json", {"status": "passed", "scope": "Actual node loss backward, no optimizer or real-data fit",
        **identity(recipe), "torch": torch.__version__, "device": "cpu", "records": rows})


def prefix_identity(model, optimizer):
    return {"model_state_sha256": common.state_digest(model.state_dict()),
            "optimizer_state_sha256": common.nested_digest(optimizer.state_dict()),
            "rng_sha256": common.nested_digest(common.rng_state())}


def reload_probe(model, optimizer, recipe, method, step):
    import torch
    expected = prefix_identity(model, optimizer)
    stream = io.BytesIO()
    torch.save(common.checkpoint_payload(model, optimizer, step, recipe, method), stream)
    stream.seek(0)
    restored = common.make_model(recipe, "cuda:0")
    restored_optimizer = common.make_optimizer(restored, recipe)
    payload = torch.load(stream, map_location="cpu", weights_only=False)
    restored_step = common.restore_payload(payload, restored, restored_optimizer, recipe, method)
    if restored_step != step or prefix_identity(restored, restored_optimizer) != expected:
        raise ValueError("Exact model/Adam/RNG serialization reload failed")
    # No trained checkpoint is retained by timing. Use the restored state onward.
    return restored, restored_optimizer


def gpu_worker(recipe, args):
    require_cpu_preflight(recipe, args.cpu_preflight)
    paired_timing = require_timing(recipe, args.timing_run) if args.stage == "train" else None
    torch = common.runtime(recipe, "cuda:0")
    records = [common.tensors(r, torch, "cuda:0") for r in common.load_tasks(recipe)[args.method]]
    model = common.make_model(recipe, "cuda:0")
    optimizer = common.make_optimizer(model, recipe)
    folder = Path(args.output)
    device = torch.cuda.get_device_name(0)
    expected = next(r for r in paired_timing["records"] if r["method"] == args.method) if paired_timing else None
    if expected and (expected["device"] != device or expected["torch"] != torch.__version__):
        raise ValueError("Timing device/runtime differs; do not silently reuse prefix")
    cap = recipe["timing_updates_per_arm"] if args.stage == "timing" else recipe["updates_per_arm"]
    report = {"status": "running", "stage": args.stage, "method": args.method, "device": device,
              "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "cap": cap, "updates": 0, **identity(recipe), "evaluations": []}
    print(f"{args.stage.upper()} method={args.method} device={device} cap={cap} fresh_seed=0 ordinary_BN_inference", flush=True)
    torch.cuda.reset_peak_memory_stats()
    if args.stage == "train":
        common.save_checkpoint(folder / "checkpoint_step000000.pt", model, optimizer, 0, recipe, args.method)
    report["evaluations"].append(common.evaluate(model, records, recipe, folder, 0))
    durations = []
    started = time.perf_counter()
    with (folder / "updates.jsonl").open("x", encoding="utf-8") as journal:
        for step in range(1, cap+1):
            record = records[(step-1) % len(records)]
            torch.cuda.synchronize(); one = time.perf_counter()
            metrics = common.update(model, optimizer, record, recipe)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-one
            durations.append(elapsed)
            row = {"step": step, "alias": record["metadata"]["alias"], "tile_id": record["metadata"]["tile_id"],
                   "seconds": elapsed, **metrics}
            journal.write(json.dumps(row, allow_nan=False)+"\n"); journal.flush()
            report["updates"] = step
            if args.stage == "timing" or step % recipe["log_every"] == 0:
                print(f"UPDATE {step}/{cap} {row['alias']}/{row['tile_id']} loss={row['loss']:.5f} class={row['classification']:.5f} center={row['center']:.5f} seconds={elapsed:.3f}", flush=True)
            if step == recipe["timing_updates_per_arm"]:
                report["prefix_identity"] = prefix_identity(model, optimizer)
                if args.stage == "timing":
                    model, optimizer = reload_probe(model, optimizer, recipe, args.method, step)
                    report["checkpoint_reload_passed"] = True
                elif report["prefix_identity"] != expected["prefix_identity"]:
                    common.write(folder / "prefix_mismatch.json", {"expected": expected["prefix_identity"], "actual": report["prefix_identity"]})
                    raise ValueError("First eight updates do not exactly replay passed timing; stop")
                else:
                    print("PREFIX step=8 model/Adam/RNG exactly match passed timing", flush=True)
            if step in recipe["evaluation_steps"]:
                if args.stage == "train":
                    common.save_checkpoint(folder / f"checkpoint_step{step:06d}.pt", model, optimizer, step, recipe, args.method)
                report["evaluations"].append(common.evaluate(model, records, recipe, folder, step))
    report.update(status="completed", elapsed_after_initial_evaluation_seconds=time.perf_counter()-started,
                  update_seconds=durations, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  learnability_gate=report["evaluations"][-1]["learnability_gate"] if args.stage == "train" else None)
    if args.stage == "timing":
        # Exclude first four cold updates: one complete second cycle is measured.
        mean = float(np.mean(durations[4:]))
        report["estimated_update_minutes_for_one_arm"] = [mean*recipe["updates_per_arm"]*.8/60,
                                                         mean*recipe["updates_per_arm"]*1.2/60]
        print(f"ESTIMATE one arm update time {report['estimated_update_minutes_for_one_arm']}; excludes evaluations/checkpoints", flush=True)
    common.write(folder / "report.json", report)
    print(f"COMPLETED {args.stage} {args.method} gate={report['learnability_gate']} report={folder / 'report.json'}", flush=True)


def launch_worker(command, folder, timeout):
    """Owned child only; live stdout + log; explicit timeout and exit status."""
    with (folder / "console.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        def relay():
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end="", flush=True)
        reader = threading.Thread(target=relay, daemon=True); reader.start()
        try:
            code = process.wait(timeout=timeout)
            status = "completed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); code, status = process.returncode, "timeout"
        except BaseException:
            process.kill(); process.wait(); raise
        finally:
            reader.join()
    return code, status


def controller(recipe, args):
    require_cpu_preflight(recipe, args.cpu_preflight)
    if args.action == "train":
        require_timing(recipe, args.timing_run)
    common.load_tasks(recipe)  # Check all inputs/targets before any GPU worker.
    folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
    common.write(folder / "recipe.json", recipe)
    rows, stopped = [], False
    for method in recipe["methods"]:
        if stopped:
            rows.append({"method": method, "status": "not_run_after_execution_failure"})
            continue
        destination = folder / method; destination.mkdir()
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--action", "_worker",
                   "--stage", args.action, "--method", method, "--config", str(Path(args.config).resolve()),
                   "--output", str(destination)]
        if args.timing_run:
            command += ["--timing-run", str(Path(args.timing_run).resolve())]
        if args.cpu_preflight:
            command += ["--cpu-preflight", str(Path(args.cpu_preflight).resolve())]
        timeout = recipe[f"{args.action}_worker_timeout_seconds"]
        code, status = launch_worker(command, destination, timeout)
        report_path = destination / "report.json"
        row = json.loads(report_path.read_text()) if report_path.exists() else {"method": method, "status": "failed_missing_report"}
        if status != "completed":
            row["status"] = status
        if row.get("status") == "completed" and row.get("updates") != (recipe["timing_updates_per_arm"] if args.action == "timing" else recipe["updates_per_arm"]):
            row["status"] = "failed_incomplete_updates"
        row["exit_code"] = code; rows.append(row)
        stopped = row["status"] != "completed"
    result = {"stage": args.action, "status": "stopped" if stopped else "completed", **identity(recipe), "records": rows}
    if args.action == "train":
        result["learnability_gates"] = {r["method"]: r.get("learnability_gate") for r in rows}
    elif not stopped:
        result["estimated_paired_update_minutes"] = [sum(r["estimated_update_minutes_for_one_arm"][i] for r in rows) for i in (0, 1)]
        print(f"PAIRED UPDATE ESTIMATE minutes={result['estimated_paired_update_minutes']}; plus evaluation/checkpoint overhead", flush=True)
    common.write(folder / "report.json", result)
    print(f"FINISHED {args.action} status={result['status']} report={folder / 'report.json'}", flush=True)
    return 1 if stopped else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("cpucheck", "timing", "train", "_worker"), required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/sampler_node_fit.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-run")
    parser.add_argument("--cpu-preflight")
    parser.add_argument("--stage", choices=("timing", "train"))
    parser.add_argument("--method", choices=("pn1_range_random", "pn2_sato"))
    args = parser.parse_args()
    recipe = common.load_recipe(args.config)
    if args.action == "cpucheck":
        folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
        common.write(folder / "recipe.json", recipe); cpu_check(recipe, folder)
        return 0
    if args.action == "_worker":
        if args.stage is None or args.method is None:
            raise ValueError("Worker needs explicit stage/method")
        gpu_worker(recipe, args)
        return 0
    return controller(recipe, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

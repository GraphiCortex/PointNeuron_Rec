"""Matched class-normalization comparison; baseline sources remain unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from pointneuron.models.balanced_query_loss import POLICY, balanced_query_loss, loss_from_assignments
from pointneuron.models.query_fit_state import fit_checkpoint, restore_fit
from scripts.pointneuron2 import branched_query_experiment as base
from scripts.pointneuron2.compare_candidate_margin import compare_prediction

SCHEMA = "pointneuron2-balanced-query-comparison-v1"


def contract(args):
    spec = base.read(args.config)
    if (spec != base.read(ROOT/"configs/pointneuron2/balanced_query_comparison.json") or spec["schema"] != SCHEMA
            or spec["classification_policy"] != POLICY or spec["candidate_radii_for_reporting"] != [8., 12.]
            or spec["primary_candidate_radius"] != 8. or spec["candidate_neighbors"] != 16):
        raise ValueError("Comparison differs from the agreed loss-only experiment")
    baseline = ROOT/spec["baseline_experiment"]
    path = baseline/"model/report.json"
    if base.file_digest(path) != spec["baseline_report_sha256"]:
        raise ValueError("Completed baseline report changed")
    fit = base.read(path)
    recipe = base.recipe_at(ROOT/spec["baseline_recipe"])
    if fit["recipe"] != recipe or fit["status"] != "completed" or fit["optimizer_updates"] != 4000:
        raise ValueError("Expected completed matched baseline")
    base.verify_hashes(fit["provenance"]["source_files_sha256"])
    training_recipe = dict(recipe, classification_policy=POLICY)
    return spec, baseline, fit, recipe, training_recipe


def update(model, optimizer, record, recipe, device):
    model.train(); optimizer.zero_grad(set_to_none=True)
    prediction = base.forward(model, record, recipe, device)
    nodes = torch.from_numpy(record["nodes"])[None].to(device)
    result = balanced_query_loss(prediction, nodes, torch.ones(nodes.shape[:2], dtype=torch.bool, device=device), **recipe["query_loss"])
    if result.supervised_crops != 1 or not bool(torch.isfinite(result.total)):
        raise ValueError("Expected finite supervised balanced loss")
    result.total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError("Nonfinite updated parameter")
    return {"total": float(result.total.detach()), "classification": float(result.objectness),
            "center": float(result.center), "radius": float(result.radius), "gradient_norm_before_clip": float(norm)}


def evaluate(model, records, recipe, device, directory, step):
    # The unchanged evaluator supplies the SAME Hungarian cost/labels and primary radius-eight gates.
    evaluation = base.evaluate(model, records, recipe, device, directory, step)
    for i, (row, record) in enumerate(zip(evaluation["crops"], records)):
        with np.load(directory/f"step_{step:04d}_crop_{i}.npz", allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        output = SimpleNamespace(center_proposals=torch.from_numpy(saved["centers"]),
            radius=torch.from_numpy(saved["radii"]), objectness_logits=torch.from_numpy(saved["logits"]))
        nodes = torch.from_numpy(record["nodes"])[None]
        with torch.no_grad():
            result = loss_from_assignments(output, nodes, torch.ones(nodes.shape[:2], dtype=torch.bool),
                ((saved["pred_indices"].tolist(), saved["target_indices"].tolist()),), **recipe["query_loss"])
        row["reference_weighted_total_loss"] = row.pop("loss")
        row["reference_weighted_classification_loss"] = row.pop("classification_loss")
        row.update(optimization_total_loss=float(result.total), balanced_classification_loss=float(result.objectness))
        row["candidate_margin_comparison"] = compare_prediction(saved, record, recipe)
        if row["candidate_margin_comparison"][0]["connectivity_supervision"] != row["connectivity_supervision"]:
            raise ValueError("Primary candidate evaluation changed")
    evaluation["primary_gate_candidate_radius"] = 8.
    return evaluation


def snapshot(model, optimizer, records, recipe, device, directory, step, provenance, backend, report):
    path = directory/f"step_{step:04d}.pt"; temporary = path.with_suffix(".pt.writing")
    if path.exists() or temporary.exists():
        raise FileExistsError("Preserve checkpoint evidence")
    cp = fit_checkpoint(model, optimizer, step=step, recipe=recipe, provenance=provenance, backend=backend)
    cp["experiment_schema"] = SCHEMA
    torch.save(cp, temporary); temporary.replace(path)
    report.update(last_checkpoint=path.name, last_checkpoint_step=step, last_checkpoint_sha256=base.file_digest(path))
    base.write(directory/"report.json", report)
    report["evaluations"].append(evaluate(model, records, recipe, device, directory, step))
    base.write(directory/"report.json", report)


def run(args):
    spec, baseline, baseline_fit, original_recipe, recipe = contract(args)
    args.experiment_dir = args.experiment_dir.resolve()
    if not args.experiment_dir.is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Outputs belong under ignored tmp/pointneuron2")
    if args.action == "prepare":
        if args.device != "cpu" or args.experiment_dir.exists():
            raise ValueError("Preparation is CPU-only into a fresh folder")
        base.load_records(SimpleNamespace(experiment_dir=baseline), original_recipe)
        args.experiment_dir.mkdir(parents=True)
        shutil.copytree(baseline/"cache", args.experiment_dir/"cache")
        print(f"PREPARED byte-identical baseline cache: {args.experiment_dir}", flush=True)
        return
    if (args.action == "check" and args.device != "cpu") or (args.action != "check" and args.device != "cuda"):
        raise ValueError("Checks are CPU-only; timing and fitting are explicit CUDA actions")
    torch.set_num_threads(recipe["torch_threads"])
    manifest, records = base.load_records(args, original_recipe)
    manifest_path = args.experiment_dir/"cache/manifest.json"
    if base.file_digest(manifest_path) != baseline_fit["provenance"]["cache_manifest_sha256"]:
        raise ValueError("Comparison cache is not byte-identical to baseline")
    backend = base.backend_for(args.device)
    sources = base.source_hashes(SimpleNamespace(recipe=args.config))
    for path in (ROOT/spec["baseline_recipe"], ROOT/"scripts/pointneuron2/run_balanced_query.ps1"):
        sources[str(path)] = base.file_digest(path)
    provenance = {"cache_manifest_sha256": base.file_digest(manifest_path), "source_files_sha256": sources,
        "recipe_sha256": base.file_digest(args.config), "baseline_report_sha256": spec["baseline_report_sha256"], "experiment_schema": SCHEMA}
    directory = args.experiment_dir/("cpu_check" if args.action == "check" else "timing" if args.action == "timing" else "model")
    if args.action == "resume":
        if (args.checkpoint is None or not args.checkpoint.resolve().is_relative_to(args.experiment_dir)
                or not (args.checkpoint.parent.name == "model" or args.checkpoint.parent.name.startswith("resume_"))):
            raise ValueError("Explicit experiment checkpoint required")
        directory = args.experiment_dir/f"resume_{args.checkpoint.stem}"
    if directory.exists():
        raise FileExistsError("Preserve existing action outputs")
    if args.action in ("train", "resume"):
        timing = base.read(args.experiment_dir/"timing/report.json")
        if timing["status"] != "completed" or timing["backend"] != backend or timing["provenance"] != provenance:
            raise ValueError("Review matching comparison GPU timing first")
        if backend != baseline_fit["backend"]:
            raise ValueError("Paired comparison backend changed")
    directory.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "running", "action": args.action, "recipe": recipe, "comparison_spec": spec,
        "backend": backend, "provenance": provenance, "optimizer_updates": 0, "evaluations": [],
        "primary_gate_candidate_radius": 8., "scope": "four training crops; confidence normalization only; no GATv2/dev/test evaluation"}
    base.write(directory/"report.json", report); started = time.perf_counter()
    try:
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(0); torch.cuda.reset_peak_memory_stats()
        if args.action == "resume":
            cp = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if cp.get("experiment_schema") != SCHEMA:
                raise ValueError("Different experiment checkpoint")
            model, optimizer, start = restore_fit(cp, recipe=recipe, provenance=provenance, backend=backend, device=args.device)
            report.update(resumed_at_step=start, optimizer_updates=start, resumed_from=str(args.checkpoint), resumed_checkpoint_sha256=base.file_digest(args.checkpoint))
        else:
            model = base.PointNextCropNormQueryModel(0).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]); start = 0
        report["initial_model_sha256"] = {k: base.tensor_digest(v) for k, v in model.state_dict().items()}
        if args.action != "resume" and report["initial_model_sha256"] != baseline_fit["initial_model_sha256"]:
            raise ValueError("Fresh initialization differs from matched baseline")
        for path in sources:
            target = directory/"sources"/Path(path).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        print(f"ACTION={args.action} balanced confidence | {backend['device_name']} | cap=4000 | output={directory}", flush=True)
        if args.action == "check":
            snapshot(model, optimizer, records, recipe, "cpu", directory, 0, provenance, backend, report)
            cp = torch.load(directory/"step_0000.pt", map_location="cpu", weights_only=True)
            report["checkpoint_reload_max_delta"] = base.check_reload(model, cp, records, recipe, "cpu")
            restored, _, _ = restore_fit(cp, recipe=recipe, provenance=provenance, backend=backend, device="cpu")
            assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
            assert report["initial_model_sha256"] == {k: base.tensor_digest(v) for k, v in model.state_dict().items()}
        elif args.action == "timing":
            durations = []
            for j in range(8):
                base.sync(args.device); one = time.perf_counter()
                metrics = update(model, optimizer, records[j%4], recipe, args.device)
                base.sync(args.device); elapsed = time.perf_counter()-one
                report["throwaway_optimizer_updates"] = j+1
                if j >= 4: durations.append(elapsed)
                print(f"THROWAWAY {j+1}/8 crop={j%4} loss={metrics['total']:.5f} seconds={elapsed:.3f}", flush=True)
            one = time.perf_counter(); evaluate(model, records, recipe, args.device, directory, 8); base.sync(args.device)
            eval_seconds = time.perf_counter()-one
            cp = fit_checkpoint(model, optimizer, step=8, recipe=recipe, provenance=provenance, backend=backend)
            path = directory/"throwaway_state.pt"; torch.save(cp, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            report["checkpoint_reload_max_delta"] = base.check_reload(model, loaded, records, recipe, args.device)
            restore_fit(loaded, recipe=recipe, provenance=provenance, backend=backend, device=args.device)
            path.unlink()
            estimate = float(np.mean(durations))*4000 + eval_seconds*5
            report.update(throwaway_optimizer_updates=8, checkpoint_retained=False, timed_update_seconds=durations,
                estimated_fit_seconds=estimate, planning_range_seconds=[estimate, 1.5*estimate], baseline_actual_fit_seconds=baseline_fit["elapsed_seconds"])
            print(f"ESTIMATE {estimate/60:.1f}-{1.5*estimate/60:.1f} minutes plus IO/system variability; saved baseline took {baseline_fit['elapsed_seconds']/60:.1f} minutes.", flush=True)
        else:
            for step in range(start, 4001):
                if step > start:
                    metrics = update(model, optimizer, records[(step-1)%4], recipe, args.device); report["optimizer_updates"] = step
                    if step%50 == 0:
                        print(f"UPDATE={step}/4000 crop={(step-1)%4} loss={metrics['total']:.5f} balanced_class={metrics['classification']:.5f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                if step in recipe["snapshots"] and (args.action != "resume" or step > start):
                    snapshot(model, optimizer, records, recipe, args.device, directory, step, provenance, backend, report)
            final = directory/"step_4000.pt"
            cp = torch.load(final, map_location="cpu", weights_only=True)
            report.update(final_checkpoint_sha256=base.file_digest(final), checkpoint_reload_max_delta=base.check_reload(model, cp, records, recipe, args.device),
                          combined_learnability_gate_passed=report["evaluations"][-1]["combined_learnability_gate_passed"])
        base.verify_hashes(sources); base.verify_hashes(manifest["protected_files_sha256"])
        report.update(status="completed", elapsed_seconds=time.perf_counter()-started,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device == "cuda" else None,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if args.device == "cuda" else None,
            next_step="Review paired baseline/variant metrics at the fixed cap; no threshold search, promotion or GATv2 fit.")
        base.write(directory/"report.json", report)
        print(f"COMPLETED {args.action}; report={directory/'report.json'}", flush=True)
    except BaseException as exc:
        (directory/"error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), error_type=type(exc).__name__)
        base.write(directory/"report.json", report); raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("prepare", "check", "timing", "train", "resume"), required=True)
    p.add_argument("--config", type=Path, default=ROOT/"configs/pointneuron2/balanced_query_comparison.json")
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--checkpoint", type=Path)
    run(p.parse_args())

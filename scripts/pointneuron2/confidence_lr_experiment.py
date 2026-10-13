"""Matched learning-rate replay using the immutable completed confidence cache."""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from scripts.pointneuron2 import query_confidence_experiment as fixed
from pointneuron.models.frozen_query_confidence import FrozenQueryConfidence, confidence_checkpoint, load_confidence_checkpoint

BASE_RECIPE = ROOT / "configs/pointneuron2/query_confidence.json"
REPLAY_RECIPE = ROOT / "configs/pointneuron2/query_confidence_lr.json"


def recipe_contract(recipe, baseline):
    expected = dict(baseline, learning_rate=.001)
    if baseline["learning_rate"] != .0001 or recipe != expected:
        raise ValueError("Replay must change only learning_rate .0001 -> .001")
    return {"learning_rate": {"baseline":.0001,"replay":.001}}


def source_contract(args):
    # Read the original cache through the unchanged validated baseline loader.
    base_args = Namespace(recipe=BASE_RECIPE, experiment_directory=args.baseline_directory, device=args.device)
    old_recipe, manifest, cache, protected = fixed.load_cache(base_args)
    recipe = fixed.base.read_json(args.recipe)
    difference = recipe_contract(recipe, old_recipe)
    path = args.baseline_directory / "model/fit_report.json"
    baseline = fixed.base.read_json(path)
    if (baseline.get("status") != "completed" or baseline.get("optimizer_updates") != 2000
            or baseline.get("recipe") != old_recipe or baseline.get("trainable_parameters") != 258
            or baseline.get("cached_geometry_features_labels_unchanged") is not True
            or baseline.get("feature_manifest_sha256") != fixed.base.file_digest(args.baseline_directory / "cache/manifest.json")
            or [r["step"] for r in baseline["evaluations"]] != recipe["snapshots"]):
        raise ValueError("Expected the completed matched 2,000-update baseline")
    if any([r["crop"] for r in e["crops"]] != [0,1]
           or [r["targets"] for r in e["crops"]] != old_recipe["expected_positive_counts"] for e in baseline["evaluations"]):
        raise ValueError("Baseline snapshot crop/target identities differ")
    protected.update(baseline["source_files_sha256"])
    protected[str(path.resolve())] = fixed.base.file_digest(path)
    protected[str(args.recipe.resolve())] = fixed.base.file_digest(args.recipe)
    for step in recipe["snapshots"]:
        cp = args.baseline_directory / f"model/step_{step:04d}.pt"
        protected[str(cp.resolve())] = fixed.base.file_digest(cp)
    last = args.baseline_directory / "model/step_2000.pt"
    if fixed.base.file_digest(last) != baseline["final_checkpoint_sha256"]:
        raise ValueError("Baseline final checkpoint identity differs")
    fixed.base.protected_unchanged(protected)
    initial = torch.load(args.baseline_directory / "model/step_0000.pt",map_location="cpu",weights_only=True)
    loaded = load_confidence_checkpoint(initial)
    if (initial["step"] != 0 or initial["recipe"] != old_recipe
            or not torch.equal(loaded.linear.weight,cache["initial_weight"].cpu())
            or not torch.equal(loaded.linear.bias,cache["initial_bias"].cpu())):
        raise ValueError("Baseline initialization differs from immutable cached source weights")
    return recipe,manifest,cache,protected,baseline,difference


def paired_comparison(baseline, replay):
    if [r["step"] for r in baseline["evaluations"]] != [r["step"] for r in replay["evaluations"]]:
        raise ValueError("Paired snapshots differ")
    rows=[]
    for a,b in zip(baseline["evaluations"],replay["evaluations"]):
        if [r["crop"] for r in a["crops"]] != [r["crop"] for r in b["crops"]]:
            raise ValueError("Paired crop identities differ")
        for old,new in zip(a["crops"],b["crops"]):
            rows.append({"step":a["step"],"crop":old["crop"],"baseline":old,"replay":new,
                         "classification_loss_delta":new["weighted_classification_loss"]-old["weighted_classification_loss"],
                         "matched_accepted_delta":new["matched_accepted"]-old["matched_accepted"],
                         "unmatched_accepted_delta":new["unmatched_accepted"]-old["unmatched_accepted"]})
    return rows


def check_initial_metrics(baseline, evaluation):
    original=baseline["evaluations"][0]
    if evaluation["step"] != 0 or original["step"] != 0:
        raise ValueError("Expected step-zero comparison")
    if [r["crop"] for r in original["crops"]] != [0,1] or [r["crop"] for r in evaluation["crops"]] != [0,1]:
        raise ValueError("Initial paired crop set differs")
    for a,b in zip(original["crops"],evaluation["crops"]):
        for key in ("crop","targets","matched_accepted","unmatched_accepted","accepted_count","gate_passed"):
            if a[key] != b[key]:raise ValueError("Initial paired decisions differ")
        for key in ("weighted_classification_loss","matched_score_min","matched_score_mean","highest_unmatched_score",
                    "mean_matched_distance","max_matched_distance","accepted_curve_coverage_6"):
            if abs(a[key]-b[key])>1e-5:raise ValueError("Initial paired float32 metrics differ")


def run(args):
    # check deliberately loads CPU tensors and does not execute a head forward.
    if args.action == "check":
        args.device="cpu"
        _,_,_,_,_,difference=source_contract(args)
        print(f"Status: matched source checks passed; recipe difference={difference}; model forwards=0; optimizer updates=0")
        return
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; same-device comparison required")
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    recipe,manifest,cache,protected,baseline,difference=source_contract(args)
    backend={"device":args.device,"torch":str(torch.__version__),"cuda":torch.version.cuda,
             "device_name":torch.cuda.get_device_name() if args.device=="cuda" else "CPU","matmul_tf32":False}
    if backend!=baseline["backend"]:raise ValueError("Use the saved baseline backend/device for the paired replay")
    sources={str(p.resolve()):fixed.base.file_digest(p) for p in
             (Path(__file__),args.recipe,ROOT/"scripts/pointneuron2/run_confidence_lr.ps1")}
    protected.update(sources)
    before=fixed.tensor_digests(cache)
    output=args.experiment_directory
    if output.resolve()==args.baseline_directory.resolve():raise ValueError("Replay outputs must be separate from baseline")
    timing_path=output/"timing.json"
    if args.action=="preflight" and output.exists():raise FileExistsError("Preserve existing replay output directory")
    if args.action=="train":
        timing=fixed.base.read_json(timing_path)
        if (timing["status"]!="completed" or timing["backend"]!=backend or timing["source_files_sha256"]!=sources
                or timing["baseline_fit_sha256"]!=fixed.base.file_digest(args.baseline_directory/"model/fit_report.json")
                or timing["feature_manifest_sha256"]!=baseline["feature_manifest_sha256"]):
            raise ValueError("Expected matching reviewed timing/source/cache contract")
        if (output/"model").exists():raise FileExistsError("Preserve existing trained/failed replay")
        (output/"model").mkdir()
    else:
        output.mkdir(parents=True)
    head=FrozenQueryConfidence(cache["initial_weight"],cache["initial_bias"]).to(args.device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
    report={"schema":"pointneuron2-confidence-lr-replay-v1","status":"running","recipe":recipe,"backend":backend,
            "recipe_difference":difference,"source_files_sha256":sources,"trainable_parameters":258,"optimizer_updates":0,
            "baseline_fit_sha256":fixed.base.file_digest(args.baseline_directory/"model/fit_report.json"),
            "feature_manifest_sha256":baseline["feature_manifest_sha256"],"evaluations":[],"pointnext_transformer_forward_passes":0}
    print(f"Device: {backend['device_name']} | fresh confidence LR=.001 | same original weights/cache | baseline LR=.0001",flush=True)
    report_path=timing_path if args.action=="preflight" else output/"model/fit_report.json"
    fixed.base.write_json(report_path,report)
    started=time.perf_counter()
    provenance={"baseline_fit_sha256":report["baseline_fit_sha256"],"feature_manifest_sha256":report["feature_manifest_sha256"],"recipe_difference":difference}
    try:
        initial=fixed.evaluate(head,cache,manifest,recipe,0)
        check_initial_metrics(baseline,initial)
        report["initial_paired_metrics_passed"]=True
        if args.action=="preflight":
            durations=[]
            for step in range(4):
                fixed.base.sync(args.device);one=time.perf_counter();loss=fixed.update(head,optimizer,cache,recipe);fixed.base.sync(args.device)
                if step>=2:durations.append(time.perf_counter()-one)
                print(f"throwaway update={step+1}/4 confidence_loss={loss:.5f}",flush=True)
            one=time.perf_counter();fixed.evaluate(head,cache,manifest,recipe,4);evaluation_seconds=time.perf_counter()-one
            cp=confidence_checkpoint(head,step=4,recipe=recipe,provenance={**provenance,"timing_only":True})
            loaded=load_confidence_checkpoint(cp,args.device)
            with torch.no_grad():delta=float((head(cache["features"])-loaded(cache["features"])).abs().max())
            if delta>1e-5 or fixed.tensor_digests(cache)!=before:raise ValueError("Reload/frozen-input mismatch")
            fixed.base.protected_unchanged(protected)
            seconds=float(np.mean(durations))*recipe["updates"]+evaluation_seconds*len(recipe["snapshots"])
            report.update(status="completed",throwaway_optimizer_updates=4,checkpoint_retained=False,reload_max_logit_delta=delta,
                          estimated_fit_seconds=seconds,planning_range_seconds=[seconds,seconds*1.5+10],
                          frozen_cache_and_baseline_unchanged=True)
            fixed.base.write_json(report_path,report)
            print(f"Estimated replay: {seconds:.1f}s; planning range {seconds:.1f}-{seconds*1.5+10:.1f}s plus Python startup. Training remains separate.\nreport: {report_path}")
            return report
        model_output=output/"model"
        for step in range(recipe["updates"]+1):
            if step:
                loss=fixed.update(head,optimizer,cache,recipe);report["optimizer_updates"]=step
                if step%100==0:print(f"update={step}/2000 paired_confidence_loss={loss:.6f} elapsed={time.perf_counter()-started:.1f}s",flush=True)
            if step in recipe["snapshots"]:
                # Write detailed scores using exactly the same frozen-label evaluator.
                result=fixed.evaluate(head,cache,manifest,recipe,step,model_output)
                report["evaluations"].append(result)
                torch.save(confidence_checkpoint(head,step=step,recipe=recipe,provenance=provenance),model_output/f"step_{step:04d}.pt")
                for row in result["crops"]:
                    print(f"eval step={step} crop={row['crop']} matched={row['matched_accepted']}/{row['targets']} extras={row['unmatched_accepted']} curve6={row['accepted_curve_coverage_6']:.4f}",flush=True)
                fixed.base.write_json(report_path,report)
        path=model_output/"step_2000.pt"
        loaded=load_confidence_checkpoint(torch.load(path,map_location="cpu",weights_only=True),args.device)
        with torch.no_grad():delta=float((head(cache["features"])-loaded(cache["features"])).abs().max())
        if delta>1e-5 or fixed.tensor_digests(cache)!=before:raise ValueError("Reload/frozen-input mismatch")
        fixed.base.protected_unchanged(protected)
        report.update(status="completed",reload_max_logit_delta=delta,frozen_cache_and_baseline_unchanged=True,
                      final_checkpoint_sha256=fixed.base.file_digest(path),gate_passed=report["evaluations"][-1]["gate_passed"],
                      elapsed_seconds=time.perf_counter()-started)
        comparison={"status":"completed","paired_checks_passed":True,"recipe_difference":difference,
                    "baseline_fit_sha256":report["baseline_fit_sha256"],"rows":paired_comparison(baseline,report),
                    "baseline_gate_passed":baseline["gate_passed"],"replay_gate_passed":report["gate_passed"]}
        fixed.base.write_json(report_path,report);fixed.base.write_json(output/"comparison.json",comparison)
        print(f"Status: completed; paired checks/reload passed; confidence gate: {report['gate_passed']}\nreport: {output/'comparison.json'}")
        return report
    except Exception as exc:
        report.update(status="failed",error=str(exc));fixed.base.write_json(report_path,report);raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action",choices=["check","preflight","train"],required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--recipe",type=Path,default=REPLAY_RECIPE)
    parser.add_argument("--baseline-directory",type=Path,default=ROOT/"tmp/pointneuron2/local_runs/query_confidence_20261013")
    parser.add_argument("--experiment-directory",type=Path,default=ROOT/"tmp/pointneuron2/local_runs/query_confidence_lr_20261014")
    run(parser.parse_args())

"""Separate CPU feature preparation, timing and bounded confidence-only fit."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from torch.nn import functional as F
from scripts.pointneuron2 import pointnext_query_experiment as base
from pointneuron.models.pointnext_query_cropnorm import load_cropnorm_checkpoint
from pointneuron.models.frozen_query_confidence import FrozenQueryConfidence, fixed_confidence_loss, confidence_checkpoint, load_confidence_checkpoint, SCHEMA

CACHE_SCHEMA = "pointneuron2-frozen-confidence-cache-v1"


def load_recipe(path):
    recipe = base.read_json(path)
    if (recipe != base.read_json(ROOT / "configs/pointneuron2/query_confidence.json") or recipe["schema"] != SCHEMA
            or recipe["training_crops"] != [[0,0],[0,1]] or recipe["expected_positive_counts"] != [6,12]
            or recipe["updates"] != 2000 or recipe["snapshots"] != [0,100,500,1000,2000]
            or recipe["update_scope"] != "mean_of_two_cached_crop_losses"):
        raise ValueError("Recipe differs from the declared fixed-feature confidence control")
    return recipe


def cache_tensors(model, records, source_recipe):
    """Capture final-layer inputs; reference labels never enter the forward pass."""
    captured = []
    handle = model.proposal.prediction[-1].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().clone()))
    predictions, fixed, metadata = [], [], []
    try:
        model.eval().requires_grad_(False)
        for record in records:
            with torch.no_grad():
                prediction = base.forward(model, record)
                loss = base.query_set_loss(prediction, record["nodes"], record["mask"], **source_recipe["query_loss"])
            predicted, target = loss.assignments[0]
            labels = torch.zeros(512, dtype=torch.long)
            labels[list(predicted)] = 1
            fixed.append(labels); predictions.append(prediction)
            metadata.append({**record["metadata"], "assigned_queries": list(predicted), "assigned_targets": list(target),
                             "nodes": record["nodes"][0].tolist(), "geometry": base.pack_geometry(record["geometry"])})
    finally:
        handle.remove()
    final = model.proposal.prediction[-1]
    tensors = {"features": torch.cat(captured), "labels": torch.stack(fixed),
               "centers": torch.cat([p.center_proposals.cpu() for p in predictions]),
               "radii": torch.cat([p.radius.cpu() for p in predictions]),
               "baseline_logits": torch.cat([p.objectness_logits.cpu() for p in predictions]),
               "initial_weight": final.weight[:2].detach().cpu().clone(),
               "initial_bias": final.bias[:2].detach().cpu().clone()}
    delta = float((F.linear(tensors["features"], tensors["initial_weight"], tensors["initial_bias"])
                   - tensors["baseline_logits"]).abs().max())
    if delta > 1e-5 or any(t.requires_grad for t in tensors.values()):
        raise ValueError("Detached confidence feature replay differs from original logits")
    return tensors, metadata, delta


def prepare(args):
    recipe = load_recipe(args.recipe)
    cache_dir = args.experiment_directory / "cache"
    if cache_dir.exists():
        raise FileExistsError("Preserve existing feature caches; choose a fresh experiment")
    source_args = base.parser().parse_args([
        "--action", "preflight", "--device", "cpu", "--experiment-dir", str(args.source_directory),
        "--native-manifest", str(ROOT / "tmp/pointneuron2/local_runs/field_pilot_20261012/cache/manifest.json"),
        "--field-models", str(ROOT / "tmp/pointneuron2/local_runs/field_attention_20261013/baseline/model"),
        "--data-root", str(ROOT.parent / "data/gold166")])
    source_recipe, manifest, records, protected = base.load_cache(source_args)
    comparison_path = args.control_directory / "comparison.json"
    checkpoint_path = args.control_directory / "cropnorm_step_2000.pt"
    comparison = base.read_json(comparison_path)
    if (comparison["status"] != "completed" or comparison["optimizer_updates"] != 0
            or comparison["converted_checkpoint_sha256"] != base.file_digest(checkpoint_path)
            or comparison["unchanged_learned_parameters"] is not True):
        raise ValueError("Expected completed saved-weight per-crop normalization control")
    protected.update(comparison["protected_files_sha256"])
    protected.update(comparison["control_source_sha256"])
    protected.update({str(comparison_path.resolve()): base.file_digest(comparison_path),
                      str(checkpoint_path.resolve()): base.file_digest(checkpoint_path),
                      str(args.recipe.resolve()): base.file_digest(args.recipe)})
    base.protected_unchanged(protected)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != 2000 or checkpoint["provenance"]["optimizer_updates"] != 0:
        raise ValueError("Expected source-step 2000 conversion without retraining")
    torch.set_num_threads(2)
    model = load_cropnorm_checkpoint(checkpoint)
    tensors, rows, delta = cache_tensors(model, records, source_recipe)
    final_rows = {r["crop"]: r for r in comparison["rows"] if r["step"] == 2000}
    for i, row in enumerate(rows):
        old = {r["query"]: r["matched_target"] for r in final_rows[i]["matched_queries"]}
        new = dict(zip(row["assigned_queries"], row["assigned_targets"]))
        if old != new or len(old) != recipe["expected_positive_counts"][i]:
            raise ValueError("Source assignments changed; inspect before substituting labels")
    base.protected_unchanged(protected)
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "features.pt"
    torch.save({"schema": CACHE_SCHEMA, **tensors}, cache_path)
    result = {"schema": CACHE_SCHEMA, "status": "completed", "recipe": recipe, "records": rows,
              "features_sha256": base.file_digest(cache_path), "source_checkpoint_sha256": base.file_digest(checkpoint_path),
              "protected_files_sha256": protected, "cpu_model_forward_passes": 2, "optimizer_updates": 0,
              "confidence_replay_max_logit_delta": delta, "trainable_parameters": 258,
              "next_step": "Separate preflight; no fit starts automatically."}
    base.write_json(cache_dir / "manifest.json", result)
    print(f"Status: prepared; CPU frozen forwards=2; optimizer updates=0; positives={[int(r.sum()) for r in tensors['labels']]}\nmanifest: {cache_dir/'manifest.json'}")
    return result


def tensor_digests(cache):
    return {k: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,t in cache.items()}


def load_cache(args):
    recipe = load_recipe(args.recipe)
    path = args.experiment_directory / "cache/manifest.json"
    manifest = base.read_json(path)
    cache_path = path.parent / "features.pt"
    if (manifest["schema"] != CACHE_SCHEMA or manifest["status"] != "completed" or manifest["recipe"] != recipe
            or manifest["optimizer_updates"] != 0 or base.file_digest(cache_path) != manifest["features_sha256"]
            or [(r["sample_index"],r["patch_index"],r["partition"]) for r in manifest["records"]] != [(0,0,"train"),(0,1,"train")]):
        raise ValueError("Expected immutable two-training-crop confidence cache")
    base.protected_unchanged(manifest["protected_files_sha256"])
    saved = torch.load(cache_path, map_location="cpu", weights_only=True)
    if saved.pop("schema") != CACHE_SCHEMA:
        raise ValueError("Feature cache schema differs")
    expected = {"features":(2,512,128), "labels":(2,512), "centers":(2,512,3), "radii":(2,512,1),
                "baseline_logits":(2,512,2), "initial_weight":(2,128), "initial_bias":(2,)}
    if set(saved) != set(expected):
        raise ValueError("Unexpected cache tensors")
    for name,t in saved.items():
        if t.shape != expected[name] or t.dtype != (torch.long if name == "labels" else torch.float32) or not bool(torch.isfinite(t).all()):
            raise ValueError("Invalid fixed-feature cache dimensions/precision")
    for i,row in enumerate(manifest["records"]):
        label = torch.zeros(512,dtype=torch.long); label[row["assigned_queries"]] = 1
        if not torch.equal(label,saved["labels"][i]) or int(label.sum()) != recipe["expected_positive_counts"][i]:
            raise ValueError("Fixed assignments/labels differ")
    if bool((saved["radii"] <= 0).any()):
        raise ValueError("Invalid frozen radii")
    protected = dict(manifest["protected_files_sha256"])
    protected.update({str(path.resolve()):base.file_digest(path),str(cache_path.resolve()):base.file_digest(cache_path)})
    return recipe,manifest,{k:t.to(args.device) for k,t in saved.items()},protected


def evaluate(head, cache, manifest, recipe, step, output=None):
    head.eval()
    with torch.no_grad():
        logits = head(cache["features"])
        total, losses = fixed_confidence_loss(logits,cache["labels"],recipe)
    scores = logits.softmax(-1)[...,1].cpu().double().numpy()
    centers = cache["centers"].cpu().double().numpy()
    rows = []
    for i,record in enumerate(manifest["records"]):
        q,t = record["assigned_queries"],record["assigned_targets"]
        distances = np.linalg.norm(centers[i,q]-np.asarray(record["nodes"])[t,1:4],axis=1)
        positive = cache["labels"][i].cpu().numpy().astype(bool)
        accepted = scores[i] >= recipe["score_threshold"]
        correct, extras = int((positive & accepted).sum()),int((~positive & accepted).sum())
        support = base.measure_support(centers[i,accepted],base.unpack_geometry(record["geometry"]))
        coverage = support["reference_to_points"]["within_voxels"]["6"]
        row = {"crop":i,"weighted_classification_loss":float(losses[i]),"targets":int(positive.sum()),
               "matched_accepted":correct,"unmatched_accepted":extras,"accepted_count":int(accepted.sum()),
               "matched_score_min":float(scores[i,positive].min()),"matched_score_mean":float(scores[i,positive].mean()),
               "highest_unmatched_score":float(scores[i,~positive].max()),
               "mean_matched_distance":float(distances.mean()),"max_matched_distance":float(distances.max()),
               "accepted_curve_coverage_6":coverage,
               "gate_passed":correct==int(positive.sum()) and extras==0 and distances.max()<=1 and coverage>=.95}
        rows.append(row)
        if output:
            with (output/f"step_{step:04d}_crop_{i}_scores.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.writer(f);writer.writerow(["query","score","fixed_positive","accepted"])
                writer.writerows((j,float(scores[i,j]),bool(positive[j]),bool(accepted[j])) for j in range(512))
    return {"step":step,"mean_weighted_confidence_loss":float(total),"crops":rows,"gate_passed":all(r["gate_passed"] for r in rows)}


def update(head, optimizer, cache, recipe):
    head.train();optimizer.zero_grad(set_to_none=True)
    loss,_ = fixed_confidence_loss(head(cache["features"]),cache["labels"],recipe)
    if not bool(torch.isfinite(loss)):raise ValueError("Nonfinite confidence loss")
    loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),recipe["gradient_clip"],error_if_nonfinite=True);optimizer.step()
    return float(loss.detach())


def source_files(args):
    return {str(p.resolve()):base.file_digest(p) for p in
        (Path(__file__),args.recipe,ROOT/"src/pointneuron/models/frozen_query_confidence.py",ROOT/"scripts/pointneuron2/run_query_confidence.ps1")}


def run(args):
    if args.action == "prepare":return prepare(args)
    if args.device == "cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; choose CPU explicitly")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    recipe,manifest,cache,protected = load_cache(args)
    sources = source_files(args); protected.update(sources)
    before = tensor_digests(cache)
    backend = {"device":args.device,"torch":str(torch.__version__),"cuda":torch.version.cuda,
               "device_name":torch.cuda.get_device_name() if args.device=="cuda" else "CPU","matmul_tf32":False}
    timing_path = args.experiment_directory/"timing.json"
    if args.action == "preflight" and timing_path.exists():raise FileExistsError("Preserve existing timing")
    output = args.experiment_directory/"model"
    if args.action == "train":
        timing=base.read_json(timing_path)
        if (timing["status"]!="completed" or timing["backend"]!=backend or timing["source_files_sha256"]!=sources
                or timing["feature_manifest_sha256"]!=base.file_digest(args.experiment_directory/"cache/manifest.json")):
            raise ValueError("Expected matching reviewed timing, code and feature cache")
        if output.exists():raise FileExistsError("Preserve existing fitted/failed outputs")
        output.mkdir()
    head=FrozenQueryConfidence(cache["initial_weight"],cache["initial_bias"]).to(args.device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
    report={"schema":SCHEMA,"status":"running","recipe":recipe,"backend":backend,"source_files_sha256":sources,
            "feature_manifest_sha256":base.file_digest(args.experiment_directory/"cache/manifest.json"),
            "trainable_parameters":sum(p.numel() for p in head.parameters()),"optimizer_updates":0,"evaluations":[],
            "pointnext_transformer_forward_passes":0,"cached_crops_per_update":2}
    print(f"Device: {backend['device_name']} | confidence parameters=258 | fixed geometry and assignments | both cached crops per update",flush=True)
    started=time.perf_counter()
    if args.action == "preflight":
        durations=[]
        for step in range(4):
            base.sync(args.device);one=time.perf_counter();loss=update(head,optimizer,cache,recipe);base.sync(args.device)
            if step>=2:durations.append(time.perf_counter()-one)
            print(f"throwaway update={step+1}/4 confidence_loss={loss:.5f}",flush=True)
        one=time.perf_counter();evaluate(head,cache,manifest,recipe,4);eval_time=time.perf_counter()-one
        cp=confidence_checkpoint(head,step=4,recipe=recipe,provenance={"timing_only":True})
        loaded=load_confidence_checkpoint(cp,args.device)
        with torch.no_grad():delta=float((head(cache["features"])-loaded(cache["features"])).abs().max())
        if delta>1e-5 or tensor_digests(cache)!=before:raise ValueError("Reload or frozen-cache check failed")
        base.protected_unchanged(protected)
        seconds=float(np.mean(durations))*recipe["updates"]+eval_time*len(recipe["snapshots"])
        report.update(status="completed",throwaway_optimizer_updates=4,checkpoint_retained=False,reload_max_logit_delta=delta,
                      estimated_fit_seconds=seconds,planning_range_seconds=[seconds,seconds*1.5+10],
                      cached_geometry_features_labels_unchanged=True)
        base.write_json(timing_path,report)
        print(f"Estimated fixed-feature fit: {seconds:.1f} seconds; planning range {seconds:.1f}-{seconds*1.5+10:.1f}, plus Python startup. No checkpoint retained.\nreport: {timing_path}")
        return report
    report_path=output/"fit_report.json";base.write_json(report_path,report)
    try:
        for step in range(recipe["updates"]+1):
            if step:
                loss=update(head,optimizer,cache,recipe);report["optimizer_updates"]=step
                if step%100==0:print(f"update={step}/2000 paired_confidence_loss={loss:.6f} elapsed={time.perf_counter()-started:.1f}s",flush=True)
            if step in recipe["snapshots"]:
                result=evaluate(head,cache,manifest,recipe,step,output);report["evaluations"].append(result)
                torch.save(confidence_checkpoint(head,step=step,recipe=recipe,provenance={"feature_manifest_sha256":report["feature_manifest_sha256"]}),output/f"step_{step:04d}.pt")
                for row in result["crops"]:
                    print(f"eval step={step} crop={row['crop']} loss={row['weighted_classification_loss']:.5f} matched={row['matched_accepted']}/{row['targets']} extras={row['unmatched_accepted']} curve6={row['accepted_curve_coverage_6']:.4f}",flush=True)
                base.write_json(report_path,report)
        path=output/"step_2000.pt"
        loaded=load_confidence_checkpoint(torch.load(path,map_location="cpu",weights_only=True),args.device)
        with torch.no_grad():delta=float((head(cache["features"])-loaded(cache["features"])).abs().max())
        if delta>1e-5 or tensor_digests(cache)!=before:raise ValueError("Reload or frozen-cache check failed")
        base.protected_unchanged(protected)
        report.update(status="completed",reload_max_logit_delta=delta,cached_geometry_features_labels_unchanged=True,
                      final_checkpoint_sha256=base.file_digest(path),elapsed_seconds=time.perf_counter()-started,
                      gate_passed=report["evaluations"][-1]["gate_passed"],
                      next_step="Review confidence capacity and fixed-placement limits; no automatic full fit or promotion.")
        base.write_json(report_path,report)
        print(f"Status: completed; frozen cache and reload passed; confidence gate: {report['gate_passed']}\nreport: {report_path}")
        return report
    except Exception as exc:
        report.update(status="failed",error=str(exc));base.write_json(report_path,report);raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action",choices=["prepare","preflight","train"],required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--recipe",type=Path,default=ROOT/"configs/pointneuron2/query_confidence.json")
    parser.add_argument("--experiment-directory",type=Path,default=ROOT/"tmp/pointneuron2/local_runs/query_confidence_20261013")
    parser.add_argument("--source-directory",type=Path,default=ROOT/"tmp/pointneuron2/local_runs/pointnext_query_20261013")
    parser.add_argument("--control-directory",type=Path,default=ROOT/"tmp/pointneuron2/local_runs/pointnext_cropnorm_20261013")
    run(parser.parse_args())

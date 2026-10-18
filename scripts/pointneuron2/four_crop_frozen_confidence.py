"""Separate CPU preparation/check and a capped, frozen four-crop confidence stage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"),str(ROOT)]
from pointneuron.models.four_crop_frozen_confidence import FourCropFrozenConfidence, stage_loss, SCHEMA
from pointneuron.models.query_fit_state import cpu_snapshot, capture_rng, restore_rng
from scripts.pointneuron2 import branched_query_experiment as base
from scripts.pointneuron2.audit_four_crop_confidence_capacity import reconstruct_inputs

CACHE_SCHEMA = SCHEMA+"-cache"


def load_recipe(path):
    recipe = base.read(path)
    if (recipe != base.read(ROOT/"configs/pointneuron2/four_crop_frozen_confidence.json") or recipe["schema"] != SCHEMA
            or recipe["updates"] != 4000 or recipe["expected_positive_counts"] != [6,12,16,418]
            or recipe["update_scope"] != "mean_of_four_per_crop_balanced_CE_losses_times_two"
            or recipe["device"] != "cpu" or recipe["score_threshold"] != .5):
        raise ValueError("Stage recipe differs from the declared frozen four-crop control")
    return recipe


def fingerprints(tensors):
    return {k:base.tensor_digest(v) for k,v in tensors.items()}


def prepare(args,recipe):
    if args.directory.exists():
        raise FileExistsError("Preserve existing stage outputs")
    source = ROOT/recipe["source_experiment"]
    fit = base.read(source/"model/report.json")
    if base.file_digest(source/"model/report.json") != recipe["source_fit_sha256"]:
        raise ValueError("Balanced fit report changed")
    checkpoint_path = source/"model/step_4000.pt"
    if base.file_digest(checkpoint_path) != recipe["source_checkpoint_sha256"]:
        raise ValueError("Balanced model checkpoint changed")
    original = base.recipe_at(ROOT/"configs/pointneuron2/branched_query_fit.json")
    manifest,records = base.load_records(SimpleNamespace(experiment_dir=source),original)
    protected = {**manifest["protected_files_sha256"],**fit["provenance"]["source_files_sha256"],
        str(source/"model/report.json"):recipe["source_fit_sha256"],str(checkpoint_path):recipe["source_checkpoint_sha256"],
        str(source/"cache/manifest.json"):base.file_digest(source/"cache/manifest.json")}
    reference = base.read(ROOT/"docs/pointneuron2/hard_negative_query_results_seed0/summary.json")["balanced"]
    base.verify_hashes(protected)
    cp = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if cp["step"] != 4000 or cp["provenance"] != fit["provenance"]:
        raise ValueError("Invalid source checkpoint")
    state = cp["inference_checkpoint"]["model"]
    groups = {k:[] for k in ("features","labels","centers","radii","baseline_logits","decoder_features")}
    tensor_cache,rows = {},[]
    for i,record in enumerate(records):
        path = source/f"model/step_4000_crop_{i}.npz"
        protected[str(path)] = reference["prediction_files_sha256"][path.name]
        if base.file_digest(path) != protected[str(path)]:
            raise ValueError("Frozen predictions changed")
        with np.load(path,allow_pickle=False) as z:
            saved = {k:z[k].copy() for k in z.files}
        with torch.no_grad():
            hidden,raw = reconstruct_inputs(torch.from_numpy(saved["decoder_features"]),state)
        logits = torch.from_numpy(saved["logits"])
        torch.testing.assert_close(raw[...,:2],logits,rtol=1e-5,atol=1e-5)
        label = torch.zeros(512,dtype=torch.long); label[saved["pred_indices"]] = 1
        if int(label.sum()) != recipe["expected_positive_counts"][i]:
            raise ValueError("Fixed target count changed")
        groups["features"].append(hidden[0]); groups["labels"].append(label)
        for name,key in (("centers","centers"),("radii","radii"),("baseline_logits","logits"),("decoder_features","decoder_features")):
            groups[name].append(torch.from_numpy(saved[key][0]))
        for key in ("candidate_edges","edge_labels"):
            tensor_cache[f"{key}_{i}"] = torch.from_numpy(saved[key])
        row = fit["evaluations"][-1]["crops"][i]
        rows.append({"sample_index":row["sample_index"],"patch_index":row["patch_index"],"partition":"train",
            "assigned_queries":saved["pred_indices"].tolist(),"assigned_targets":saved["target_indices"].tolist(),
            "source_snapshot_row":row,"replay_max_logit_delta":float((raw[...,:2]-logits).abs().max())})
    tensor_cache.update({k:torch.stack(v) for k,v in groups.items()})
    tensor_cache.update(initial_weight=state["proposal.prediction.2.weight"][:2].clone(),
        initial_bias=state["proposal.prediction.2.bias"][:2].clone())
    if hashlib.sha256(tensor_cache["features"].numpy().tobytes()).hexdigest() != recipe["feature_tensor_sha256"]:
        raise ValueError("Classifier inputs differ from the verified capacity oracle")
    base.verify_hashes(protected)
    cache_dir = args.directory/"cache"; cache_dir.mkdir(parents=True)
    path = cache_dir/"features.pt"; torch.save(tensor_cache,path)
    base.write(cache_dir/"manifest.json",{"schema":CACHE_SCHEMA,"status":"completed","recipe":recipe,
        "cache_sha256":base.file_digest(path),"tensor_sha256":fingerprints(tensor_cache),"records":rows,
        "protected_files_sha256":protected,"optimizer_updates":0,"source_model_forwards":0})
    print(f"PREPARED exact frozen features/labels/geometry | positives=[6,12,16,418] | {cache_dir}",flush=True)


def load_cache(args,recipe):
    path = args.directory/"cache/manifest.json"
    manifest = base.read(path)
    cache_path = path.parent/"features.pt"
    if manifest["schema"] != CACHE_SCHEMA or manifest["recipe"] != recipe or manifest["status"] != "completed" or base.file_digest(cache_path) != manifest["cache_sha256"]:
        raise ValueError("Invalid or changed frozen cache")
    base.verify_hashes(manifest["protected_files_sha256"])
    cache = torch.load(cache_path,map_location="cpu",weights_only=True)
    expected = {"features":(4,512,128),"labels":(4,512),"centers":(4,512,3),"radii":(4,512,1),
        "baseline_logits":(4,512,2),"decoder_features":(4,512,128),"initial_weight":(2,128),"initial_bias":(2,)}
    expected.update({f"{key}_{i}":None for i in range(4) for key in ("candidate_edges","edge_labels")})
    if set(cache) != set(expected) or fingerprints(cache) != manifest["tensor_sha256"]:
        raise ValueError("Frozen tensor contents changed")
    for key,t in cache.items():
        if t.requires_grad or t.device.type != "cpu" or not bool(torch.isfinite(t).all()) or (expected[key] is not None and tuple(t.shape) != expected[key]):
            raise ValueError("Expected finite detached CPU cache tensors")
        required_dtype = torch.long if key == "labels" or key.startswith("candidate_edges_") else torch.float32
        if t.dtype != required_dtype:
            raise ValueError("Frozen precision changed")
    for i,row in enumerate(manifest["records"]):
        label = torch.zeros(512,dtype=torch.long); label[row["assigned_queries"]] = 1
        if row["partition"] != "train" or [row["sample_index"],row["patch_index"]] != recipe["training_crops"][i] or not torch.equal(label,cache["labels"][i]):
            raise ValueError("Crop identity or frozen labels changed")
    if cache["labels"].sum(1).tolist() != recipe["expected_positive_counts"]:
        raise ValueError("Expected the exact 6/12/16/418 fixed positives")
    original = base.recipe_at(ROOT/"configs/pointneuron2/branched_query_fit.json")
    _,records = base.load_records(SimpleNamespace(experiment_dir=ROOT/recipe["source_experiment"]),original)
    return manifest,cache,records


def make_head(cache):
    return FourCropFrozenConfidence(cache["initial_weight"],cache["initial_bias"])


def update(head,optimizer,cache,recipe):
    optimizer.zero_grad(set_to_none=True)
    loss,_ = stage_loss(head(cache["features"]),cache["labels"])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite stage loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(),recipe["gradient_clip"],error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise FloatingPointError("Nonfinite confidence rows")
    return {"loss":float(loss.detach()),"preclip_norm":float(norm)}


def checkpoint(head,optimizer,step,recipe,provenance,purpose="fit"):
    if not isinstance(step,int) or not 0 <= step <= recipe["updates"]:
        raise ValueError("Stage step is outside original cap")
    if {int(s["step"]) for s in optimizer.state.values()} != ({step} if step else set()):
        raise ValueError("Optimizer steps do not match stage checkpoint step")
    return {"schema":SCHEMA,"step":step,"recipe":recipe,"provenance":provenance,"purpose":purpose,
        "resumable":purpose == "fit" and step < recipe["updates"],"head_state":cpu_snapshot(head.state_dict()),
        "optimizer_state":cpu_snapshot(optimizer.state_dict()),"rng":capture_rng()}


def restore(cp,recipe,provenance,*,resume):
    if cp["schema"] != SCHEMA or cp["recipe"] != recipe or cp["provenance"] != provenance or not 0 <= cp["step"] <= recipe["updates"]:
        raise ValueError("Different confidence stage or provenance")
    if resume and (not cp["resumable"] or cp["purpose"] != "fit" or cp["step"] >= recipe["updates"]):
        raise ValueError("Check-only or terminal checkpoint cannot resume")
    head = FourCropFrozenConfidence(cp["head_state"]["linear.weight"],cp["head_state"]["linear.bias"])
    optimizer = torch.optim.AdamW(head.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
    optimizer.load_state_dict(cp["optimizer_state"]); restore_rng(cp["rng"])
    if {int(s["step"]) for s in optimizer.state.values()} != ({cp["step"]} if cp["step"] else set()):
        raise ValueError("Saved optimizer cursor differs from stage step")
    return head,optimizer,cp["step"]


def evaluate(head,cache,manifest,records,step,directory=None):
    with torch.no_grad():
        logits = head(cache["features"]); loss,per_crop = stage_loss(logits,cache["labels"])
    rows = []
    for i,(meta,record) in enumerate(zip(manifest["records"],records)):
        difference = logits[i,:,1]-logits[i,:,0]
        accepted = difference >= 0; positive = cache["labels"][i] == 1
        edges,labels = cache[f"candidate_edges_{i}"],cache[f"edge_labels_{i}"]
        remaining = int(((labels == 1) & accepted[edges].all(-1)).sum())
        required = len(record["target_edges"])
        support = base.measure_support(cache["centers"][i,accepted].numpy(),record["geometry"])["reference_to_points"]
        coverage = None if support is None else support["within_voxels"]["6"]
        correct,extras = int((positive & accepted).sum()),int((~positive & accepted).sum())
        row = {"sample_index":meta["sample_index"],"patch_index":meta["patch_index"],"targets":int(positive.sum()),
            "accepted_matched":correct,"accepted_unmatched":extras,"balanced_classification_loss":float(per_crop[i]),
            "minimum_matched_probability":float(difference[positive].sigmoid().min()),
            "maximum_unmatched_probability":float(difference[~positive].sigmoid().max()),
            "strict_fixed_label_gate":correct == int(positive.sum()) and extras == 0,
            "accepted_curve_support_6":coverage,"required_edge_recall_after_cutoff":remaining/required if required else None,
            "frozen_trusted_targets":meta["source_snapshot_row"]["connectivity_supervision"]["trusted_matches"],
            "frozen_trusted_junctions":meta["source_snapshot_row"]["trusted_visible_junctions"]}
        rows.append(row)
        print(f"EVAL stage_step={step} train={row['sample_index']}/{row['patch_index']} accepted={correct}/{row['targets']}+{extras} strict={row['strict_fixed_label_gate']}",flush=True)
    if directory is not None:
        np.savez_compressed(directory/f"scores_{step:04d}.npz",logits=logits.numpy())
    return {"stage_step":step,"optimization_loss":float(loss),"crops":rows,
        "strict_fixed_label_gate":all(r["strict_fixed_label_gate"] for r in rows)}


def snapshot(head,optimizer,step,recipe,provenance,report,directory,cache,manifest,records):
    path = directory/f"step_{step:04d}.pt"; temporary = path.with_suffix(".pt.writing")
    if path.exists() or temporary.exists():
        raise FileExistsError("Preserve checkpoint evidence")
    torch.save(checkpoint(head,optimizer,step,recipe,provenance),temporary); temporary.replace(path)
    report.update(last_checkpoint=path.name,last_checkpoint_step=step,last_checkpoint_sha256=base.file_digest(path))
    base.write(directory/"report.json",report)
    report["evaluations"].append(evaluate(head,cache,manifest,records,step,directory))
    base.write(directory/"report.json",report)


def run(args):
    recipe = load_recipe(args.recipe); args.directory = args.directory.resolve()
    if not args.directory.is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Generated stage outputs belong under ignored tmp/pointneuron2")
    torch.set_num_threads(recipe["torch_threads"])
    if args.action == "prepare":
        return prepare(args,recipe)
    manifest,cache,records = load_cache(args,recipe)
    before = fingerprints(cache)
    sources = base.source_hashes(SimpleNamespace(recipe=args.recipe))
    launcher = ROOT/"scripts/pointneuron2/run_four_crop_frozen_confidence.ps1"
    sources[str(launcher)] = base.file_digest(launcher)
    provenance = {"manifest_sha256":base.file_digest(args.directory/"cache/manifest.json"),"source_files_sha256":sources,
        "backend":{"device":"cpu","torch":str(torch.__version__),"threads":torch.get_num_threads()}}
    directory = args.directory/("check" if args.action == "check" else "model")
    if args.action == "resume":
        if args.checkpoint is None or not args.checkpoint.resolve().is_relative_to(args.directory) or not (args.checkpoint.parent.name == "model" or args.checkpoint.parent.name.startswith("resume_")):
            raise ValueError("Explicit checkpoint from this stage fit required")
        directory = args.directory/f"resume_{args.checkpoint.stem}"
    if directory.exists():
        raise FileExistsError("Existing action outputs cannot be overwritten")
    if args.action in ("train","resume"):
        checked = base.read(args.directory/"check/report.json")
        if checked["status"] != "completed" or checked["provenance"] != provenance or checked["recipe"] != recipe:
            raise ValueError("Expected matching completed CPU readiness check")
    directory.mkdir(parents=True)
    report = {"schema":SCHEMA,"status":"running","action":args.action,"recipe":recipe,"provenance":provenance,
        "source_joint_updates":4000,"stage_optimizer_updates":0,"trainable_parameters":258,"evaluations":[],
        "source_model_forwards":0,"cached_crops_per_update":4,"scope":"fixed training-label learnability only; geometry unchanged"}
    base.write(directory/"report.json",report); started = time.perf_counter()
    try:
        if args.action == "resume":
            cp = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
            head,optimizer,start = restore(cp,recipe,provenance,resume=True)
            report.update(resumed_from=str(args.checkpoint),resumed_at_step=start,stage_optimizer_updates=start)
        else:
            head = make_head(cache); optimizer = torch.optim.AdamW(head.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"]); start = 0
        if sum(p.numel() for p in head.parameters()) != 258 or len(list(head.parameters())) != 2:
            raise ValueError("Only the two confidence parameter tensors may train")
        for path in sources:
            target = directory/"sources"/Path(path).relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
        print(f"ACTION={args.action} CPU | only 258 confidence parameters | all four frozen crops per update | cap=4000",flush=True)
        if args.action == "check":
            report["evaluations"].append(evaluate(head,cache,manifest,records,0))
            with torch.no_grad():
                predicted = head(cache["features"])
                report["initial_replay_max_delta"] = float((predicted-cache["baseline_logits"]).abs().max())
                if not torch.equal(predicted[...,1] >= predicted[...,0],cache["baseline_logits"][...,1] >= cache["baseline_logits"][...,0]):
                    raise ValueError("Initial selection differs from balanced source")
            durations = []
            for j in range(8):
                one = time.perf_counter(); metrics = update(head,optimizer,cache,recipe)
                if j >= 4: durations.append(time.perf_counter()-one)
                print(f"THROWAWAY stage_update={j+1}/8 loss={metrics['loss']:.6f}",flush=True)
            temporary = directory/"throwaway.pt"; torch.save(checkpoint(head,optimizer,8,recipe,provenance,purpose="check"),temporary)
            loaded,_,_ = restore(torch.load(temporary,map_location="cpu",weights_only=True),recipe,provenance,resume=False)
            with torch.no_grad(): report["reload_max_delta"] = float((head(cache["features"])-loaded(cache["features"])).abs().max())
            temporary.unlink()
            estimate = float(np.mean(durations))*4000
            report.update(throwaway_optimizer_updates=8,trained_checkpoint_retained=False,
                estimated_update_loop_seconds=estimate,planning_range_seconds=[estimate,1.5*estimate+10])
            print(f"ESTIMATE update loop {estimate:.1f} seconds; allow {estimate:.1f}-{1.5*estimate+10:.1f} plus startup/checkpoint/evaluation IO",flush=True)
        else:
            for step in range(start,recipe["updates"]+1):
                if step > start:
                    metrics = update(head,optimizer,cache,recipe); report["stage_optimizer_updates"] = step
                    if step % 100 == 0: print(f"UPDATE stage={step}/4000 loss={metrics['loss']:.6f} elapsed={time.perf_counter()-started:.1f}s",flush=True)
                if step in recipe["snapshots"] and (args.action != "resume" or step > start):
                    snapshot(head,optimizer,step,recipe,provenance,report,directory,cache,manifest,records)
            cp = torch.load(directory/"step_4000.pt",map_location="cpu",weights_only=True)
            loaded,_,_ = restore(cp,recipe,provenance,resume=False)
            with torch.no_grad(): report["reload_max_delta"] = float((head(cache["features"])-loaded(cache["features"])).abs().max())
            report.update(final_checkpoint_sha256=base.file_digest(directory/"step_4000.pt"),strict_fixed_label_gate=report["evaluations"][-1]["strict_fixed_label_gate"])
        if fingerprints(cache) != before or report["reload_max_delta"] != 0:
            raise ValueError("Frozen cache or checkpoint reload changed")
        base.verify_hashes(manifest["protected_files_sha256"]); base.verify_hashes(sources)
        report.update(status="completed",elapsed_seconds=time.perf_counter()-started,cached_features_labels_geometry_unchanged=True,
            next_step="Review fixed-label result; no deployment, new joint fit, threshold change or automatic extension.")
        base.write(directory/"report.json",report); print(f"COMPLETED {args.action}; report={directory/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=str(exc))
        (directory/"error_traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        base.write(directory/"report.json",report); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action",choices=("prepare","check","train","resume"),required=True)
    parser.add_argument("--recipe",type=Path,default=ROOT/"configs/pointneuron2/four_crop_frozen_confidence.json")
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    run(parser.parse_args())

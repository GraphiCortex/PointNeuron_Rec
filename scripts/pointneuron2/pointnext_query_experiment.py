"""Separate prepare, timing and bounded-fit actions for the native prototype."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from pointneuron.data.field_pilot import load_manifest, write_json
from pointneuron.data.query_cache import file_digest
from pointneuron.data.field_guided_sampling import candidate_pools, select_cloud, local_xyz
from pointneuron.data.curve_targets import curve_geometry, patch_targets, target_arrays
from pointneuron.data.swc import parse_swc
from pointneuron.models.centerline_unet import CenterlineUNet
from pointneuron.models.pointnext_query import PointNextQueryModel, model_checkpoint, load_model_checkpoint, CHECKPOINT_SCHEMA
from pointneuron.models.query_loss import query_set_loss
from scripts.pointneuron2.compare_field_sampling import matching_source, protected_unchanged
from scripts.pointneuron2.field_sampling_metrics import reference_geometry, measure_support, reference_distances
from scripts.pointneuron2.train_query_comparison import inspect_fit, inspect_selection, inspect_matches

CACHE_SCHEMA = "pointneuron2-pointnext-query-cache-v1"


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def recipe_at(path):
    recipe = read_json(path)
    expected = read_json(ROOT / "configs/pointneuron2/pointnext_query.json")
    if (recipe != expected or recipe.get("schema") != CHECKPOINT_SCHEMA or recipe.get("input_budget") != 2048
            or recipe.get("updates") != 2000 or recipe.get("batch_size") != 1
            or recipe.get("training_crops") != [[0,0],[0,1]] or recipe.get("expected_target_counts") != [6,12]
            or recipe.get("snapshots") != [0,500,1000,2000] or recipe.get("augmentation") is not False):
        raise ValueError("Recipe differs from the declared bounded PointNeXt/query check")
    return recipe


def safe_path(base, relative):
    p = (base / relative).resolve()
    if not p.is_relative_to(base.resolve()):
        raise ValueError("Artifact escapes declared source/cache directory")
    return p


def source_contract(args):
    recipe = recipe_at(args.recipe)
    manifest, cfg = load_manifest(args.native_manifest, args.partition, args.field_config)
    fit_path, cp_path = args.field_models / "pilot_report.json", args.field_models / "checkpoints/field_epoch_030.pt"
    fit = read_json(fit_path)
    if (fit.get("status") != "completed" or fit.get("config") != cfg
            or fit.get("model_spec", {}).get("kind") != "centerline_unet_v1"
            or not fit.get("model_spec", {}).get("shared_initialization_sha256")
            or fit.get("manifest_sha256") != file_digest(args.native_manifest)
            or fit.get("partition_sha256") != file_digest(args.partition)
            or fit.get("final_checkpoint_sha256") != file_digest(cp_path)
            or fit.get("checkpoint_reload_max_logit_delta", float("inf")) > 1e-6):
        raise ValueError("Expected the fixed fresh-baseline field/cache contract")
    lookup = {(r["sample_index"],r["patch_index"]):r for r in manifest["records"] if r["partition"] == "train"}
    rows = [lookup[tuple(identity)] for identity in recipe["training_crops"]]
    if not all(r["supervised"] for r in rows):
        raise ValueError("Declared learnability crops lack local supervision")
    protected = {str(p.resolve()):file_digest(p) for p in
                 [args.recipe, args.partition, args.field_config, args.native_manifest, fit_path, cp_path]}
    for row in rows:
        protected[str(safe_path(args.native_manifest.parent,row["path"]))] = row["sha256"]
    protected_unchanged(protected)
    return recipe, rows, cfg, fit, cp_path, protected


def pack_geometry(geometry):
    return {"full_segments": geometry["full_segments"].tolist(), "critical_points": geometry["critical_points"].tolist(),
            "chains": [{**{k:v for k,v in c.items() if k != "probe"}, "probe": {
                "points": c["probe"]["points"].tolist(), "weights": c["probe"]["weights"].tolist()}} for c in geometry["chains"]]}


def unpack_geometry(saved):
    return {"full_segments":np.asarray(saved["full_segments"]).reshape(-1,2,3),
            "critical_points":np.asarray(saved["critical_points"]).reshape(-1,3),
            "chains":[{**{k:v for k,v in c.items() if k != "probe"}, "probe": {
                k:np.asarray(v) for k,v in c["probe"].items()}} for c in saved["chains"]]}


def inspect_prepared(raw,valid,points,nodes,geometry,origin,threshold,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scale=float(np.quantile(raw[valid],.995)) if valid.any() else 1.
    display=np.clip(raw/max(scale,1),0,1)
    xyz=points[:,:3]-origin;target=nodes[:,1:4]-origin
    reference=np.concatenate([c["probe"]["points"]-origin for c in geometry["chains"]])
    foreground=points[:,3]>threshold
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout="constrained")
    for ax,axis,dims,name in [(axes[0],0,(0,1),"XY"),(axes[1],1,(0,2),"XZ")]:
        ax.imshow(display.max(axis=axis),cmap="gray",origin="lower",vmin=0,vmax=1)
        for mask,color,label in [(foreground,"orange","image foreground"),(~foreground,"lime","field-only addition")]:
            ax.scatter(xyz[mask,dims[0]],xyz[mask,dims[1]],s=3,color=color,alpha=.45,label=label)
        ax.scatter(reference[:,dims[0]],reference[:,dims[1]],s=1,color="cyan",label="reference curve")
        ax.scatter(target[:,dims[0]],target[:,dims[1]],s=20,marker="x",color="magenta",label="training targets")
        ax.set_title(f"{name} | unique input N={len(points)} | targets={len(nodes)}")
        ax.set_xlabel("Local X voxel");ax.set_ylabel("Local Y / Z voxel")
    axes[0].legend(fontsize=7,loc="lower left")
    fig.suptitle("Fixed training crop; expanded uniform input frozen before reference scoring\nImage candidates and training targets are separate; projections are not graph edges",fontsize=10)
    fig.savefig(path,dpi=130);plt.close(fig)


def prepare(args):
    recipe, rows, cfg, fit, cp_path, protected = source_contract(args)
    cache = args.experiment_dir / "cache"
    if cache.exists():
        raise FileExistsError("Preserve prepared/failed caches; choose a fresh experiment directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select CPU explicitly")
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("epoch") != 30 or checkpoint.get("config") != cfg
            or checkpoint.get("model_spec") != fit["model_spec"] or checkpoint.get("manifest_sha256") != fit["manifest_sha256"]):
        raise ValueError("Field checkpoint/report identity differs")
    field = CenterlineUNet(cfg["base_channels"]).to(args.device).eval().requires_grad_(False)
    field.load_state_dict(checkpoint["model"], strict=True)
    torch.set_num_threads(2)
    cache.mkdir(parents=True)
    started = time.perf_counter()
    result = {"schema":CACHE_SCHEMA,"status":"running","recipe":recipe,"optimizer_updates":0,
              "field_inference_passes":0,"pointnext_inference_passes":0,"reserved_test_records":0,
              "source_field_checkpoint_sha256":file_digest(cp_path),"records":[],"phase":"freezing image-only clouds"}
    result["field_backend"]={"device":args.device,"torch":str(torch.__version__),
        "device_name":torch.cuda.get_device_name() if args.device=="cuda" else "CPU",
        "cudnn_tf32":torch.backends.cudnn.allow_tf32,"matmul_tf32":torch.backends.cuda.matmul.allow_tf32}
    frozen = []
    try:
        for row in rows:
            with np.load(safe_path(args.native_manifest.parent,row["path"]),allow_pickle=False) as z:
                raw, image, valid = (z[k].copy() for k in ("raw_image","image","valid"))
                meta = json.loads(str(z["metadata"]))
            if (not np.array_equal(raw.astype(np.float32)/meta["image_scale"],image)
                    or any(meta[k] != row[k] for k in ("sample_index","sample_id","patch_index","partition"))):
                raise ValueError("Native image normalization/identity differs")
            with torch.inference_mode():
                probability = field(torch.from_numpy(image)[None,None].to(args.device)).sigmoid()[0,0].cpu().numpy()
            result["field_inference_passes"] += 1
            pools = candidate_pools(raw,valid,probability,meta["threshold"])
            if len(pools.foreground) != meta["foreground_count"]:
                raise ValueError("Native foreground availability differs")
            n = min(recipe["input_budget"],len(pools.expanded))
            if n < 256:
                raise ValueError("Insufficient distinct support for the fixed S hierarchy; no input padding")
            seed = recipe["sampling_seed"]+row["sample_index"]*10000+row["patch_index"]
            cloud = select_cloud(raw,valid,probability,meta["origin_xyz"],meta["threshold"],"expanded_random",n,seed,pools=pools)
            name = Path(row["path"]).stem
            p = cache / (name+"_points.npz")
            np.savez_compressed(p,points=cloud.points,voxel_indices=cloud.voxel_indices,field_scores=cloud.field_scores,
                                probability=probability,metadata=np.array(json.dumps(meta)))
            protected[str(p.resolve())] = file_digest(p)
            frozen.append((row,meta,p,raw.shape))
            print(f"frozen train neuron={row['sample_index']} crop={row['patch_index']} F={len(pools.foreground)} C={len(pools.expanded)} unique N={n}; field inference only",flush=True)
        del field
        result["phase"] = "all point selections frozen; construct reference-only targets"
        write_json(cache/"manifest.json",result)
        for i,(row,meta,points_path,shape) in enumerate(frozen):
            directory = safe_path(args.data_root,row["sample_id"])
            swc = matching_source(directory,{".swc"},meta["swc_sha256"])
            protected[str(swc.resolve())] = meta["swc_sha256"]
            tree = parse_swc(swc)
            targets = patch_targets(curve_geometry(tree),meta["patch_center"],recipe["patch_radius"],recipe["target_spacing"])
            if len(targets) != recipe["expected_target_counts"][i]:
                raise ValueError("Declared 6/12 target counts changed; inspect without substituting crops")
            nodes,roles = target_arrays(targets,512)
            geometry = reference_geometry(tree,np.array(meta["origin_xyz"]),shape)
            # Store geometry in native coordinates for proposal diagnostics.
            origin = np.asarray(meta["origin_xyz"])
            geometry["full_segments"] += origin
            geometry["critical_points"] += origin
            for chain in geometry["chains"]: chain["probe"]["points"] += origin
            target_path = cache / (Path(row["path"]).stem+"_targets.npz")
            np.savez_compressed(target_path,nodes=nodes,roles=roles,geometry=np.array(json.dumps(pack_geometry(geometry))))
            protected[str(target_path.resolve())] = file_digest(target_path)
            with np.load(points_path,allow_pickle=False) as z: points=z["points"]
            with np.load(safe_path(args.native_manifest.parent,row["path"]),allow_pickle=False) as z:
                raw,valid=z["raw_image"],z["valid"]
            inspection_path=cache/(Path(row["path"]).stem+"_inspection.png")
            inspect_prepared(raw,valid,points,nodes,geometry,origin,meta["threshold"],inspection_path)
            result["records"].append({**{k:row[k] for k in ("sample_index","sample_id","patch_index","partition")},
                "points_path":points_path.name,"points_sha256":file_digest(points_path),"targets_path":target_path.name,
                "targets_sha256":file_digest(target_path),"actual_unique_count":len(points),"target_count":len(nodes),
                "patch_center":meta["patch_center"],"image_scale":meta["image_scale"],
                "inspection_path":inspection_path.name,
                "reference_identities":[t["reference_identity"] for t in targets]})
            print(f"target neuron={row['sample_index']} crop={row['patch_index']} count={len(nodes)}; frozen points unchanged",flush=True)
        protected_unchanged(protected)
        result.update(status="completed",phase="prepared",protected_files_sha256=protected,
                      elapsed_seconds=time.perf_counter()-started,next_step="Inspect cache; run separate preflight. No PointNeXt fit starts automatically.")
        write_json(cache/"manifest.json",result)
        print(f"Status: prepared; field inference passes=2; PointNeXt inference passes=0; optimizer updates=0\nmanifest: {cache/'manifest.json'}",flush=True)
        return result
    except Exception as exc:
        result.update(status="failed",error=str(exc));write_json(cache/"manifest.json",result);raise


def load_cache(args):
    recipe = recipe_at(args.recipe)
    path = args.experiment_dir / "cache/manifest.json"
    manifest = read_json(path)
    if (manifest.get("schema") != CACHE_SCHEMA or manifest.get("status") != "completed" or manifest.get("recipe") != recipe
            or manifest.get("optimizer_updates") != 0 or manifest.get("reserved_test_records") != 0
            or [(r["sample_index"],r["patch_index"]) for r in manifest["records"]] != [(0,0),(0,1)]):
        raise ValueError("Expected the immutable two-training-crop native cache")
    protected_unchanged(manifest["protected_files_sha256"])
    records = []
    for r in manifest["records"]:
        p,t = safe_path(path.parent,r["points_path"]),safe_path(path.parent,r["targets_path"])
        if r["partition"] != "train" or file_digest(p) != r["points_sha256"] or file_digest(t) != r["targets_sha256"]:
            raise ValueError("Changed/nontraining cache record")
        with np.load(p,allow_pickle=False) as z:
            points, indices = z["points"].copy(), z["voxel_indices"].copy()
            meta = json.loads(str(z["metadata"]))
        with np.load(t,allow_pickle=False) as z:
            nodes,geometry = z["nodes"].copy(),unpack_geometry(json.loads(str(z["geometry"])))
        if (points.shape != (r["actual_unique_count"],4) or len(points)<256 or len(np.unique(indices)) != len(points)
                or nodes.shape != (r["target_count"],6) or r["target_count"] not in [6,12]
                or not np.isfinite(points).all() or not np.isfinite(nodes).all()
                or meta["patch_center"] != r["patch_center"] or meta["image_scale"] != r["image_scale"]):
            raise ValueError("Invalid point/target/scale count or geometry")
        records.append({"metadata":r,"points":torch.from_numpy(points)[None].to(args.device),
            "nodes":torch.from_numpy(nodes)[None].to(args.device),"mask":torch.ones(1,len(nodes),dtype=torch.bool,device=args.device),
            "centers":torch.tensor([r["patch_center"]],dtype=torch.float32,device=args.device),
            "scales":torch.tensor([r["image_scale"]],dtype=torch.float32,device=args.device),"geometry":geometry})
    protected = dict(manifest["protected_files_sha256"])
    protected[str(path.resolve())] = file_digest(path)
    return recipe,manifest,records,protected


def forward(model,r):
    return model(r["points"],r["centers"],r["scales"],48.)


def update(model,optimizer,r,recipe):
    model.train();optimizer.zero_grad(set_to_none=True)
    output=forward(model,r)
    loss=query_set_loss(output,r["nodes"],r["mask"],**recipe["query_loss"])
    if loss.supervised_crops != 1 or not bool(torch.isfinite(loss.total)):
        raise ValueError("Expected one finite supervised update")
    loss.total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),recipe["gradient_clip"],error_if_nonfinite=True);optimizer.step()
    return float(loss.total.detach())


def evaluate(model,records,output_dir,step):
    model.eval();rows=[]
    with torch.no_grad():
        for i,r in enumerate(records):
            output=forward(model,r)
            loss=query_set_loss(output,r["nodes"],r["mask"])
            match=inspect_matches(output,r["nodes"],r["mask"],loss)[0]
            scores=output.objectness_logits.softmax(-1)[0,:,1]
            accepted=scores>=.5
            centers=output.center_proposals[0,accepted].cpu().double().numpy()
            support=measure_support(centers,r["geometry"])
            curve6=None if support["reference_to_points"] is None else support["reference_to_points"]["within_voxels"]["6"]
            radius_error=float((output.radius[0,torch.tensor(loss.assignments[0][0],device=output.radius.device),0]-
                r["nodes"][0,torch.tensor(loss.assignments[0][1],device=r["nodes"].device),4].clamp_min(1)).abs().mean())
            passed=(match["max_matched_distance"]<=1 and match["matched_accepted_within_6"]==match["targets"]
                    and match["unused_accepted"]==0 and curve6 is not None and curve6>=.95
                    and bool(torch.isfinite(output.radius).all()) and bool((output.radius>0).all()))
            row={"sample_index":r["metadata"]["sample_index"],"patch_index":r["metadata"]["patch_index"],
                "actual_input_count":r["metadata"]["actual_unique_count"],"geometry":inspect_fit(output,r["nodes"],r["mask"]),
                "selection":inspect_selection(output,r["nodes"],r["mask"]),"matching":match,
                "assignments":[[int(v) for v in part] for part in loss.assignments[0]],"radius_mae_floored_target":radius_error,
                "accepted_curve_support":support,"learnability_gate_passed":passed}
            rows.append(row)
            if output_dir:
                np.savez_compressed(output_dir/f"step_{step:04d}_crop_{i}.npz",centers=output.center_proposals.cpu().numpy(),
                    logits=output.objectness_logits.cpu().numpy(),radii=output.radius.cpu().numpy())
    return {"step":step,"crops":rows,"learnability_gate_passed":all(r["learnability_gate_passed"] for r in rows)}


def sync(device):
    if device=="cuda":torch.cuda.synchronize()


def source_files(args):
    paths=[Path(__file__),args.recipe,ROOT/"scripts/pointneuron2/run_pointnext_query.ps1",
        *[ROOT/"src/pointneuron/models"/name for name in ["pointnext_ops.py","pointnext_native.py","pointnext_query.py","query_proposal.py","query_loss.py"]],
        ROOT/"scripts/pointneuron2/train_query_comparison.py",ROOT/"scripts/pointneuron2/field_sampling_metrics.py",
        ROOT/"docs/pointneuron2/pointnext_reference/reference_manifest.json"]
    return {str(p.resolve()):file_digest(p) for p in paths}


def run_model_action(args):
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; select CPU explicitly")
    torch.set_num_threads(2)
    recipe,manifest,records,protected=load_cache(args)
    sources=source_files(args);protected.update(sources)
    backend={"device":args.device,"device_name":torch.cuda.get_device_name() if args.device=="cuda" else "CPU",
             "torch":str(torch.__version__),"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version()}
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    backend.update(cudnn_benchmark=False,matmul_tf32=False,cudnn_tf32=False,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic)
    timing_path=args.experiment_dir/"timing.json"
    if args.action=="preflight" and timing_path.exists():raise FileExistsError("Preserve existing timing; use a fresh experiment")
    if args.action=="train":
        timing=read_json(timing_path)
        if (timing.get("status")!="completed" or timing.get("backend")!=backend or timing.get("source_files_sha256")!=sources
                or timing.get("cache_manifest_sha256")!=file_digest(args.experiment_dir/"cache/manifest.json")):
            raise ValueError("Expected matching reviewed GPU timing/cache/source contract")
        output_dir=args.experiment_dir/"model"
        if output_dir.exists():raise FileExistsError("Preserve trained/failed outputs; use a fresh experiment")
        output_dir.mkdir()
    if args.device=="cuda":torch.cuda.reset_peak_memory_stats()
    model=PointNextQueryModel(recipe["seed"]).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
    print(f"Device: {backend['device_name']} | native PointNeXt S + Transformer | actual counts={[r['points'].shape[1] for r in records]} | batch=1 float32",flush=True)
    report={"schema":CHECKPOINT_SCHEMA,"status":"running","scope":"timing only" if args.action=="preflight" else "two fixed training crops; no development/test or SWC reconstruction",
            "recipe":recipe,"backend":backend,"model_spec":model.model_spec(),"source_files_sha256":sources,
            "cache_manifest_sha256":file_digest(args.experiment_dir/"cache/manifest.json"),"optimizer_updates":0,"evaluations":[]}
    started=time.perf_counter()
    if args.action=="preflight":
        durations=[]
        for step in range(4):
            sync(args.device);one=time.perf_counter();value=update(model,optimizer,records[step%2],recipe);sync(args.device)
            if step>=2:durations.append(time.perf_counter()-one)
            print(f"throwaway update={step+1}/4 crop={step%2} loss={value:.4f}",flush=True)
        model.eval()
        probe=records[0]
        with torch.no_grad():before=forward(model,probe)
        cp=model_checkpoint(model,seed=recipe["seed"],step=4,recipe=recipe,provenance={"timing_only":True})
        loaded=load_model_checkpoint(cp,args.device)
        with torch.no_grad():after=forward(loaded,probe)
        delta=max(float((getattr(before,k)-getattr(after,k)).abs().max()) for k in ["center_proposals","objectness_logits","radius"])
        if delta>1e-5:raise ValueError("Prototype reload mismatch")
        estimate=float(np.mean(durations))*recipe["updates"]
        protected_unchanged(protected)
        report.update(status="completed",throwaway_optimizer_updates=4,checkpoint_retained=False,
                      checkpoint_reload_max_delta=delta,estimated_fit_seconds=estimate,
                      planning_range_seconds=[estimate,estimate*1.5],elapsed_seconds=time.perf_counter()-started,
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device=="cuda" else None,
                      next_step="Review inputs and timing; train is a separate explicit command.")
        write_json(timing_path,report)
        print(f"Estimated 2,000-update fit compute: {estimate/60:.1f} minutes; planning range {estimate/60:.1f}-{estimate*1.5/60:.1f}. No checkpoint retained.\nreport: {timing_path}",flush=True)
        return report
    report_path=output_dir/"fit_report.json"
    write_json(report_path,report)
    try:
        for step in range(recipe["updates"]+1):
            if step:
                value=update(model,optimizer,records[(step-1)%2],recipe);report["optimizer_updates"]=step
                if step%50==0:print(f"update={step}/{recipe['updates']} crop={(step-1)%2} loss={value:.5f} elapsed={time.perf_counter()-started:.1f}s",flush=True)
            if step in recipe["snapshots"]:
                evaluation=evaluate(model,records,output_dir,step);report["evaluations"].append(evaluation)
                cp=model_checkpoint(model,seed=recipe["seed"],step=step,recipe=recipe,
                    provenance={"cache_manifest_sha256":report["cache_manifest_sha256"],"source_files_sha256":sources})
                torch.save(cp,output_dir/f"step_{step:04d}.pt")
                print(f"eval step={step} learnability_gate_passed={evaluation['learnability_gate_passed']}",flush=True)
                write_json(report_path,report)
        final_path=output_dir/"step_2000.pt"
        reloaded=load_model_checkpoint(torch.load(final_path,map_location="cpu",weights_only=True),args.device)
        model.eval();maximum=0.
        with torch.no_grad():
            for r in records:
                a,b=forward(model,r),forward(reloaded,r)
                maximum=max(maximum,*(float((getattr(a,k)-getattr(b,k)).abs().max()) for k in ["center_proposals","objectness_logits","radius"]))
        if maximum>1e-5:raise ValueError("Final checkpoint reload mismatch")
        protected_unchanged(protected)
        report.update(status="completed",checkpoint_reload_max_delta=maximum,final_checkpoint_sha256=file_digest(final_path),
            learnability_gate_passed=report["evaluations"][-1]["learnability_gate_passed"],elapsed_seconds=time.perf_counter()-started,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device=="cuda" else None,
            next_step="Review geometry, confidence, curves and gaps together; execution success is not generalization or final reconstruction.")
        write_json(report_path,report)
        print(f"Status: completed; checkpoint reload passed; learnability gate: {report['learnability_gate_passed']}\nreport: {report_path}",flush=True)
        return report
    except Exception as exc:
        report.update(status="failed",error=str(exc));write_json(report_path,report);raise


def main(args):
    if args.action=="check":
        recipe,rows,*_=source_contract(args)
        print("Status: source contracts passed; exactly two declared training crops; no inference or optimizer updates.",flush=True)
        return
    if args.action=="prepare":return prepare(args)
    return run_model_action(args)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action",choices=["check","prepare","preflight","train"],required=True)
    p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--native-manifest",type=Path,required=True)
    p.add_argument("--field-models",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--recipe",type=Path,default=ROOT/"configs/pointneuron2/pointnext_query.json")
    p.add_argument("--field-config",type=Path,default=ROOT/"configs/pointneuron2/field_pilot.json")
    p.add_argument("--partition",type=Path,default=ROOT/"configs/pointneuron2/partition_seed0.json")
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    return p


if __name__=="__main__":main(parser().parse_args())

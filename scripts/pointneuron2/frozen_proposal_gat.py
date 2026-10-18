"""Bounded connectivity learning on frozen, oracle-selected TRAINING proposals."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from scripts.pointneuron2 import branched_query_experiment as base
from pointneuron.data.connectivity_targets import masked_edge_loss
from pointneuron.data.curve_targets import clip_parameters
from pointneuron.graph.proposal_graph import ProposalGraphInput, CandidateGraph, build_candidate_graph
from pointneuron.graph.connectivity_decoder import decode_forest, export_decoded_swc
from pointneuron.graph.representation_oracle import ReferenceGraph, geometry_report
from pointneuron.models.gatv2_connectivity import ConnectivityConfig, GATv2Connectivity
from pointneuron.models.query_fit_state import cpu_snapshot, capture_rng, restore_rng

SCHEMA = "pointneuron2-frozen-proposal-gat-fit-v1"
FIELDS = ("positions", "radii", "objectness_logits", "local_centers", "decoder_features", "node_ids")


def oracle_labels(edges, target_indices, target_edges, components):
    """Training-only correspondence; never fed to the scorer or candidate builder."""
    target_indices = np.asarray(target_indices)
    n = len(target_indices)
    if sorted(target_indices.tolist()) != list(range(n)):
        raise ValueError("Expected a complete one-to-one target correspondence")
    required_targets = {tuple(sorted(map(int, e))) for e in target_edges}
    inverse = np.argsort(target_indices)
    required = {tuple(sorted((int(inverse[u]), int(inverse[v])))) for u, v in required_targets}
    labels = []
    for u, v in edges.tolist():
        a, b = int(target_indices[u]), int(target_indices[v])
        labels.append(1. if tuple(sorted((a, b))) in required_targets else
                      0. if components[a] >= 0 and components[a] == components[b] else -1.)
    if not required.issubset({tuple(sorted(map(int, e))) for e in edges.tolist()}):
        raise ValueError("A required reference connection is absent from candidates")
    return torch.tensor(labels, dtype=torch.float32), torch.tensor(sorted(required), dtype=torch.long).reshape(-1, 2)


def packet(record, device="cpu"):
    return ProposalGraphInput(*(record[k].to(device) for k in FIELDS), {"scope": "oracle_selected_training_control"})


def candidates(record, recipe, device="cpu"):
    return CandidateGraph(record["edges"].to(device), recipe["candidate_neighbors"], recipe["candidate_radius"])


def graph(record, edges):
    return ReferenceGraph([("query", int(i)) for i in record["node_ids"]], record["positions"].numpy().astype(float),
                          record["radii"].numpy()[:, 0].astype(float), [tuple(map(int, e)) for e in edges])


def reference_graph(record):
    seg = record["reference_segments"].numpy(); n = len(seg)
    return ReferenceGraph([("segment_endpoint", i) for i in range(2*n)], seg.reshape(-1, 3), np.ones(2*n),
                          [(2*i, 2*i+1) for i in range(n)])


def graph_metrics(record, edge_logits, recipe, directory=None, prefix="graph"):
    """Independent edge decisions plus actual score-thresholded forest decoding."""
    p = packet(record); c = candidates(record, recipe)
    logits = edge_logits.detach().cpu()
    # Oracle node retention applies ONLY to decoding. Scorer sees saved confidence.
    decoded = decode_forest(replace(p, objectness_logits=torch.zeros_like(p.objectness_logits)), c, logits,
                            node_cutoff=.5, edge_cutoff=recipe["edge_cutoff"])
    labels = record["labels"]
    predicted = logits >= 0  # Fixed .5 cutoff; no threshold selection.
    expected = {tuple(map(int, e)) for e in record["required_edges"].tolist()}
    selected = {tuple(sorted(e)) for e in decoded.selected_edges}
    expected = {tuple(sorted(e)) for e in expected}
    # Decoder orientation is a serialization convention; compare undirected incidence.
    degree = np.zeros(len(p.positions), int); wanted = degree.copy()
    for a, b in selected: degree[a] += 1; degree[b] += 1
    for a, b in expected: wanted[a] += 1; wanted[b] += 1
    expected_components = len(p.positions)-len(expected)
    row = {"sample_index": record["sample_index"], "patch_index": record["patch_index"],
           "nodes": len(p.positions), "candidate_edges": len(c.edges), "required_edges": len(expected),
           "positive_edges_rejected": int(((labels == 1) & ~predicted).sum()),
           "known_negative_edges_accepted": int(((labels == 0) & predicted).sum()),
           "unknown_edges_above_cutoff": int(((labels == -1) & predicted).sum()),
           "decoded_missing_required_edges": len(expected-selected), "decoded_nonreference_edges": len(selected-expected),
           "decoded_components": decoded.statistics["components"], "reference_components": expected_components,
           "degree_errors": int((degree != wanted).sum()),
           "junction_incidence_errors": int(((degree != wanted) & (wanted >= 3)).sum()),
           "supervised_classification_gate": bool(not ((labels == 1) & ~predicted).any() and not ((labels == 0) & predicted).any()),
           "exact_reference_graph_gate": selected == expected,
           "geometry": geometry_report(reference_graph(record), graph(record, decoded.selected_edges)),
           "decoding": decoded.statistics}
    if directory is not None:
        row["swc_roundtrip"] = export_decoded_swc(p, decoded, directory/f"{prefix}.swc",
            description="TRAINING DIAGNOSTIC: oracle-selected frozen balanced proposals; learned GAT edges; not automatic inference.")
    return row


def recipe_at(path):
    recipe = base.read(path)
    if (recipe != base.read(ROOT/"configs/pointneuron2/frozen_proposal_gat.json") or recipe["schema"] != SCHEMA
            or recipe["updates"] != 4000 or recipe["edge_cutoff"] != .5
            or recipe["training_crops"] != [[0,0],[0,1],[92,4],[86,3]]):
        raise ValueError("Recipe differs from the declared connectivity control")
    return recipe


def fingerprints(records):
    return [{k:base.tensor_digest(v) for k,v in row.items() if isinstance(v, torch.Tensor)} for row in records]


def prepare(args, recipe):
    if args.directory.exists(): raise FileExistsError("Preserve existing control outputs")
    source = ROOT/recipe["source_experiment"]
    original = base.recipe_at(ROOT/"configs/pointneuron2/branched_query_fit.json")
    manifest, originals = base.load_records(SimpleNamespace(experiment_dir=source), original)
    fit = base.read(source/"model/report.json")
    protected = {**manifest["protected_files_sha256"], **fit["provenance"]["source_files_sha256"],
        str(source/"model/report.json"): recipe["source_report_sha256"],
        str(source/"model/step_4000.pt"): recipe["source_checkpoint_sha256"],
        str(source/"cache/manifest.json"): base.file_digest(source/"cache/manifest.json")}
    base.verify_hashes(protected)
    saved_hashes = base.read(ROOT/"docs/pointneuron2/hard_negative_query_results_seed0/summary.json")["balanced"]["prediction_files_sha256"]
    rows, summaries = [], []
    for i, original_record in enumerate(originals):
        path = source/f"model/step_4000_crop_{i}.npz"; protected[str(path)] = saved_hashes[path.name]
        if base.file_digest(path) != protected[str(path)]: raise ValueError("Saved balanced predictions changed")
        with np.load(path, allow_pickle=False) as z: saved = {k:z[k].copy() for k in z.files}
        # Sort by original query ID, never by target identity, to avoid injecting target order.
        order = np.argsort(saved["pred_indices"]); pi = saved["pred_indices"][order]; ti = saved["target_indices"][order]
        r = {"sample_index": recipe["training_crops"][i][0], "patch_index": recipe["training_crops"][i][1], "partition": "train"}
        for name, key in (("positions","centers"),("radii","radii"),("objectness_logits","logits"),("decoder_features","decoder_features")):
            r[name] = torch.from_numpy(saved[key][0,pi].copy())
        center = np.asarray(original_record["meta"]["patch_center"])
        r["local_centers"] = (r["positions"]-torch.tensor(center, dtype=torch.float32))/original["patch_radius"]
        r["node_ids"] = torch.from_numpy(pi.copy())
        p = packet(r); c = build_candidate_graph(p.positions, p.node_ids,
            neighbors=recipe["candidate_neighbors"], radius=recipe["candidate_radius"])
        r["edges"] = c.edges
        r["labels"], r["required_edges"] = oracle_labels(c.edges, ti, original_record["target_edges"], original_record["target_components"])
        # Full segments in the cache cover the neuron. Clip to the exact target box.
        segments = []
        for a,b in original_record["geometry"]["full_segments"]:
            interval = clip_parameters(a,b,center-original["patch_radius"],center+original["patch_radius"])
            if interval is not None:
                lo,hi = interval
                if (hi-lo)*np.linalg.norm(b-a) > 0: segments.append([a+lo*(b-a),a+hi*(b-a)])
        r["reference_segments"] = torch.tensor(np.asarray(segments), dtype=torch.float64)
        r["target_indices"] = torch.from_numpy(ti.copy())
        if len(p.positions) != recipe["expected_nodes"][i] or len(r["required_edges"]) != recipe["expected_required_edges"][i]:
            raise ValueError("Declared reference size differs")
        if not bool((r["labels"] == 1).any()) or not bool((r["labels"] == 0).any()):
            raise ValueError("Every declared crop must have positive and known negative edges")
        oracle_logits = torch.where(r["labels"] == 1, 20., -20.)
        oracle = graph_metrics(r, oracle_logits, recipe)
        if not oracle["exact_reference_graph_gate"]: raise ValueError("Oracle graph/decoder feasibility failed")
        summary = {"sample_index":r["sample_index"],"patch_index":r["patch_index"],"partition":"train",
            "nodes":len(p.positions),"candidates":len(c.edges),
            "positive":int((r["labels"] == 1).sum()),"negative":int((r["labels"] == 0).sum()),
            "unknown":int((r["labels"] == -1).sum()),"oracle":oracle}
        summaries.append(summary); rows.append(r)
        print(f"PREPARED train={r['sample_index']}/{r['patch_index']} nodes={len(p.positions)} labels={summary['positive']}+/{summary['negative']}-/{summary['unknown']}? oracle_graph=True", flush=True)
    base.verify_hashes(protected)
    output = args.directory/"cache"; output.mkdir(parents=True)
    torch.save(rows, output/"graphs.pt")
    base.write(output/"manifest.json", {"schema":SCHEMA+"-cache", "status":"completed", "recipe":recipe,
        "cache_sha256":base.file_digest(output/"graphs.pt"),"tensor_sha256":fingerprints(rows),
        "records":summaries,"protected_files_sha256":protected,"optimizer_updates":0,"upstream_forwards":0})


def load_cache(args, recipe):
    folder = args.directory/"cache"; manifest = base.read(folder/"manifest.json")
    if manifest["schema"] != SCHEMA+"-cache" or manifest["recipe"] != recipe or manifest["status"] != "completed" or base.file_digest(folder/"graphs.pt") != manifest["cache_sha256"]:
        raise ValueError("Invalid or changed frozen cache")
    base.verify_hashes(manifest["protected_files_sha256"])
    rows = torch.load(folder/"graphs.pt", map_location="cpu", weights_only=True)
    if len(rows) != 4 or fingerprints(rows) != manifest["tensor_sha256"]: raise ValueError("Frozen tensors changed")
    for i,r in enumerate(rows):
        if r["partition"] != "train" or [r["sample_index"],r["patch_index"]] != recipe["training_crops"][i]: raise ValueError("Invalid crop partition/identity")
        packet(r).validate(); candidates(r,recipe).validate(packet(r))
        for v in r.values():
            if isinstance(v,torch.Tensor) and (v.requires_grad or v.device.type != "cpu" or not bool(torch.isfinite(v).all())):
                raise ValueError("Cache must be detached finite CPU tensors")
    return manifest,rows


def fresh(recipe, device):
    model = GATv2Connectivity(ConnectivityConfig(**recipe["model"])).to(device)
    return model, torch.optim.AdamW(model.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])


def checkpoint(model, optimizer, step, recipe, provenance, purpose="fit"):
    if type(step) is not int or not 0 <= step <= recipe["updates"]: raise ValueError("Invalid stage step")
    if {int(s["step"]) for s in optimizer.state.values()} != ({step} if step else set()): raise ValueError("Optimizer cursor differs")
    return {"schema":SCHEMA,"step":step,"next_crop":step%4,"recipe":recipe,"provenance":provenance,"purpose":purpose,
        "resumable":purpose == "fit" and step < recipe["updates"],"spec":model.model_spec(),
        "model":cpu_snapshot(model.state_dict()),"optimizer":cpu_snapshot(optimizer.state_dict()),"rng":capture_rng()}


def restore(cp, recipe, provenance, device, *, resume):
    if (cp["schema"] != SCHEMA or cp["recipe"] != recipe or cp["provenance"] != provenance
            or type(cp["step"]) is not int or not 0 <= cp["step"] <= recipe["updates"] or cp["next_crop"] != cp["step"]%4):
        raise ValueError("Different experiment, recipe, provenance or cursor")
    if resume and (not cp["resumable"] or cp["purpose"] != "fit" or cp["step"] >= recipe["updates"]): raise ValueError("Terminal/check-only checkpoint cannot resume")
    model,optimizer = fresh(recipe,device)
    if model.model_spec() != cp["spec"]: raise ValueError("Different connectivity specification")
    model.load_state_dict(cp["model"]); optimizer.load_state_dict(cp["optimizer"]); restore_rng(cp["rng"])
    if {int(s["step"]) for s in optimizer.state.values()} != ({cp["step"]} if cp["step"] else set()): raise ValueError("Optimizer cursor differs")
    return model,optimizer,cp["step"]


def update(model,optimizer,record,recipe,device):
    model.train(); optimizer.zero_grad(set_to_none=True)
    scores = model(packet(record,device), candidates(record,recipe,device))
    result = masked_edge_loss(scores.edge_logits,record["labels"].to(device))
    if not result.supervised_crops: raise ValueError("No supervised edges: skip rather than optimize unknown labels")
    result.total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),recipe["gradient_clip"],error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()): raise FloatingPointError("Nonfinite parameters")
    return {"loss":float(result.total.detach()),"preclip_norm":float(norm)}


def evaluate(model,records,recipe,device,step,directory=None):
    model.eval(); rows=[]; arrays={}
    with torch.no_grad():
        for i,r in enumerate(records):
            logits = model(packet(r,device),candidates(r,recipe,device)).edge_logits.cpu()
            row = graph_metrics(r,logits,recipe,directory,f"step_{step:04d}_crop_{i}")
            row["balanced_edge_loss"] = float(masked_edge_loss(logits,r["labels"]).total)
            rows.append(row); arrays[f"crop_{i}"] = logits.numpy()
            print(f"EVAL step={step} train={r['sample_index']}/{r['patch_index']} missing={row['decoded_missing_required_edges']} extra={row['decoded_nonreference_edges']} components={row['decoded_components']}/{row['reference_components']} junction_errors={row['junction_incidence_errors']} graph={row['exact_reference_graph_gate']}",flush=True)
    if directory is not None: np.savez_compressed(directory/f"scores_{step:04d}.npz",**arrays)
    return {"step":step,"crops":rows,"graph_gate":all(r["exact_reference_graph_gate"] for r in rows),
            "supervised_classification_gate":all(r["supervised_classification_gate"] for r in rows)}


def snapshot(model,optimizer,step,recipe,provenance,report,directory,records,device):
    path=directory/f"step_{step:04d}.pt"; temporary=path.with_suffix(".pt.writing")
    if path.exists() or temporary.exists(): raise FileExistsError("Preserve checkpoint evidence")
    torch.save(checkpoint(model,optimizer,step,recipe,provenance),temporary); temporary.replace(path)
    report.update(last_checkpoint=path.name,last_checkpoint_step=step,last_checkpoint_sha256=base.file_digest(path))
    base.write(directory/"report.json",report)
    report["evaluations"].append(evaluate(model,records,recipe,device,step,directory))
    base.write(directory/"report.json",report)


def reload_delta(model,loaded,records,recipe,device):
    model.eval(); loaded.eval()
    with torch.no_grad():
        return max(float((model(packet(r,device),candidates(r,recipe,device)).edge_logits-
                          loaded(packet(r,device),candidates(r,recipe,device)).edge_logits).abs().max()) for r in records)


def run(args):
    recipe=recipe_at(args.recipe); args.directory=args.directory.resolve()
    if not args.directory.is_relative_to((ROOT/"tmp/pointneuron2").resolve()): raise ValueError("Generated outputs belong under ignored tmp/pointneuron2")
    torch.set_num_threads(recipe["torch_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if args.action == "prepare": return prepare(args,recipe)
    if args.action == "check" and args.device != "cpu": raise ValueError("Readiness check is CPU only")
    device=torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    manifest,records=load_cache(args,recipe); before=fingerprints(records)
    sources=base.source_hashes(SimpleNamespace(recipe=args.recipe))
    launcher=ROOT/"scripts/pointneuron2/run_frozen_proposal_gat.ps1"; sources[str(launcher)]=base.file_digest(launcher)
    provenance={"manifest_sha256":base.file_digest(args.directory/"cache/manifest.json"),"sources":sources,
                "backend":{"device":str(device),"torch":str(torch.__version__),"threads":torch.get_num_threads(),
                    "deterministic_algorithms":True,
                    "device_name":torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}}
    folder=args.directory/("model" if args.action == "train" else args.action)
    if args.action == "resume":
        if args.checkpoint is None or not args.checkpoint.resolve().is_relative_to(args.directory) or not (args.checkpoint.parent.name == "model" or args.checkpoint.parent.name.startswith("resume_")):
            raise ValueError("Explicit checkpoint from this stage required")
        folder=args.directory/f"resume_{args.checkpoint.stem}"
    if folder.exists(): raise FileExistsError("Existing action outputs cannot be overwritten")
    if args.action in ("timing","train","resume"):
        checked=base.read(args.directory/"check/report.json")
        if (checked["status"] != "completed" or checked["recipe"] != recipe or checked["provenance"]["sources"] != sources
                or checked["provenance"]["manifest_sha256"] != provenance["manifest_sha256"]):
            raise ValueError("Expected matching completed CPU check")
    if args.action in ("train","resume"):
        timed=base.read(args.directory/"timing/report.json")
        if timed["status"] != "completed" or timed["recipe"] != recipe or timed["provenance"] != provenance:
            raise ValueError("Expected completed timing on the same backend/source/cache")
    folder.mkdir(parents=True)
    report={"schema":SCHEMA,"status":"running","action":args.action,"recipe":recipe,"provenance":provenance,
        "scope":"oracle-selected frozen TRAINING nodes; not automatic reconstruction","optimizer_updates":0,"evaluations":[],"upstream_forwards":0}
    base.write(folder/"report.json",report); start_time=time.perf_counter()
    try:
        for path in sources:
            target=folder/"sources"/Path(path).relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
        model,optimizer=fresh(recipe,device); start=0
        if args.action == "resume":
            model,optimizer,start=restore(torch.load(args.checkpoint,map_location="cpu",weights_only=True),recipe,provenance,device,resume=True)
            report.update(resumed_from=str(args.checkpoint),optimizer_updates=start)
        report.update(model_spec=model.model_spec(),trainable_parameters=sum(p.numel() for p in model.parameters()))
        report["initial_state_sha256"]={k:base.tensor_digest(v) for k,v in model.state_dict().items()}
        print(f"ACTION={args.action} device={provenance['backend']['device_name']} cap=4000 | frozen oracle-selected training nodes",flush=True)
        if args.action in ("check","timing"):
            report["evaluations"].append(evaluate(model,records,recipe,device,0,folder))
            durations=[]
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
            for j in range(4 if args.action == "check" else recipe["timing_updates"]):
                if device.type == "cuda": torch.cuda.synchronize()
                one=time.perf_counter()
                if args.action == "check":
                    optimizer.zero_grad(set_to_none=True)
                    scores=model(packet(records[j],device),candidates(records[j],recipe,device))
                    masked_edge_loss(scores.edge_logits,records[j]["labels"].to(device)).total.backward()
                    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise ValueError("Missing/nonfinite gradients")
                else:
                    metrics=update(model,optimizer,records[j%4],recipe,device)
                if device.type == "cuda": torch.cuda.synchronize()
                durations.append(time.perf_counter()-one)
                print(f"{'BACKWARD_ONLY' if args.action == 'check' else 'THROWAWAY'} step={j+1} seconds={durations[-1]:.3f}",flush=True)
            step=0 if args.action == "check" else recipe["timing_updates"]
            temporary=folder/"throwaway.pt"; torch.save(checkpoint(model,optimizer,step,recipe,provenance,purpose=args.action),temporary)
            loaded,_,_=restore(torch.load(temporary,map_location="cpu",weights_only=True),recipe,provenance,device,resume=False)
            report["reload_max_delta"]=reload_delta(model,loaded,records,recipe,device); temporary.unlink()
            report.update(throwaway_optimizer_updates=step,retained_trained_checkpoint=False)
            if args.action == "check":
                if {k:base.tensor_digest(v) for k,v in model.state_dict().items()} != report["initial_state_sha256"]:
                    raise ValueError("CPU backward-only check changed model parameters")
                report["model_parameters_unchanged"]=True
            if args.action == "timing":
                estimate=float(np.mean(durations[4:]))*recipe["updates"]
                report.update(estimated_update_seconds=estimate,planning_range_seconds=[estimate,1.5*estimate+60],
                    peak_memory_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
                print(f"ESTIMATE {estimate/60:.1f}-{(1.5*estimate+60)/60:.1f} minutes plus evaluation/checkpoint IO; training is separate",flush=True)
        else:
            clipped=0; maximum_norm=0.
            for step in range(start,recipe["updates"]+1):
                if step > start:
                    metrics=update(model,optimizer,records[(step-1)%4],recipe,device)
                    clipped+=metrics["preclip_norm"] > recipe["gradient_clip"]; maximum_norm=max(maximum_norm,metrics["preclip_norm"])
                    report.update(optimizer_updates=step,clipped_updates_in_this_invocation=clipped,maximum_preclip_norm_in_this_invocation=maximum_norm)
                    if step%50 == 0: print(f"UPDATE {step}/4000 crop={(step-1)%4} loss={metrics['loss']:.6f} elapsed={time.perf_counter()-start_time:.1f}s",flush=True)
                if step in recipe["snapshots"] and (args.action != "resume" or step > start):
                    snapshot(model,optimizer,step,recipe,provenance,report,folder,records,device)
            loaded,_,_=restore(torch.load(folder/"step_4000.pt",map_location="cpu",weights_only=True),recipe,provenance,device,resume=False)
            report.update(reload_max_delta=reload_delta(model,loaded,records,recipe,device),graph_gate=report["evaluations"][-1]["graph_gate"])
        if report["reload_max_delta"] != 0 or fingerprints(records) != before: raise ValueError("Checkpoint reload/frozen cache changed")
        base.verify_hashes(manifest["protected_files_sha256"]); base.verify_hashes(sources)
        report.update(status="completed",elapsed_seconds=time.perf_counter()-start_time,frozen_tensors_unchanged=True,
            next_step="Review graph and geometry results; no automatic extension, selection policy promotion or joint fit.")
        base.write(folder/"report.json",report); print(f"COMPLETED {args.action}; report={folder/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=str(exc))
        (folder/"error_traceback.txt").write_text(traceback.format_exc(),encoding="utf-8"); base.write(folder/"report.json",report); raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action",choices=("prepare","check","timing","train","resume"),required=True)
    parser.add_argument("--recipe",type=Path,default=ROOT/"configs/pointneuron2/frozen_proposal_gat.json")
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--checkpoint",type=Path)
    run(parser.parse_args())

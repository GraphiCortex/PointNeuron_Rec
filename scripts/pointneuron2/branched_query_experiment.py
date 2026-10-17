"""Separate preparation, CPU checks, GPU timing and a capped four-crop fit."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from pointneuron.data.connectivity_targets import assign_edge_labels, full_reference_components
from pointneuron.data.curve_targets import curve_geometry, patch_targets, target_arrays
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from pointneuron.graph.proposal_graph import ProposalGraphInput, build_candidate_graph, tensor_digest
from pointneuron.graph.representation_oracle import target_graph, topology
from pointneuron.models.pointnext_query_cropnorm import PointNextCropNormQueryModel, POLICY, load_cropnorm_checkpoint
from pointneuron.models.query_fit_state import FIT_SCHEMA, fit_checkpoint, restore_fit
from pointneuron.models.query_loss import query_set_loss
from scripts.pointneuron2.field_sampling_metrics import reference_geometry, measure_support
from scripts.pointneuron2.pointnext_query_experiment import pack_geometry, unpack_geometry


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Artifact path escapes declared root")
    return path


def verify_hashes(protected):
    for path, expected in protected.items():
        if file_digest(path) != expected:
            raise ValueError(f"Protected source/cache changed: {path}")


def recipe_at(path):
    recipe = read(path)
    expected = read(ROOT / "configs/pointneuron2/branched_query_fit.json")
    if (recipe != expected or recipe["schema"] != FIT_SCHEMA or recipe["normalization"] != POLICY
            or recipe["training_crops"] != [[0, 0], [0, 1], [92, 4], [86, 3]]
            or recipe["updates"] != 4000 or recipe["snapshots"] != [0, 500, 1000, 2000, 4000]
            or recipe["batch_size"] != 1 or recipe["augmentation"] is not False):
        raise ValueError("Recipe differs from the agreed bounded experiment")
    return recipe


def source_hashes(args):
    paths = {Path(__file__).resolve(), args.recipe.resolve(), ROOT / "scripts/pointneuron2/run_branched_query.ps1"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == ".py" and any(path.is_relative_to(ROOT / d) for d in ("src", "scripts")):
                paths.add(path)
    return {str(path): file_digest(path) for path in sorted(paths)}


def prepare(args, recipe):
    cache = args.experiment_dir / "cache"
    if cache.exists():
        raise FileExistsError("Preserve cache; choose a fresh experiment directory")
    audit_path = safe(ROOT, recipe["audit_summary"])
    if file_digest(audit_path) != recipe["audit_summary_sha256"]:
        raise ValueError("Expected the reviewed upstream audit")
    audit = read(audit_path)
    if audit["status"] != "completed" or audit["optimizer_updates"] != 0:
        raise ValueError("Incomplete source audit")
    protected = dict(audit["protected_files_sha256"])
    protected[str(audit_path)] = recipe["audit_summary_sha256"]
    verify_hashes(protected)
    partition_path = ROOT / "configs/pointneuron2/partition_seed0.json"
    partition = read(partition_path)
    train = {(r["sample_index"], r["sample_id"]) for r in partition["partitions"]["train"]}
    other = {(r["sample_index"], r["sample_id"]) for name in ("val", "test") for r in partition["partitions"][name]}
    if train & other:
        raise ValueError("Partition overlap")
    target_audit = read(ROOT / "docs/pointneuron2/query_target_counts_seed0/summary.json")
    lookup = {(r["sample_index"], r["patch_index"]): r for r in audit["records"]}
    cache.mkdir(parents=True)
    manifest = {"schema": FIT_SCHEMA + "-cache", "status": "running", "recipe": recipe,
                "partition_sha256": file_digest(partition_path), "source_audit_sha256": file_digest(audit_path),
                "records": [], "optimizer_updates": 0, "field_forwards": 0, "query_forwards": 0}
    try:
        for i, key in enumerate(recipe["training_crops"]):
            row = lookup[tuple(key)]
            if row["partition"] != "train" or (key[0], row["sample_id"]) not in train:
                raise ValueError("Selected crop is not training data")
            stem = f"sample_{key[0]:04d}_patch_{key[1]:03d}"
            cloud_path = audit_path.parent / (stem + "_cloud.npz")
            if file_digest(cloud_path) != row["cloud_sha256"]:
                raise ValueError("Frozen input cloud changed")
            with np.load(cloud_path, allow_pickle=False) as z:
                points, indices = z["points"].copy(), z["voxel_indices"].copy()
                shape = z["probability"].shape
                meta = json.loads(str(z["metadata"]))
            if (points.shape != (recipe["expected_input_counts"][i], 4) or len(np.unique(indices)) != len(points)
                    or not np.isfinite(points).all() or meta["partition"] != "train"
                    or meta["sample_index"] != key[0] or meta["patch_index"] != key[1]):
                raise ValueError("Frozen cloud count, coordinates or training identity differs")
            source = next(s for s in target_audit["sources"] if s["sample_index"] == key[0])
            swc = safe(args.data_root, source["path"])
            if file_digest(swc) != source["sha256"] or meta["swc_sha256"] != source["sha256"]:
                raise ValueError("Training reference differs")
            protected[str(swc)] = source["sha256"]
            tree = parse_swc(swc)
            targets = patch_targets(curve_geometry(tree), meta["patch_center"], recipe["patch_radius"], recipe["target_spacing"])
            nodes, roles = target_arrays(targets, recipe["query_capacity"])
            graph = target_graph(tree, [(meta["patch_center"], recipe["patch_radius"])], recipe["target_spacing"])
            if (len(nodes) != recipe["expected_target_counts"][i] or graph.identities != [tuple(t["reference_identity"]) for t in targets]
                    or not np.allclose(nodes[:, 1:4], graph.xyz, atol=1e-4, rtol=0)):
                raise ValueError("Target count/order differs; no truncation or replacement crops")
            geometry = reference_geometry(tree, np.array(meta["origin_xyz"]), shape)
            origin = np.array(meta["origin_xyz"])
            geometry["full_segments"] += origin
            geometry["critical_points"] += origin
            for chain in geometry["chains"]:
                chain["probe"]["points"] += origin
            path = cache / (stem + ".npz")
            np.savez_compressed(path, points=points, voxel_indices=indices, nodes=nodes, roles=roles,
                target_edges=np.array(graph.edges, dtype=np.int64).reshape(-1, 2),
                target_components=full_reference_components(tree, graph.identities), junction_mask=topology(graph)["degree"] >= 3,
                geometry=np.array(json.dumps(pack_geometry(geometry))), metadata=np.array(json.dumps(meta)))
            manifest["records"].append({"sample_index": key[0], "patch_index": key[1], "sample_id": row["sample_id"],
                "partition": "train", "path": path.name, "sha256": file_digest(path),
                "source_cloud_sha256": row["cloud_sha256"], "point_tensor_sha256": row["point_tensor_sha256"],
                "input_count": len(points), "target_count": len(nodes), "visible_junctions": row["visible_junctions"]})
            print(f"PREPARED train={key} exact frozen input={len(points)} targets={len(nodes)} junctions={row['visible_junctions']}", flush=True)
        verify_hashes(protected)
        manifest.update(status="completed", protected_files_sha256=protected)
        write(cache / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=str(exc)); write(cache / "manifest.json", manifest)
        raise


def load_records(args, recipe):
    path = args.experiment_dir / "cache/manifest.json"
    manifest = read(path)
    if (manifest.get("schema") != FIT_SCHEMA + "-cache" or manifest.get("status") != "completed"
            or manifest.get("recipe") != recipe or len(manifest["records"]) != 4):
        raise ValueError("Expected completed, matching four-crop cache")
    verify_hashes(manifest["protected_files_sha256"])
    records = []
    for i, row in enumerate(manifest["records"]):
        path = safe(args.experiment_dir / "cache", row["path"])
        if row["partition"] != "train" or [row["sample_index"], row["patch_index"]] != recipe["training_crops"][i] or file_digest(path) != row["sha256"]:
            raise ValueError("Changed/nontraining cache record")
        with np.load(path, allow_pickle=False) as z:
            record = {k: z[k].copy() for k in ("points", "nodes", "target_edges", "target_components", "junction_mask")}
            record["meta"] = json.loads(str(z["metadata"]))
            record["geometry"] = unpack_geometry(json.loads(str(z["geometry"])))
        if (record["points"].shape != (recipe["expected_input_counts"][i], 4)
                or record["nodes"].shape != (recipe["expected_target_counts"][i], 6)
                or tensor_digest(torch.from_numpy(record["points"])) != row["point_tensor_sha256"]
                or not len(record["nodes"]) or not np.isfinite(record["nodes"]).all()):
            raise ValueError("Invalid cloud/target cache; empty annotation is not background")
        record["identity"] = row
        records.append(record)
    return manifest, records


def forward(model, record, recipe, device):
    # Only image-derived points, image scale and crop center reach the model.
    return model(torch.from_numpy(record["points"])[None].to(device),
                 torch.tensor([record["meta"]["patch_center"]], dtype=torch.float32, device=device),
                 torch.tensor([record["meta"]["image_scale"]], dtype=torch.float32, device=device), recipe["patch_radius"])


def loss_for(output, record, recipe, device):
    nodes = torch.from_numpy(record["nodes"])[None].to(device)
    return query_set_loss(output, nodes, torch.ones(nodes.shape[:2], dtype=torch.bool, device=device), **recipe["query_loss"])


def update(model, optimizer, record, recipe, device):
    model.train(); optimizer.zero_grad(set_to_none=True)
    prediction = forward(model, record, recipe, device)
    loss = loss_for(prediction, record, recipe, device)
    if loss.supervised_crops != 1 or not bool(torch.isfinite(loss.total)):
        raise ValueError("Expected one finite supervised crop loss")
    loss.total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError("Nonfinite updated model")
    return {"total": float(loss.total.detach()), "classification": float(loss.objectness),
            "center": float(loss.center), "radius": float(loss.radius), "gradient_norm_before_clip": float(norm)}


def advancement_gates(row, recipe):
    stats = row["connectivity_supervision"]
    geometry = (row["trusted_target_fraction"] >= recipe["minimum_trusted_target_fraction"]
        and stats["attainable_recall_all_target_edges"] is not None
        and stats["attainable_recall_all_target_edges"] >= recipe["minimum_attainable_edge_recall"]
        and row["trusted_visible_junctions"] == row["visible_junctions"]
        and stats["positive"] > 0 and stats["negative"] > 0)
    selection = (row["accepted_assigned"] / row["targets"] >= recipe["minimum_assigned_acceptance"]
        and row["accepted_unassigned"] <= int(np.floor(recipe["maximum_unassigned_acceptance_fraction"] * row["targets"]))
        and row["accepted_curve_support_6"] is not None
        and row["accepted_curve_support_6"] >= recipe["minimum_accepted_curve_support_6"])
    return {"geometry_gate_passed": bool(geometry), "selection_gate_passed": bool(selection),
            "combined_learnability_gate_passed": bool(geometry and selection)}


def evaluate(model, records, recipe, device, directory, step):
    model.eval(); rows = []
    with torch.no_grad():
        for i, record in enumerate(records):
            prediction = forward(model, record, recipe, device)
            loss = loss_for(prediction, record, recipe, device)
            cpu = type(prediction)(**{f.name: getattr(prediction, f.name).detach().cpu() for f in fields(prediction)})
            packet = ProposalGraphInput(cpu.center_proposals[0], cpu.radius[0], cpu.objectness_logits[0],
                cpu.local_centers[0], cpu.decoder_features[0], torch.arange(recipe["query_capacity"]),
                {"normalization_policy": POLICY, "scope": "training-only metric calculation"})
            candidates = build_candidate_graph(packet.positions, packet.node_ids,
                neighbors=recipe["candidate_neighbors"], radius=recipe["candidate_radius"])
            labels = assign_edge_labels(packet, candidates, record["nodes"][:, 1:4], record["target_edges"],
                record["target_components"], loss.assignments[0], maximum_match_error=recipe["trusted_match_tolerance"])
            pi, ti = (np.asarray(a, dtype=np.int64) for a in loss.assignments[0])
            distance = np.linalg.norm(cpu.center_proposals[0].numpy()[pi].astype(float) - record["nodes"][ti, 1:4], axis=1)
            trusted = distance <= recipe["trusted_match_tolerance"]
            accepted = cpu.objectness_logits.softmax(-1)[0, :, 1].numpy() >= recipe["score_threshold"]
            support = measure_support(packet.positions.numpy()[accepted], record["geometry"])
            summary = support["reference_to_points"]
            edges = candidates.edges.numpy()
            required = labels.statistics["required_target_edges"]
            accepted_edges = int(((labels.labels.numpy() == 1) & accepted[edges].all(axis=1)).sum())
            row = {"sample_index": record["identity"]["sample_index"], "patch_index": record["identity"]["patch_index"],
                "input_count": len(record["points"]), "targets": len(ti), "assigned_count": len(pi),
                "trusted_target_fraction": int(trusted.sum()) / len(ti),
                "visible_junctions": int(record["junction_mask"].sum()),
                "trusted_visible_junctions": int(record["junction_mask"][ti[trusted]].sum()),
                "accepted_queries": int(accepted.sum()), "accepted_assigned": int(accepted[pi].sum()),
                "accepted_trusted_assigned": int(accepted[pi[trusted]].sum()),
                "accepted_unassigned": int(accepted.sum() - accepted[pi].sum()),
                "accepted_curve_support_6": None if summary is None else summary["within_voxels"]["6"],
                "accepted_curve_support": support, "candidate_edges": len(edges), "connectivity_supervision": labels.statistics,
                "attainable_edge_recall_after_node_cutoff": None if not required else accepted_edges / required,
                "matched_distance_mean": float(distance.mean()), "matched_distance_max": float(distance.max()),
                "radius_mae_floored_target": float(np.abs(packet.radii.numpy()[pi, 0] - np.maximum(record["nodes"][ti, 4], 1)).mean()),
                "loss": float(loss.total), "classification_loss": float(loss.objectness), "center_loss": float(loss.center),
                "assignments": [pi.tolist(), ti.tolist()]}
            row.update(advancement_gates(row, recipe)); rows.append(row)
            np.savez_compressed(directory / f"step_{step:04d}_crop_{i}.npz",
                centers=cpu.center_proposals.numpy(), logits=cpu.objectness_logits.numpy(), radii=cpu.radius.numpy(),
                decoder_features=cpu.decoder_features.numpy(), pred_indices=pi, target_indices=ti,
                candidate_edges=edges, edge_labels=labels.labels.numpy())
            print(f"EVAL step={step} train={row['sample_index']}/{row['patch_index']} trusted={int(trusted.sum())}/{len(ti)} "
                  f"junctions={row['trusted_visible_junctions']}/{row['visible_junctions']} accepted={row['accepted_assigned']}+{row['accepted_unassigned']} "
                  f"edge_ceiling={labels.statistics['attainable_recall_all_target_edges']} gate={row['combined_learnability_gate_passed']}", flush=True)
    return {"step": step, "crops": rows, "combined_learnability_gate_passed": all(r["combined_learnability_gate_passed"] for r in rows)}


def backend_for(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU training fallback")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {"device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": False, "cudnn_tf32": False, "matmul_tf32": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "threads": torch.get_num_threads()}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def check_reload(model, checkpoint, records, recipe, device):
    reloaded = load_cropnorm_checkpoint(checkpoint["inference_checkpoint"], device)
    model.eval(); maximum = 0.
    with torch.no_grad():
        for record in records:
            a, b = forward(model, record, recipe, device), forward(reloaded, record, recipe, device)
            for field in fields(a):
                maximum = max(maximum, float((getattr(a, field.name) - getattr(b, field.name)).abs().max()))
    if maximum > 1e-5:
        raise RuntimeError("Saved query checkpoint reload mismatch")
    return maximum


def run(args):
    recipe = recipe_at(args.recipe)
    args.experiment_dir = args.experiment_dir.resolve()
    if not args.experiment_dir.is_relative_to((ROOT / "tmp/pointneuron2").resolve()):
        raise ValueError("Generated outputs belong under ignored tmp/pointneuron2")
    torch.set_num_threads(recipe["torch_threads"])
    if args.action == "prepare":
        if args.device != "cpu":
            raise ValueError("Preparation is CPU-only with zero model forwards")
        return prepare(args, recipe)
    if (args.action == "check" and args.device != "cpu") or (args.action != "check" and args.device != "cuda"):
        raise ValueError("Checks are CPU-only; timing/training require explicit CUDA")
    manifest, records = load_records(args, recipe)
    backend = backend_for(args.device)
    provenance = {"cache_manifest_sha256": file_digest(args.experiment_dir / "cache/manifest.json"),
                  "source_files_sha256": source_hashes(args), "recipe_sha256": file_digest(args.recipe)}
    directory = args.experiment_dir / ("cpu_check" if args.action == "check" else "timing" if args.action == "timing" else "model")
    if args.action == "resume":
        if (args.checkpoint is None or not args.checkpoint.resolve().is_relative_to(args.experiment_dir)
                or not (args.checkpoint.parent.name == "model" or args.checkpoint.parent.name.startswith("resume_"))):
            raise ValueError("Resume requires an explicit checkpoint from this experiment's model folder")
        directory = args.experiment_dir / f"resume_{args.checkpoint.stem}"
    if directory.exists():
        raise FileExistsError("Preserve existing action outputs; use a fresh experiment or resume checkpoint")
    if args.action in ("train", "resume"):
        timing = read(args.experiment_dir / "timing/report.json")
        if timing.get("status") != "completed" or timing.get("backend") != backend or timing.get("provenance") != provenance:
            raise ValueError("Run and inspect matching GPU timing before training")
    directory.mkdir(parents=True)
    report = {"schema": FIT_SCHEMA, "status": "running", "action": args.action, "recipe": recipe,
              "backend": backend, "provenance": provenance, "optimizer_updates": 0, "evaluations": [],
              "scope": "four fixed training crops; no development/test, GATv2 training or full reconstruction"}
    write(directory / "report.json", report)
    started = time.perf_counter()
    try:
        random.seed(recipe["seed"]); np.random.seed(recipe["seed"]); torch.manual_seed(recipe["seed"])
        if args.device == "cuda":
            torch.cuda.manual_seed_all(recipe["seed"]); torch.cuda.reset_peak_memory_stats()
        if args.action == "resume":
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model, optimizer, step = restore_fit(checkpoint, recipe=recipe, provenance=provenance, backend=backend, device=args.device)
            report.update(resumed_from=str(args.checkpoint.resolve()), resumed_checkpoint_sha256=file_digest(args.checkpoint),
                          resumed_at_step=step, optimizer_updates=step)
        else:
            model = PointNextCropNormQueryModel(recipe["seed"]).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
            step = 0
        report["initial_model_sha256"] = {k: tensor_digest(v) for k, v in model.state_dict().items()}
        print(f"ACTION={args.action} device={backend['device_name']} | fresh seed=0 except explicit resume | cap={recipe['updates']} | output={directory}", flush=True)
        source_directory = directory / "sources"
        for path in provenance["source_files_sha256"]:
            target = source_directory / Path(path).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        if args.action == "check":
            evaluation = evaluate(model, records, recipe, "cpu", directory, 0)
            report["evaluations"].append(evaluation)
            checkpoint = fit_checkpoint(model, optimizer, step=0, recipe=recipe, provenance=provenance, backend=backend)
            path = directory / "untrained_reload_check.pt"
            torch.save(checkpoint, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            report["checkpoint_reload_max_delta"] = check_reload(model, loaded, records, recipe, "cpu")
            restored, _, restored_step = restore_fit(loaded, recipe=recipe, provenance=provenance, backend=backend, device="cpu")
            if restored_step != 0 or any(not torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()):
                raise RuntimeError("Initial optimizer/RNG state reload failed")
            report["model_state_unchanged"] = report["initial_model_sha256"] == {k: tensor_digest(v) for k, v in model.state_dict().items()}
            if not report["model_state_unchanged"]:
                raise RuntimeError("CPU inference changed model state")
            report["checkpoint_retained_as_untrained_check_only"] = True
        elif args.action == "timing":
            durations = []
            for j in range(recipe["timing_updates"]):
                sync(args.device); one = time.perf_counter()
                metrics = update(model, optimizer, records[j % 4], recipe, args.device)
                report["throwaway_optimizer_updates"] = j + 1
                sync(args.device); elapsed = time.perf_counter() - one
                if j >= 4:
                    durations.append(elapsed)
                print(f"THROWAWAY update={j+1}/8 crop={j%4} loss={metrics['total']:.5f} seconds={elapsed:.3f}", flush=True)
            one = time.perf_counter()
            evaluate(model, records, recipe, args.device, directory, 8)
            sync(args.device)
            eval_seconds = time.perf_counter() - one
            checkpoint = fit_checkpoint(model, optimizer, step=8, recipe=recipe, provenance=provenance, backend=backend)
            path = directory / "throwaway_state_check.pt"
            torch.save(checkpoint, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            report["checkpoint_reload_max_delta"] = check_reload(model, loaded, records, recipe, args.device)
            _, restored_optimizer, restored_step = restore_fit(loaded, recipe=recipe, provenance=provenance, backend=backend, device=args.device)
            if restored_step != 8 or len(restored_optimizer.state) != len(optimizer.state):
                raise RuntimeError("Timing optimizer state reload failed")
            path.unlink()
            estimate = float(np.mean(durations)) * recipe["updates"] + eval_seconds * len(recipe["snapshots"])
            report.update(throwaway_optimizer_updates=8, retained_fit_optimizer_updates=0, checkpoint_retained=False,
                          timed_update_seconds=durations, evaluation_seconds=eval_seconds,
                          estimated_fit_seconds=estimate, planning_range_seconds=[estimate, 1.5 * estimate])
            print(f"ESTIMATE {estimate/60:.1f}-{1.5*estimate/60:.1f} minutes plus checkpoint IO/system variability. Training is a separate command.", flush=True)
        else:
            start_step = step
            for current in range(start_step, recipe["updates"] + 1):
                if current > start_step:
                    crop = (current - 1) % len(records)
                    metrics = update(model, optimizer, records[crop], recipe, args.device)
                    report["optimizer_updates"] = current
                    if current % recipe["log_every"] == 0:
                        sync(args.device)
                        print(f"UPDATE={current}/{recipe['updates']} crop={crop} loss={metrics['total']:.5f} "
                              f"class={metrics['classification']:.5f} center={metrics['center']:.5f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                if current in recipe["snapshots"] and (args.action != "resume" or current > start_step):
                    evaluation = evaluate(model, records, recipe, args.device, directory, current)
                    report["evaluations"].append(evaluation)
                    checkpoint = fit_checkpoint(model, optimizer, step=current, recipe=recipe, provenance=provenance, backend=backend)
                    torch.save(checkpoint, directory / f"step_{current:04d}.pt")
                    write(directory / "report.json", report)
            final = directory / "step_4000.pt"
            loaded = torch.load(final, map_location="cpu", weights_only=True)
            report["checkpoint_reload_max_delta"] = check_reload(model, loaded, records, recipe, args.device)
            report.update(final_checkpoint_sha256=file_digest(final), optimizer_updates=recipe["updates"],
                          combined_learnability_gate_passed=report["evaluations"][-1]["combined_learnability_gate_passed"])
        verify_hashes(manifest["protected_files_sha256"])
        verify_hashes(provenance["source_files_sha256"])
        report.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device == "cuda" else None,
                      cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if args.device == "cuda" else None,
                      next_step="Review measured gates; no automatic extension, threshold change, GATv2 fit or model promotion.")
        write(directory / "report.json", report)
        print(f"COMPLETED {args.action}; report={directory / 'report.json'}", flush=True)
        return report
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), elapsed_seconds=time.perf_counter()-started)
        write(directory / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "check", "timing", "train", "resume"), required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=ROOT / "configs/pointneuron2/branched_query_fit.json")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data/gold166")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    run(parser.parse_args())

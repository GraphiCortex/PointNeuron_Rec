"""Training-only CPU inference audit; no optimizer or GATv2 model is created."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from pointneuron.data.connectivity_targets import assign_edge_labels, full_reference_components
from pointneuron.data.curve_targets import curve_geometry, patch_targets, target_arrays
from pointneuron.data.field_guided_sampling import candidate_pools, select_cloud, local_xyz
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from pointneuron.graph.proposal_graph import from_query_output, build_candidate_graph, tensor_digest
from pointneuron.graph.representation_oracle import target_graph, topology
from pointneuron.models.centerline_unet import CenterlineUNet
from pointneuron.models.pointnext_query_cropnorm import load_cropnorm_checkpoint, POLICY
from pointneuron.models.query_loss import query_set_loss
from scripts.pointneuron2.field_sampling_metrics import reference_geometry, measure_support


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def within(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Source path escapes declared root")
    return path


def plot_case(raw, origin, points, graph, packet, trusted, accepted, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    predicted = packet.positions.numpy() - origin
    target = graph.xyz - origin
    cloud = points[:, :3] - origin
    display = np.clip(raw / max(float(np.quantile(raw, .995)), 1), 0, 1)
    for ax, axis, dims, name in [(axes[0], 0, (0, 1), "XY"), (axes[1], 1, (0, 2), "XZ")]:
        ax.imshow(display.max(axis=axis), cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.scatter(cloud[:, dims[0]], cloud[:, dims[1]], s=1, c="orange", alpha=.2, label="image cloud")
        for u, v in graph.edges:
            ax.plot(target[[u, v], dims[0]], target[[u, v], dims[1]], c="cyan", lw=.8)
        ax.scatter(target[:, dims[0]], target[:, dims[1]], s=12, marker="+", c="cyan", label="reference targets")
        ax.scatter(predicted[:, dims[0]], predicted[:, dims[1]], s=7, c="gray", alpha=.35, label="all 512 queries")
        ax.scatter(predicted[accepted, dims[0]], predicted[accepted, dims[1]], s=14, facecolors="none", edgecolors="magenta", label="accepted at .5")
        ax.scatter(predicted[trusted, dims[0]], predicted[trusted, dims[1]], s=24, marker="x", c="lime", label="trusted assigned query")
        ax.set_title(name); ax.set_xlabel("Local X voxel"); ax.set_ylabel("Local Y / Z voxel")
    axes[0].legend(fontsize=7)
    fig.suptitle("Frozen upstream inference; reference overlay for scoring only\nProjections conceal depth; no predicted graph edges are shown", fontsize=10)
    fig.savefig(path, dpi=130); plt.close(fig)


def run(args):
    started = time.perf_counter()
    cfg = read(args.config)
    if cfg["schema"] != "pointneuron2-upstream-connectivity-audit-v1" or cfg["device"] != "cpu":
        raise ValueError("Expected explicit CPU audit config")
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "tmp/pointneuron2").resolve()):
        raise ValueError("Generated caches belong under tmp/pointneuron2")
    if output.exists():
        raise FileExistsError("Preserve previous/failed evidence; choose a fresh output directory")
    protected = {}

    def protect(path, expected=None):
        path = Path(path).resolve()
        sha = file_digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f"Source digest changed: {path}")
        protected[str(path)] = sha
        return path

    protect(args.config)
    for key in ("partition", "target_audit", "original_cloud_manifest", "native_manifest", "field_checkpoint", "query_checkpoint"):
        protect(within(ROOT, cfg[key]), cfg[key + "_sha256"])
    protect(within(ROOT, cfg["query_recipe"]))
    # Snapshot every imported project source, including transitive diagnostic helpers.
    source_paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == ".py" and any(path.is_relative_to(ROOT / directory) for directory in ("src", "scripts")):
                source_paths.add(path)
    for path in source_paths:
        protect(path)
    partition, audit = (read(ROOT / cfg[key]) for key in ("partition", "target_audit"))
    train = {(r["sample_index"], r["sample_id"]) for r in partition["partitions"]["train"]}
    if any(train & {(r["sample_index"], r["sample_id"]) for r in partition["partitions"][part]}
           for part in ("val", "test")):
        raise ValueError("Partition overlap")
    native = read(ROOT / cfg["native_manifest"])
    original = read(ROOT / cfg["original_cloud_manifest"])
    if (native["status"] != "completed" or native["partition_sha256"] != cfg["partition_sha256"]
            or original["status"] != "completed" or audit["partition_sha256"] != cfg["partition_sha256"]):
        raise ValueError("Incomplete or mismatched source manifests")
    lookup = {(r["sample_index"], r["patch_index"]): r for r in native["records"] if r["partition"] == "train"}
    old_lookup = {(r["sample_index"], r["patch_index"]): r for r in original["records"] if r["partition"] == "train"}
    audited = {(r["sample_index"], r["patch_index"]): r for r in audit["records"]}
    torch.set_num_threads(cfg["torch_threads"])
    field_cp = torch.load(ROOT / cfg["field_checkpoint"], map_location="cpu", weights_only=True)
    if field_cp["manifest_sha256"] != cfg["native_manifest_sha256"] or field_cp["epoch"] != 30:
        raise ValueError("Field checkpoint identity differs")
    field = CenterlineUNet(field_cp["config"]["base_channels"]).eval().requires_grad_(False)
    field.load_state_dict(field_cp["model"], strict=True)
    query_cp = torch.load(ROOT / cfg["query_checkpoint"], map_location="cpu", weights_only=True)
    if query_cp["step"] != 2000 or query_cp["recipe"]["training_crops"] != [[0, 0], [0, 1]]:
        raise ValueError("Expected two-crop upstream checkpoint")
    model = load_cropnorm_checkpoint(query_cp).requires_grad_(False)
    initial_state = {key: tensor_digest(value) for key, value in model.state_dict().items()}
    output.mkdir(parents=True)
    report = {"schema": cfg["schema"], "status": "running", "config": cfg,
              "scope": "four training crops; CPU inference and reference scoring; no training or GATv2 forwards",
              "backend": {"device": "cpu", "torch": str(torch.__version__), "threads": torch.get_num_threads()},
              "normalization_policy": POLICY, "confidence_policy": "original 2000-update query head; no confidence-replay overlay",
              "optimizer_updates": 0, "field_inference_passes": 0, "pointnext_inference_passes": 0,
              "development_cache_reads": 0, "test_cache_reads": 0, "records": []}
    write(output / "summary.json", report)
    try:
        frozen = []
        # Freeze every cloud before opening any SWC. Only selected training NPZs are opened;
        # the general field loader would validate/open development caches, so it is not used.
        for case in cfg["cases"]:
            key = (case["sample_index"], case["patch_index"])
            row = lookup[key]
            if (row["sample_index"], row["sample_id"]) not in train or not row["supervised"]:
                raise ValueError("Case is not supervised training data")
            native_path = protect(within((ROOT / cfg["native_manifest"]).parent, row["path"]), row["sha256"])
            with np.load(native_path, allow_pickle=False) as z:
                raw, image, valid = (z[k].copy() for k in ("raw_image", "image", "valid"))
                meta = json.loads(str(z["metadata"]))
            if (any(meta[k] != row[k] for k in ("sample_index", "sample_id", "patch_index", "partition"))
                    or not np.array_equal(raw.astype(np.float32) / meta["image_scale"], image)
                    or meta["patch_center"] != audited[key]["patch_center"]):
                raise ValueError("Image scaling, training identity or crop box differs")
            if case["role"] == "fitted_control":
                old = old_lookup[key]
                path = protect(within((ROOT / cfg["original_cloud_manifest"]).parent, old["points_path"]), old["points_sha256"])
                with np.load(path, allow_pickle=False) as z:
                    points, indices, probability = (z[k].copy() for k in ("points", "voxel_indices", "probability"))
                    old_meta = json.loads(str(z["metadata"]))
                if old_meta != meta:
                    raise ValueError("Original cloud native metadata changed")
                field_backend = "saved original CUDA/TF32 field; exact original cloud reused"
            else:
                with torch.inference_mode():
                    probability = field(torch.from_numpy(image)[None, None]).sigmoid()[0, 0].numpy()
                report["field_inference_passes"] += 1
                pools = candidate_pools(raw, valid, probability, meta["threshold"])
                n = min(cfg["input_budget"], len(pools.expanded))
                cloud = select_cloud(raw, valid, probability, meta["origin_xyz"], meta["threshold"], "expanded_random", n,
                                     cfg["sampling_seed"] + case["sample_index"] * 10000 + case["patch_index"], pools=pools)
                points, indices = cloud.points, cloud.voxel_indices
                field_backend = "new CPU float32 field; no CPU/GPU equality claim"
            pools = candidate_pools(raw, valid, probability, meta["threshold"])
            if (cfg["field_threshold"] != .5 or len(pools.foreground) != meta["foreground_count"]
                    or len(points) < 256 or len(points) != min(cfg["input_budget"], len(pools.expanded))
                    or len(np.unique(indices)) != len(points) or not np.isin(indices, pools.expanded).all()
                    or not np.array_equal(points[:, :3], local_xyz(indices, raw.shape) + np.asarray(meta["origin_xyz"]))
                    or not np.array_equal(points[:, 3], raw.ravel()[indices])):
                raise ValueError("Distinct cloud coordinates/intensities/admission differ")
            path = output / f"sample_{key[0]:04d}_patch_{key[1]:03d}_cloud.npz"
            np.savez_compressed(path, points=points, voxel_indices=indices, probability=probability, metadata=np.array(json.dumps(meta)))
            protect(path)
            frozen.append((case, row, meta, raw, points, pools, path, field_backend))
            print(f"FROZEN train={key} N={len(points)} pool={len(pools.expanded)}; no reference loaded", flush=True)
        del field
        for case, row, meta, raw, points, pools, cloud_path, backend in frozen:
            key = (case["sample_index"], case["patch_index"])
            source = next(r for r in audit["sources"] if r["sample_index"] == key[0])
            swc = protect(within(args.data_root, source["path"]), source["sha256"])
            if source["sha256"] != row["swc_sha256"] or source["sha256"] != meta["swc_sha256"]:
                raise ValueError("Reference identity differs")
            tree = parse_swc(swc)
            targets = patch_targets(curve_geometry(tree), meta["patch_center"], cfg["patch_radius_voxels"], cfg["target_spacing_voxels"])
            nodes, roles = target_arrays(targets, 512)
            graph = target_graph(tree, [(meta["patch_center"], cfg["patch_radius_voxels"])], cfg["target_spacing_voxels"])
            identities = [tuple(t["reference_identity"]) for t in targets]
            if identities != graph.identities or len(nodes) != case["expected_targets"] or not np.allclose(nodes[:, 1:4], graph.xyz, atol=1e-4, rtol=0):
                raise ValueError("Reference target order/count changed")
            with torch.inference_mode():
                prediction = model(torch.from_numpy(points)[None], torch.tensor([meta["patch_center"]], dtype=torch.float32),
                                   torch.tensor([meta["image_scale"]], dtype=torch.float32), cfg["patch_radius_voxels"])
                loss = query_set_loss(prediction, torch.from_numpy(nodes)[None], torch.ones(1, len(nodes), dtype=torch.bool),
                                      **read(ROOT / cfg["query_recipe"])["query_loss"])
            report["pointnext_inference_passes"] += 1
            packet = from_query_output(prediction, crop_center=meta["patch_center"], patch_radius=cfg["patch_radius_voxels"],
                                       provenance={"upstream_schema": query_cp["schema"], "normalization_policy": POLICY,
                                                   "checkpoint_sha256": cfg["query_checkpoint_sha256"], "cloud_sha256": file_digest(cloud_path),
                                                   "feature_sha256": tensor_digest(prediction.decoder_features[0]), "partition": "train"})
            candidates = build_candidate_graph(packet.positions, packet.node_ids, neighbors=cfg["candidate_neighbors"], radius=cfg["candidate_radius_voxels"])
            assignment = loss.assignments[0]
            labels = assign_edge_labels(packet, candidates, nodes[:, 1:4], graph.edges, full_reference_components(tree, identities), assignment,
                                        maximum_match_error=cfg["trusted_match_tolerance_voxels"])
            pred_idx, target_idx = (np.asarray(a, dtype=np.int64) for a in assignment)
            distances = np.linalg.norm(packet.positions.numpy()[pred_idx] - nodes[target_idx, 1:4], axis=1)
            trusted = distances <= cfg["trusted_match_tolerance_voxels"]
            accepted = packet.objectness_logits.softmax(-1)[:, 1].numpy() >= cfg["node_cutoff"]
            matched = set(map(int, pred_idx))
            junction_targets = set(np.flatnonzero(topology(graph)["degree"] >= 3).tolist())
            trusted_targets = set(map(int, target_idx[trusted]))
            required = labels.statistics["required_target_edges"]
            edge_array = candidates.edges.numpy()
            positive = labels.labels.numpy() == 1
            accepted_positive = int((positive & accepted[edge_array].all(axis=1)).sum())
            geometry_ok = (len(trusted_targets) / len(nodes) >= cfg["minimum_trusted_target_fraction"]
                           and labels.statistics["attainable_recall_all_target_edges"] is not None
                           and labels.statistics["attainable_recall_all_target_edges"] >= cfg["minimum_attainable_edge_recall"]
                           and (not cfg["require_all_visible_junctions"] or junction_targets <= trusted_targets)
                           and labels.statistics["positive"] > 0 and labels.statistics["negative"] > 0)
            origin = np.asarray(meta["origin_xyz"])
            geometry = reference_geometry(tree, origin, raw.shape)
            support = {name: measure_support(xyz, geometry) for name, xyz in (
                ("foreground_pool", local_xyz(pools.foreground, raw.shape)),
                ("expanded_pool", local_xyz(pools.expanded, raw.shape)),
                ("selected_cloud", points[:, :3] - origin),
                ("all_queries", packet.positions.numpy() - origin),
                ("accepted_queries", packet.positions.numpy()[accepted] - origin))}
            nearest = cKDTree(packet.positions.numpy()).query(nodes[:, 1:4], workers=1)[0]
            row_report = {**case, "partition": "train", "sample_id": row["sample_id"], "field_backend": backend,
                "input_count": len(points), "foreground_pool_count": len(pools.foreground), "expanded_pool_count": len(pools.expanded),
                "query_count": len(packet.positions), "candidate_edges": len(edge_array), "target_count": len(nodes),
                "visible_junctions": len(junction_targets), "trusted_visible_junctions": len(junction_targets & trusted_targets),
                "trusted_target_fraction": len(trusted_targets) / len(nodes), "connectivity_supervision": labels.statistics,
                "assignment_distance_voxels": {"mean": float(distances.mean()), "median": float(np.median(distances)), "max": float(distances.max())},
                "targets_with_any_query_within_2": int((nearest <= 2).sum()),
                "targets_with_any_query_within_6": int((nearest <= 6).sum()),
                "accepted_queries": int(accepted.sum()), "accepted_assigned": int(accepted[pred_idx].sum()),
                "accepted_trusted_assigned": int(accepted[pred_idx[trusted]].sum()),
                "accepted_unassigned": sum(int(i) not in matched for i in np.flatnonzero(accepted)),
                "attainable_edge_recall_after_node_cutoff": accepted_positive / required if required else None,
                "frozen_geometry_feasibility_passed": geometry_ok,
                "support": support, "cloud_sha256": file_digest(cloud_path), "point_tensor_sha256": tensor_digest(torch.from_numpy(points)),
                "feature_sha256": packet.provenance["feature_sha256"], "assignments": [a.tolist() for a in (pred_idx, target_idx)]}
            packet_path = output / f"sample_{key[0]:04d}_patch_{key[1]:03d}_packet.npz"
            np.savez_compressed(packet_path, **{name: getattr(packet, name).numpy() for name in
                ("positions", "radii", "objectness_logits", "local_centers", "decoder_features", "node_ids")},
                candidate_edges=edge_array, edge_labels=labels.labels.numpy(), target_nodes=nodes, target_roles=roles,
                target_edges=np.asarray(graph.edges).reshape(-1, 2), pred_indices=pred_idx, target_indices=target_idx,
                provenance=np.array(json.dumps(dict(packet.provenance))))
            protect(packet_path)
            plot_case(raw, origin, points, graph, packet, pred_idx[trusted], accepted, output / f"sample_{key[0]:04d}_patch_{key[1]:03d}.png")
            report["records"].append(row_report)
            write(output / "summary.json", report)
            print(f"SCORED train={key}: trusted={len(trusted_targets)}/{len(nodes)} junctions={len(junction_targets & trusted_targets)}/{len(junction_targets)} "
                  f"positive_candidates={labels.statistics['positive']}/{required} accepted={int(accepted.sum())} feasibility={geometry_ok}", flush=True)
        if initial_state != {key: tensor_digest(value) for key, value in model.state_dict().items()}:
            raise RuntimeError("Read-only upstream model state changed")
        for path, sha in protected.items():
            if file_digest(path) != sha:
                raise RuntimeError(f"Protected source/cache changed: {path}")
        branched = [r for r in report["records"] if r["role"] == "branched_feasibility"]
        if not branched:
            raise ValueError("No branched feasibility cases")
        report.update(status="completed", frozen_upstream_feasible_for_declared_cases=all(r["frozen_geometry_feasibility_passed"] for r in branched),
                      protected_files_sha256=protected, model_state_unchanged=True, elapsed_seconds=time.perf_counter() - started,
                      caveats=["Selected training cases; not generalization or reconstruction evaluation.",
                               "Original control clouds reused; new branched fields use CPU, not original CUDA/TF32 backend.",
                               "Exact target adjacency and two-voxel assignment trust rule define the reported edge ceiling.",
                               "Any-query proximity can reuse one query; only trusted one-to-one matches supervise edges.",
                               "Accepted unassigned queries are set-label extras, not proven biological false positives."])
        snapshot = output / "sources"
        for path in source_paths | {Path(args.config).resolve()}:
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        write(output / "summary.json", report)
        print(f"COMPLETED CPU audit; optimizer updates=0; frozen feasibility={report['frozen_upstream_feasible_for_declared_cases']}; {output}", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
        write(output / "summary.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pointneuron2/upstream_connectivity_audit.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())

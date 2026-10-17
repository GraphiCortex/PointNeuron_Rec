"""Read-only paired review: saved training predictions and CPU checkpoint loads only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.special import expit
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.pointneuron2 import branched_query_experiment as base
from scripts.pointneuron2.balanced_query_experiment import contract, SCHEMA
from scripts.pointneuron2.review_branched_query_fit import ranking_stats
from scripts.pointneuron2.field_sampling_metrics import measure_support, reference_distances
from pointneuron.data.curve_targets import clip_parameters


def candidates_and_labels(xyz, radius, trusted_mapping, truth):
    d2 = ((xyz[:, None] - xyz[None, :]) ** 2).sum(-1)
    pairs = set()
    for u in range(512):
        order = np.lexsort((np.arange(512), d2[u]))
        for v in np.union1d(order[order != u][:16], np.flatnonzero(d2[u] <= (radius+1e-6)**2)):
            if u != v:
                pairs.add(tuple(sorted((u, int(v)))))
    edges = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    required = {tuple(sorted(map(int, edge))) for edge in truth["target_edges"]}
    labels = []
    for u, v in edges:
        a, b = trusted_mapping.get(int(u)), trusted_mapping.get(int(v))
        if a is None or b is None:
            labels.append(-1)
        elif tuple(sorted((a, b))) in required:
            labels.append(1)
        elif truth["target_components"][a] >= 0 and truth["target_components"][a] == truth["target_components"][b]:
            labels.append(0)
        else:
            labels.append(-1)
    return edges, np.asarray(labels)


def review(directory, fit, original_recipe, balanced):
    recipe = fit["recipe"]
    assert fit["status"] == "completed" and fit["optimizer_updates"] == 4000
    assert fit["checkpoint_reload_max_delta"] == 0
    base.verify_hashes(fit["provenance"]["source_files_sha256"])
    for path, digest in fit["provenance"]["source_files_sha256"].items():
        assert base.file_digest(directory/"model/sources"/Path(path).relative_to(ROOT)) == digest
    manifest, records = base.load_records(SimpleNamespace(experiment_dir=directory), original_recipe)
    assert base.file_digest(directory/"cache/manifest.json") == fit["provenance"]["cache_manifest_sha256"]
    rows, checkpoints, prediction_hashes = [], [], {}
    mappings = [None]*4
    for evaluation in fit["evaluations"]:
        step = evaluation["step"]
        for i, (r, truth) in enumerate(zip(evaluation["crops"], records)):
            path = directory/f"model/step_{step:04d}_crop_{i}.npz"
            prediction_hashes[path.name] = base.file_digest(path)
            with np.load(path, allow_pickle=False) as saved:
                xyz, logits = saved["centers"][0].astype(float), saved["logits"][0].astype(float)
                pi, ti = saved["pred_indices"], saved["target_indices"]
                saved_edges, saved_labels = saved["candidate_edges"], saved["edge_labels"]
            assert [pi.tolist(), ti.tolist()] == r["assignments"]
            positive = np.zeros(512, dtype=bool); positive[pi] = True
            difference = logits[:, 1] - logits[:, 0]
            accepted = difference >= 0
            distance = np.linalg.norm(xyz[pi] - truth["nodes"][ti, 1:4], axis=-1)
            trusted = distance <= recipe["trusted_match_tolerance"]
            assert int(trusted.sum()) == r["connectivity_supervision"]["trusted_matches"]
            assert int(accepted[pi].sum()) == r["accepted_assigned"]
            assert int(accepted[~positive].sum()) == r["accepted_unassigned"]
            assert int(accepted[pi[trusted]].sum()) == r["accepted_trusted_assigned"]
            assert abs(distance.mean() - r["matched_distance_mean"]) < 1e-8
            assert int(truth["junction_mask"][ti[trusted]].sum()) == r["trusted_visible_junctions"]
            confidence = ranking_stats(difference, positive, recipe["query_loss"]["positive_class_weight"])
            reference_ce = r["reference_weighted_classification_loss"] if balanced else r["classification_loss"]
            assert abs(confidence["actual_weighted_ce"] - reference_ce) < 1e-6
            balanced_ce = .5*(np.logaddexp(0, -difference[positive]).mean()+np.logaddexp(0, difference[~positive]).mean())
            if balanced:
                assert abs(balanced_ce-r["balanced_classification_loss"]) < 1e-6
            mapping = {int(t): int(p) for p, t in zip(pi, ti)}
            churn = None if mappings[i] is None else sum(mapping[t] != mappings[i][t] for t in mapping)
            mappings[i] = mapping
            trusted_mapping = {int(p): int(t) for p, t in zip(pi[trusted], ti[trusted])}
            arms = []
            for radius in (8., 12.):
                edges, labels = candidates_and_labels(xyz, radius, trusted_mapping, truth)
                counts = {name: int((labels == value).sum()) for name, value in (("positive", 1), ("negative", 0), ("unknown", -1))}
                required = len(truth["target_edges"])
                after = int(((labels == 1) & accepted[edges].all(-1)).sum())
                if radius == 8:
                    assert np.array_equal(edges, saved_edges) and np.array_equal(labels, saved_labels)
                    assert all(counts[k] == r["connectivity_supervision"][k] for k in counts)
                    assert abs(after/required-r["attainable_edge_recall_after_node_cutoff"]) < 1e-10
                if balanced:
                    reported = r["candidate_margin_comparison"][len(arms)]
                    assert radius == reported["radius"] and len(edges) == reported["candidate_edges"]
                    assert all(counts[k] == reported["connectivity_supervision"][k] for k in counts)
                    assert abs(after/required-reported["edge_ceiling_after_node_cutoff"]) < 1e-10
                arms.append(dict(radius=radius, candidate_edges=len(edges), **counts,
                    required_target_edges=required, edge_ceiling_before_cutoff=counts["positive"]/required,
                    edges_after_cutoff=after, edge_ceiling_after_cutoff=after/required))
            support = measure_support(xyz[accepted], truth["geometry"])["reference_to_points"]
            coverage = None if support is None else support["within_voxels"]["6"]
            assert coverage == r["accepted_curve_support_6"]
            assert base.advancement_gates(r, recipe) == {k: r[k] for k in base.advancement_gates(r, recipe)}
            item = {k: r[k] for k in ("sample_index", "patch_index", "targets", "accepted_assigned", "accepted_unassigned",
                "accepted_trusted_assigned", "trusted_visible_junctions", "visible_junctions", "accepted_curve_support_6",
                "matched_distance_mean", "matched_distance_max", "radius_mae_floored_target", "geometry_gate_passed",
                "selection_gate_passed", "combined_learnability_gate_passed")}
            item.update(step=step, trusted_matches=int(trusted.sum()), confidence=confidence,
                balanced_reference_ce=float(balanced_ce), assignment_changes_since_previous_snapshot=churn, candidate_arms=arms)
            if step == 4000:
                lower = np.asarray(truth["meta"]["origin_xyz"])
                upper = lower+np.asarray(truth["meta"]["shape_zyx"])[::-1]-1
                segments = []
                for a, b in truth["geometry"]["full_segments"]:
                    interval = clip_parameters(a, b, lower, upper)
                    if interval is not None:
                        segments.append([a+interval[0]*(b-a), a+interval[1]*(b-a)])
                extra = xyz[accepted & ~positive]
                curve_distance = reference_distances(extra, np.asarray(segments).reshape(-1, 2, 3))
                nearest_target = cKDTree(truth["nodes"][:, 1:4]).query(extra, workers=1)[0]
                item["accepted_unassigned_geometry"] = {"count": len(extra), "within_2_of_reference_curve": int((curve_distance <= 2).sum()),
                    "within_6_of_reference_curve": int((curve_distance <= 6).sum()), "within_2_of_a_set_target": int((nearest_target <= 2).sum()),
                    "within_4_of_a_set_target": int((nearest_target <= 4).sum())}
                input_distances = cKDTree(truth["points"][:, :3]).query(truth["nodes"][ti[~trusted], 1:4], workers=1)[0]
                item["poorly_placed_targets"] = [{"target_index": int(t), "query_index": int(p), "distance": float(d),
                    "is_junction": bool(truth["junction_mask"][t]), "accepted": bool(accepted[p]),
                    "used_probability": float(expit(difference[p])), "nearest_input_distance": float(inp)}
                    for p, t, d, inp in zip(pi[~trusted], ti[~trusted], distance[~trusted], input_distances)]
            rows.append(item)
    for step in recipe["snapshots"]:
        path = directory/f"model/step_{step:04d}.pt"
        cp = torch.load(path, map_location="cpu", weights_only=True)
        assert cp["step"] == cp["inference_checkpoint"]["step"] == step
        assert cp["recipe"] == recipe and cp["provenance"] == fit["provenance"] and cp["backend"] == fit["backend"]
        assert cp["next_crop_index"] == step % 4 and cp["scheduler"] is None
        assert {int(s["step"]) for s in cp["optimizer"]["state"].values()} == ({step} if step else set())
        if balanced:
            assert cp["experiment_schema"] == SCHEMA
        if step == 0:
            assert {k: base.tensor_digest(v) for k, v in cp["inference_checkpoint"]["model"].items()} == fit["initial_model_sha256"]
        if step == 4000:
            assert not cp["resumable"] and base.file_digest(path) == fit["final_checkpoint_sha256"]
        checkpoints.append({"step": step, "sha256": base.file_digest(path), "resumable": cp["resumable"],
            "optimizer_parameter_states": len(cp["optimizer"]["state"]), "cuda_rng_states": len(cp["rng"]["torch_cuda"])})
    return {"fit_report_sha256": base.file_digest(directory/"model/report.json"), "elapsed_seconds": fit["elapsed_seconds"],
        "checkpoint_reload_max_delta_reported_by_fit": fit["checkpoint_reload_max_delta"],
        "source_cache_snapshot_checks_passed": True, "independent_metrics_checks_passed": True,
        "prediction_files_sha256": prediction_hashes, "checkpoints": checkpoints, "rows": rows}


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    metrics = [("Trusted targets", lambda r: r["trusted_matches"]/r["targets"]),
        ("Accepted assigned targets", lambda r: r["accepted_assigned"]/r["targets"]),
        ("Accepted unassigned queries", lambda r: r["accepted_unassigned"]),
        ("Assignment-label ROC AUC", lambda r: r["confidence"]["roc_auc"]),
        ("Accepted curve support within 6 voxels", lambda r: r["accepted_curve_support_6"]),
        ("Required-edge ceiling after node cutoff (R8)", lambda r: r["candidate_arms"][0]["edge_ceiling_after_cutoff"])]
    for j, crop in enumerate([(0, 0), (0, 1), (92, 4), (86, 3)]):
        for arm, style in (("baseline", "--"), ("balanced", "-")):
            rows = [r for r in result[arm]["rows"] if (r["sample_index"], r["patch_index"]) == crop]
            for ax, (title, metric) in zip(axes.ravel(), metrics):
                ax.plot([r["step"] for r in rows], [metric(r) for r in rows], style, color=f"C{j}", marker=".",
                    label=f"{crop[0]}/{crop[1]} {arm}")
                ax.set_title(title, fontsize=10); ax.set_xlabel("Optimizer updates"); ax.grid(alpha=.2)
    for i, ax in enumerate(axes.ravel()):
        if i != 2:
            ax.set_ylim(-.03, 1.03)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Matched four-crop TRAINING comparison | dashed baseline / solid balanced | no learned graph evaluation")
    fig.savefig(output/"paired_trajectory.png", dpi=150); plt.close(fig)


def run(args):
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Choose a fresh ignored tmp/pointneuron2 output directory")
    torch.set_num_threads(2)
    spec, baseline_dir, baseline, recipe, balanced_recipe = contract(SimpleNamespace(config=ROOT/"configs/pointneuron2/balanced_query_comparison.json"))
    balanced = base.read(args.experiment_dir/"model/report.json")
    assert balanced["schema"] == SCHEMA and balanced["recipe"] == balanced_recipe and balanced["comparison_spec"] == spec
    assert balanced["backend"] == baseline["backend"] and balanced["initial_model_sha256"] == baseline["initial_model_sha256"]
    assert balanced["provenance"]["cache_manifest_sha256"] == baseline["provenance"]["cache_manifest_sha256"]
    for i in range(4):
        with np.load(baseline_dir/f"model/step_0000_crop_{i}.npz", allow_pickle=False) as a, np.load(args.experiment_dir/f"model/step_0000_crop_{i}.npz", allow_pickle=False) as b:
            assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    result = {"schema": "pointneuron2-balanced-query-paired-review-v1", "status": "completed",
        "scope": "saved TRAINING predictions; CPU only; zero model forwards/optimizer updates; no cutoff search or dev/test evaluation",
        "review_script_sha256": base.file_digest(Path(__file__)), "paired_initialization_cache_backend_recipe_checks_passed": True,
        "baseline": review(baseline_dir, baseline, recipe, False), "balanced": review(args.experiment_dir, balanced, recipe, True)}
    args.output_dir.mkdir(parents=True)
    plot(result, args.output_dir)
    base.write(args.output_dir/"summary.json", result)
    print(json.dumps({"status": "completed", "final": {arm: result[arm]["rows"][-4:] for arm in ("baseline", "balanced")}}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, default=ROOT/"tmp/pointneuron2/local_runs/balanced_query_20261018")
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())

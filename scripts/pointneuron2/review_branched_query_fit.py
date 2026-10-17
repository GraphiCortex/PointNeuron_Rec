"""Read-only CPU review of cached four-crop predictions; never fits or forwards a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import rankdata
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from pointneuron.data.curve_targets import clip_parameters
from scripts.pointneuron2.branched_query_experiment import restore_scoring_geometry
from scripts.pointneuron2.field_sampling_metrics import measure_support, reference_distances


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_stats(difference, positive, weight):
    p, n = int(positive.sum()), int((~positive).sum())
    scores = expit(difference)
    ranks = rankdata(difference, method="average")
    auc = (ranks[positive].sum() - p * (p + 1) / 2) / (p * n)
    order = np.argsort(-difference, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(difference[order]) != 0), len(order)-1]
    tp = np.cumsum(positive[order])[ends]
    precision, recall = tp / (ends + 1), tp / p
    ap = np.sum(np.diff(np.r_[0., recall]) * precision)
    constant = weight*p / (weight*p + n)
    constant_loss = -constant*np.log(constant) - (1-constant)*np.log1p(-constant)
    actual = (weight*np.logaddexp(0, -difference[positive]).sum() + np.logaddexp(0, difference[~positive]).sum()) / (weight*p+n)
    return {"roc_auc": float(auc), "average_precision": float(ap), "positive_fraction": p/len(positive),
        "oracle_known_target_count_top_k_recall": float(positive[order[:p]].sum()/p),
        "matched_probability_quantiles": np.quantile(scores[positive], [0, .25, .5, .75, 1]).tolist(),
        "unassigned_probability_quantiles": np.quantile(scores[~positive], [0, .25, .5, .75, 1]).tolist(),
        "all_labels_separable_by_some_cutoff": bool(difference[positive].min() > difference[~positive].max()),
        "weighted_constant_probability_optimum": float(constant), "weighted_constant_minimum_ce": float(constant_loss),
        "actual_weighted_ce": float(actual)}


def plot_reports(report, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r['sample_index']}/{r['patch_index']}" for r in rows[-4:]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for i, label in enumerate(labels):
        series = [r for r in rows if f"{r['sample_index']}/{r['patch_index']}" == label]
        x = [r["step"] for r in series]
        axes[0].plot(x, [r["trusted_matches"]/r["targets"] for r in series], "o-", label=label)
        axes[1].plot(x, [r["accepted_assigned"]/r["targets"] for r in series], "o-", label=label)
        axes[2].plot(x, [r["accepted_unassigned"] for r in series], "o-", label=label)
    for ax, title in zip(axes, ["Trusted placements / all targets", "Accepted assigned / all targets", "Accepted unassigned queries"]):
        ax.set_title(title); ax.set_xlabel("Optimizer updates"); ax.grid(alpha=.2)
    for ax in axes[:2]:
        ax.axhline(.95, color="black", ls="--", lw=.8); ax.set_ylim(-.03, 1.03)
    axes[0].legend(title="Training neuron/crop", fontsize=8)
    fig.suptitle("Training learnability only | fixed clouds, four crops | no GATv2 or test evaluation")
    fig.savefig(output / "learning_trajectory.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, r in zip(axes.ravel(), rows[-4:]):
        for values, color, label in [(r.pop("positive_scores"), "teal", "assigned positives"),
                                      (r.pop("negative_scores"), "orange", "unassigned set negatives")]:
            ax.hist(values, bins=np.linspace(0, 1, 26), weights=np.ones(len(values))/len(values), histtype="step", lw=1.5, color=color, label=label)
        ax.axvline(.5, color="black", ls="--", label="fixed .5 cutoff")
        ax.set_title(f"Train {r['sample_index']}/{r['patch_index']} | AUC={r['confidence']['roc_auc']:.3f}")
        ax.set_xlabel("Used probability"); ax.set_ylabel("Fraction within each label class")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Final confidence | labels are the final Hungarian assignment, not biological foreground")
    fig.savefig(output / "confidence_distributions.png", dpi=140); plt.close(fig)


def run(args):
    if args.output_dir.exists():
        raise FileExistsError("Preserve prior review evidence; choose a fresh output directory")
    if not args.output_dir.resolve().is_relative_to((ROOT / "tmp/pointneuron2").resolve()):
        raise ValueError("Generated evidence belongs under tmp/pointneuron2")
    fit = load(args.experiment_dir / "model/report.json")
    manifest = load(args.experiment_dir / "cache/manifest.json")
    recipe = fit["recipe"]
    assert fit["status"] == "completed" and fit["optimizer_updates"] == recipe["updates"] == 4000
    assert not fit["combined_learnability_gate_passed"] and fit["checkpoint_reload_max_delta"] == 0
    protected = {**manifest["protected_files_sha256"], **fit["provenance"]["source_files_sha256"]}
    for path, expected in protected.items():
        assert sha(Path(path)) == expected, path
    for path, expected in fit["provenance"]["source_files_sha256"].items():
        assert sha(args.experiment_dir / "model/sources" / Path(path).relative_to(ROOT)) == expected
    cache = []
    for row in manifest["records"]:
        path = args.experiment_dir / "cache" / row["path"]
        assert row["partition"] == "train" and sha(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as z:
            cache.append({k: z[k].copy() for k in ("points", "nodes", "target_edges", "target_components", "junction_mask")}
                | {"geometry": restore_scoring_geometry(json.loads(str(z["geometry"]))), "meta": json.loads(str(z["metadata"]))})
    rows, predictions_hashes = [], {}
    last_mapping = [None]*4
    for evaluation in fit["evaluations"]:
        step = evaluation["step"]
        for i, (r, truth) in enumerate(zip(evaluation["crops"], cache)):
            path = args.experiment_dir / f"model/step_{step:04d}_crop_{i}.npz"
            predictions_hashes[path.name] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                xyz, logits, radii = (z[k][0].astype(float) for k in ("centers", "logits", "radii"))
                pi, ti, pairs, labels = (z[k] for k in ("pred_indices", "target_indices", "candidate_edges", "edge_labels"))
            assert [pi.tolist(), ti.tolist()] == r["assignments"]
            positive = np.zeros(512, dtype=bool); positive[pi] = True
            difference = logits[:, 1] - logits[:, 0]
            accepted = difference >= 0
            distance = np.linalg.norm(xyz[pi] - truth["nodes"][ti, 1:4], axis=1)
            trusted = distance <= recipe["trusted_match_tolerance"]
            assert int(trusted.sum()) == r["connectivity_supervision"]["trusted_matches"]
            assert int(accepted[pi].sum()) == r["accepted_assigned"]
            assert int(accepted[~positive].sum()) == r["accepted_unassigned"]
            assert abs(distance.mean() - r["matched_distance_mean"]) < 1e-8
            classification = ranking_stats(difference, positive, recipe["query_loss"]["positive_class_weight"])
            assert abs(classification["actual_weighted_ce"] - r["classification_loss"]) < 1e-6
            mapping = {int(t): int(p) for p, t in zip(pi, ti)}
            churn = None if last_mapping[i] is None else sum(mapping[t] != last_mapping[i][t] for t in mapping)
            last_mapping[i] = mapping
            # Independent candidate and label checks use direct NumPy distances.
            distance2 = ((xyz[:, None] - xyz[None, :])**2).sum(-1)
            candidate_set = {tuple(sorted(map(int, e))) for e in pairs}
            expected_pairs = set()
            for u in range(512):
                order = np.lexsort((np.arange(512), distance2[u]))
                for v in np.union1d(order[order != u][:16], np.flatnonzero(distance2[u] <= (8+1e-6)**2)):
                    if u != v:
                        expected_pairs.add(tuple(sorted((u, int(v)))))
            assert candidate_set == expected_pairs
            target_to_pred = {int(t): int(p) for p, t in zip(pi[trusted], ti[trusted])}
            pred_to_target = {p: t for t, p in target_to_pred.items()}
            required = {tuple(sorted(map(int, e))) for e in truth["target_edges"]}
            independent = []
            for u, v in pairs:
                a, b = pred_to_target.get(int(u)), pred_to_target.get(int(v))
                if a is None or b is None: label = -1
                elif tuple(sorted((a, b))) in required: label = 1
                elif truth["target_components"][a] >= 0 and truth["target_components"][a] == truth["target_components"][b]: label = 0
                else: label = -1
                independent.append(label)
            assert np.array_equal(labels, independent)
            missed = []
            for a, b in sorted(required):
                if a not in target_to_pred or b not in target_to_pred: continue
                u, v = target_to_pred[a], target_to_pred[b]
                if tuple(sorted((u, v))) in candidate_set: continue
                rank_u = int(np.flatnonzero(np.lexsort((np.arange(512), distance2[u])) == v)[0])
                rank_v = int(np.flatnonzero(np.lexsort((np.arange(512), distance2[v])) == u)[0])
                missed.append({"target_edge": [a, b], "query_edge": [u, v], "predicted_length": float(np.sqrt(distance2[u, v])),
                    "target_chord_length": float(np.linalg.norm(truth["nodes"][a, 1:4]-truth["nodes"][b, 1:4])),
                    "endpoint_neighbor_ranks_excluding_self": [rank_u, rank_v]})
            assert len(missed) == r["connectivity_supervision"]["missing_candidate_positive_edges"]
            item = {"step": step, "sample_index": r["sample_index"], "patch_index": r["patch_index"], "targets": len(ti),
                "trusted_matches": int(trusted.sum()), "accepted_assigned": int(accepted[pi].sum()),
                "accepted_unassigned": int(accepted[~positive].sum()), "matched_distance_mean": float(distance.mean()),
                "trusted_junctions": r["trusted_visible_junctions"], "junctions": r["visible_junctions"],
                "accepted_curve_support_6": r["accepted_curve_support_6"], "radius_mae": r["radius_mae_floored_target"],
                "confidence": classification, "assignment_changes_since_previous_snapshot": churn,
                "connectivity_supervision": r["connectivity_supervision"],
                "edge_ceiling_after_node_cutoff": r["attainable_edge_recall_after_node_cutoff"],
                "missing_candidate_edges_with_trusted_endpoints": missed}
            if step == 4000:
                # Exact distances to the annotated curve clipped to this image box.
                lower = np.asarray(truth["meta"]["origin_xyz"])
                upper = lower + np.asarray(truth["meta"]["shape_zyx"])[::-1] - 1
                segments = []
                for a, b in truth["geometry"]["full_segments"]:
                    interval = clip_parameters(a, b, lower, upper)
                    if interval is not None:
                        segments.append([a + interval[0]*(b-a), a + interval[1]*(b-a)])
                curve_distance = reference_distances(xyz[~positive], np.asarray(segments).reshape(-1, 2, 3))
                nearest_input = cKDTree(truth["points"][:, :3]).query(truth["nodes"][ti[~trusted], 1:4], workers=1)[0]
                target_r = np.maximum(truth["nodes"][:, 4], recipe["query_loss"]["radius_floor"])
                item.update(unassigned_queries_within_2_of_clipped_reference=int((curve_distance <= 2).sum()),
                    unassigned_queries_within_6_of_clipped_reference=int((curve_distance <= 6).sum()),
                    unassigned_queries=len(curve_distance), poorly_placed_targets_near_input_within_6=int((nearest_input <= 6).sum()),
                    poorly_placed_targets=len(nearest_input), floored_target_radius_range=[float(target_r.min()), float(target_r.max())],
                    positive_scores=expit(difference[positive]), negative_scores=expit(difference[~positive]))
            rows.append(item)
    checkpoints = []
    for step in recipe["snapshots"]:
        path = args.experiment_dir / f"model/step_{step:04d}.pt"
        cp = torch.load(path, map_location="cpu", weights_only=True)
        assert cp["step"] == cp["inference_checkpoint"]["step"] == step
        assert cp["recipe"] == recipe and cp["provenance"] == fit["provenance"]
        assert cp["next_crop_index"] == step % 4 and cp["scheduler"] is None
        optimizer_steps = {int(s["step"]) for s in cp["optimizer"]["state"].values()}
        assert optimizer_steps == ({step} if step else set())
        checkpoints.append({"step": step, "sha256": sha(path), "resumable": cp["resumable"],
            "optimizer_parameter_states": len(cp["optimizer"]["state"]), "cuda_rng_states": len(cp["rng"]["torch_cuda"])})
        if step == 4000:
            assert not cp["resumable"] and sha(path) == fit["final_checkpoint_sha256"]
        del cp
    args.output_dir.mkdir(parents=True)
    plot_reports(fit, rows, args.output_dir)
    result = {"status": "completed", "scope": "saved training predictions/checkpoints; no model forwards, optimizer updates, GPU work, dev/test evaluation or threshold search",
        "fit_report_sha256": sha(args.experiment_dir / "model/report.json"), "review_script_sha256": sha(Path(__file__)),
        "checkpoint_reload_max_delta_reported_by_fit": fit["checkpoint_reload_max_delta"],
        "source_cache_snapshot_checks_passed": True, "independent_candidate_label_distance_ce_checks_passed": True,
        "fit_elapsed_seconds": fit["elapsed_seconds"], "combined_learnability_gate_passed": False,
        "prediction_files_sha256": predictions_hashes, "checkpoints": checkpoints, "rows": rows,
        "caveats": ["Final assignment labels are set labels, not independently verified foreground/background.",
                    "Top-k uses reference target count as a ranking oracle, never an inference rule.",
                    "Constant-score optimum is an algebraic loss baseline, not a fitted alternative model.",
                    "Assignment changes across sparse snapshots do not prove harmful churn within individual updates."]}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "final": rows[-4:], "checkpoints": checkpoints}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())

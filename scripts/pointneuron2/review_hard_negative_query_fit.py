"""Read-only hard-negative versus balanced review; no model forward or training."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from scripts.pointneuron2 import branched_query_experiment as base
from scripts.pointneuron2 import balanced_query_experiment as balanced_runner
from scripts.pointneuron2 import hard_negative_query_experiment as hard_runner
from scripts.pointneuron2 import review_balanced_query_fit as common


def review(directory, fit, original_recipe, *, hard):
    # Read-only field adapter: the preserved reviewer expects the older weighted
    # reference CE name. Actual recipe/provenance and on-disk files are untouched.
    view = copy.deepcopy(fit)
    for evaluation in view["evaluations"]:
        for row in evaluation["crops"]:
            row["classification_loss"] = row["reference_weighted_classification_loss"]
    result = common.review(directory, view, original_recipe, balanced=False)
    expected_schema = hard_runner.SCHEMA if hard else balanced_runner.SCHEMA
    for evaluation in fit["evaluations"]:
        step = evaluation["step"]
        cp = torch.load(directory/f"model/step_{step:04d}.pt",map_location="cpu",weights_only=True)
        assert cp["experiment_schema"] == expected_schema
        del cp
        for i, reported in enumerate(evaluation["crops"]):
            row = next(r for r in result["rows"] if r["step"] == step
                and r["sample_index"] == reported["sample_index"] and r["patch_index"] == reported["patch_index"])
            with np.load(directory/f"model/step_{step:04d}_crop_{i}.npz",allow_pickle=False) as saved:
                logits = saved["logits"][0].astype(float)
                positive = np.zeros(512,dtype=bool); positive[saved["pred_indices"]] = True
            reference_balanced = reported["balanced_reference_classification_loss"] if hard else reported["balanced_classification_loss"]
            assert abs(row["balanced_reference_ce"]-reference_balanced) < 1e-6
            for checked, actual in zip(row["candidate_arms"],reported["candidate_margin_comparison"]):
                assert checked["radius"] == actual["radius"] and checked["candidate_edges"] == actual["candidate_edges"]
                assert all(checked[k] == actual["connectivity_supervision"][k] for k in ("positive","negative","unknown"))
                assert abs(checked["edge_ceiling_after_cutoff"]-actual["edge_ceiling_after_node_cutoff"]) < 1e-10
            if hard:
                margin = logits[:,1]-logits[:,0]
                negative = np.flatnonzero(~positive)
                k = min(int(positive.sum()),len(negative))
                chosen = negative[np.lexsort((negative,-margin[negative]))[:k]]
                ce = (.5*np.logaddexp(0,-margin[positive]).mean()
                    + .25*np.logaddexp(0,margin[negative]).mean()
                    + .25*np.logaddexp(0,margin[chosen]).mean())
                assert abs(ce-reported["hard_negative_classification_loss"]) < 1e-6
                assert k == reported["hard_negative_count"]
                assert set(chosen.tolist()) == set(reported["hard_negative_query_indices"])
                assert int((margin[chosen] >= 0).sum()) == reported["accepted_unmatched_in_hard_set"]
                if k < len(negative):
                    assert chosen.tolist() == reported["hard_negative_query_indices"]
                row.update(hard_negative_classification_loss=float(ce),hard_negative_count=k,
                    accepted_unmatched_in_hard_set=int((margin[chosen] >= 0).sum()))
    result["experiment_schema_checks_passed"] = True
    result["reference_and_optimization_loss_checks_passed"] = True
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(14,7),layout="constrained")
    metrics = [("Trusted targets",lambda r:r["trusted_matches"]/r["targets"]),
        ("Accepted assigned targets",lambda r:r["accepted_assigned"]/r["targets"]),
        ("Accepted unmatched queries",lambda r:r["accepted_unassigned"]),
        ("Assignment-label ROC AUC",lambda r:r["confidence"]["roc_auc"]),
        ("Accepted curve support within 6 voxels",lambda r:r["accepted_curve_support_6"]),
        ("Required-edge ceiling after node cutoff (R8)",lambda r:r["candidate_arms"][0]["edge_ceiling_after_cutoff"])]
    for j,crop in enumerate([(0,0),(0,1),(92,4),(86,3)]):
        for arm,style in (("balanced","--"),("hard_negative","-")):
            rows = [r for r in result[arm]["rows"] if (r["sample_index"],r["patch_index"]) == crop]
            for ax,(title,metric) in zip(axes.ravel(),metrics):
                ax.plot([r["step"] for r in rows],[metric(r) for r in rows],style,color=f"C{j}",marker=".",label=f"{crop[0]}/{crop[1]} {arm}")
                ax.set_title(title,fontsize=10); ax.set_xlabel("Optimizer updates"); ax.grid(alpha=.2)
    for i,ax in enumerate(axes.ravel()):
        if i != 2:
            ax.set_ylim(-.03,1.03)
    axes[0,0].legend(fontsize=6,ncol=2)
    fig.suptitle("Fixed TRAINING comparison | dashed balanced / solid hard-negative | no learned graph evaluation")
    fig.savefig(output/"paired_trajectory.png",dpi=150); plt.close(fig)


def run(args):
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Choose a fresh ignored tmp/pointneuron2 output directory")
    torch.set_num_threads(2)
    spec,baseline_dir,baseline,original_recipe,recipe = hard_runner.contract(SimpleNamespace(config=ROOT/"configs/pointneuron2/hard_negative_query_comparison.json"))
    fit = base.read(args.experiment_dir/"model/report.json")
    assert fit["schema"] == hard_runner.SCHEMA and fit["recipe"] == recipe and fit["comparison_spec"] == spec
    assert fit["backend"] == baseline["backend"] and fit["initial_model_sha256"] == baseline["initial_model_sha256"]
    assert fit["provenance"]["cache_manifest_sha256"] == baseline["provenance"]["cache_manifest_sha256"]
    assert [e["step"] for e in fit["evaluations"]] == recipe["snapshots"]
    for i in range(4):
        with np.load(baseline_dir/f"model/step_0000_crop_{i}.npz",allow_pickle=False) as a, np.load(args.experiment_dir/f"model/step_0000_crop_{i}.npz",allow_pickle=False) as b:
            assert a.files == b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    result = {"schema":"pointneuron2-hard-negative-query-paired-review-v1","status":"completed",
        "scope":"saved TRAINING predictions/checkpoints; CPU only; zero model forwards, updates, GPU calls, cutoff search or dev/test evaluation",
        "review_script_sha256":base.file_digest(Path(__file__)),
        "review_dependency_sha256":base.file_digest(Path(common.__file__)),
        "paired_initialization_cache_backend_recipe_checks_passed":True,
        "balanced":review(baseline_dir,baseline,original_recipe,hard=False),
        "hard_negative":review(args.experiment_dir,fit,original_recipe,hard=True)}
    old,new = result["balanced"]["rows"][-4:],result["hard_negative"]["rows"][-4:]
    result["decision"] = {"all_selection_gates_passed":all(r["selection_gate_passed"] for r in new),
        "all_combined_gates_passed":all(r["combined_learnability_gate_passed"] for r in new),
        "trusted_target_nonregression":all(b["trusted_matches"] >= a["trusted_matches"] for a,b in zip(old,new)),
        "junction_nonregression":all(b["trusted_visible_junctions"] >= a["trusted_visible_junctions"] for a,b in zip(old,new))}
    result["selection_hypothesis_supported"] = all(result["decision"][k] for k in
        ("all_selection_gates_passed","trusted_target_nonregression","junction_nonregression"))
    clipping = fit["gradient_clipping_this_segment"]
    assert clipping["updates_observed"] == 4000 and clipping["above_clip"] <= 4000
    result["gradient_clipping_reported_by_fit"] = clipping
    args.output_dir.mkdir(parents=True)
    plot(result,args.output_dir); base.write(args.output_dir/"summary.json",result)
    print(result["decision"],flush=True)
    print("COMPLETED read-only hard-negative paired review",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir",type=Path,default=ROOT/"tmp/pointneuron2/local_runs/hard_negative_query_20261018")
    parser.add_argument("--output-dir",type=Path,required=True)
    run(parser.parse_args())

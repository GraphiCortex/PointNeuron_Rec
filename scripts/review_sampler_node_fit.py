"""Read-only audit of saved node fits; CPU checkpoint inspection, no forwards.

Recompute node counts independently from prediction files, separate geometry
from confidence/normalization, and draw saved terminal proposals. No optimizer,
new labels, inference policy, threshold selection or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import sampler_node_common as common
from pn2.evaluation.swc import parse_swc
from pn2.supervision.curve_nodes import rectangular_curve_nodes
from pn2.evaluation.geometry import _CurveIndex


def matched_count(target, predicted, tolerance):
    """Independent dense assignment with enough dummy slots to allow all misses."""
    if not len(target) or not len(predicted):
        return 0
    distance = np.linalg.norm(target[:, None]-predicted[None], axis=-1)
    # A real admissible match costs zero; a dummy miss costs one. Inadmissible
    # real matches cost two and cannot beat the always-available dummy columns.
    cost = np.concatenate((np.where(distance <= tolerance, 0., 2.), np.ones((len(target), len(target)))), axis=1)
    rows, cols = linear_sum_assignment(cost)
    return int(np.count_nonzero((cols < len(predicted)) & (cost[rows, cols] == 0)))


def summarize(packet, box_index, mode, record, saved):
    prefix = f"box{box_index}_{mode}_"
    centers, scores, logits = (packet[prefix+k] for k in ("centers", "scores", "logits"))
    if centers.shape != (512, 3) or scores.shape != (512,) or logits.shape != (512, 2):
        raise ValueError("Unexpected prediction shapes")
    logp = logits.astype(float)-np.logaddexp.reduce(logits.astype(float), axis=1)[:, None]
    if not np.allclose(np.exp(logp[:, 1]), scores, atol=2e-7, rtol=0):
        raise ValueError("Saved scores disagree with saved logits")
    target = record["target_xyz"]
    accepted = centers[scores >= .5]
    count = matched_count(target, accepted, 2.)
    all_count = matched_count(target, centers, 2.)
    expected = saved["tolerances"]["2.0"]
    if (count != expected["accepted_nodes"]["matched"]
            or len(accepted)-count != expected["accepted_nodes"]["extras"]
            or all_count != round(expected["all_query_node_recall"]*len(target))):
        raise ValueError("Independent one-to-one counts disagree with saved evaluator")
    pi = packet[prefix+"assignment_predictions"].astype(int)
    ti = packet[prefix+"assignment_targets"].astype(int)
    if len(pi) != len(target) or len(set(pi)) != len(pi) or sorted(ti.tolist()) != list(range(len(target))):
        raise ValueError("Saved assignments do not cover the target set one-to-one")
    distance = np.linalg.norm(centers[pi]-target[ti], axis=1)
    positive = np.zeros(512, dtype=bool); positive[pi] = True
    near = cKDTree(target).query(accepted)[0] <= 2. if len(accepted) else np.zeros(0, dtype=bool)
    curve_distance = record["curve_index"].distances(accepted) if len(accepted) else np.zeros(0)
    return {"mode": mode, "target_count": len(target), "accepted": len(accepted),
            "matched_2": count, "extras_2": len(accepted)-count,
            "precision_2": count/len(accepted) if len(accepted) else 0., "recall_2": count/len(target),
            "all_query_matched_2": all_count, "all_query_recall_2": all_count/len(target),
            "accepted_farther_than_2_from_every_target_node": int((~near).sum()),
            "accepted_near_a_target_but_unmatched": int(near.sum())-count,
            "accepted_near_continuous_reference_curve_2": int((curve_distance <= 2.).sum()),
            "accepted_near_continuous_reference_curve_6": int((curve_distance <= 6.).sum()),
            "accepted_median_distance_to_continuous_reference_curve": float(np.median(curve_distance)) if len(curve_distance) else None,
            "assignment_mean_distance_voxel": float(distance.mean()),
            "assignment_median_distance_voxel": float(np.median(distance)),
            "assignment_p95_distance_voxel": float(np.quantile(distance, .95)),
            "assignment_center_loss_beta4": float(np.where(distance < 4., distance**2/8., distance-2.).mean()),
            "assignment_balanced_classification": float((-logp[positive, 1].mean()-logp[~positive, 0].mean())/2),
            "assigned_positive_score_range": [float(scores[positive].min()), float(scores[positive].max())],
            "strongest_assigned_unused_score": float(scores[~positive].max()),
            "fixed_gate_passed": saved["learnability_gate"]}


def add_curve_indices(groups, recipe):
    inventory = common.checked_json(recipe["training_inventory"], recipe["training_inventory_sha256"])
    indices = {}
    for records in groups.values():
        for record in records:
            meta = record["metadata"]
            key = (meta["case_id"], meta["tile_id"])
            if key not in indices:
                source = next(c for c in inventory["cases"] if c["case_id"] == meta["case_id"])
                path = (common.ROOT / recipe["training_inventory"]).parent / source["reference_swc"]
                morphology = parse_swc(path, type_policy="opaque_integer")
                target = rectangular_curve_nodes(morphology, meta["box"]["start_xyz"], meta["box"]["stop_xyz"])
                segments = np.concatenate([np.stack((run[:-1], run[1:]), axis=1) for run in target.reference_runs])
                segments = segments[np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1) > 0]
                indices[key] = _CurveIndex(segments, 4.)
            record["curve_index"] = indices[key]


def inspect_checkpoints(folder, arm, recipe):
    import torch
    torch.set_num_threads(1)
    paths = []
    for step in recipe["evaluation_steps"]:
        path = folder / f"checkpoint_step{step:06d}.pt"
        marker = json.loads(path.with_suffix(".complete.json").read_text())
        if marker["sha256"] != common.digest(path) or marker["step"] != step:
            raise ValueError("Checkpoint completion hash/step mismatch")
        paths.append({"step": step, "file": path.name, "sha256": marker["sha256"]})
    # Hash-checked state files written by this run only. Deserialize on CPU.
    state = torch.load(folder / paths[-1]["file"], map_location="cpu", weights_only=False)
    if (state["step"] != 2000 or state["method"] != arm["method"]
            or state["recipe_fingerprint"] != common.fingerprint(recipe)
            or state["code_identity"] != common.code_identity()
            or common.state_digest(state["model"]) != arm["evaluations"][-1]["model_state_sha256"]):
        raise ValueError("Terminal state does not match the scored evaluation")
    counters = {k: int(v) for k, v in state["model"].items() if k.endswith("num_batches_tracked")}
    variances = [v for k, v in state["model"].items() if k.endswith("running_var")]
    clocks = sorted({int(v["step"]) for v in state["optimizer"]["state"].values()})
    if (len(counters) != 16 or set(counters.values()) != {2000} or clocks != [2000]
            or not all(bool(torch.isfinite(v).all()) for v in state["model"].values())
            or not all(bool((v > 0).all()) for v in variances)):
        raise ValueError("Incomplete/nonfinite optimizer or BatchNorm state")
    return {"complete_checkpoints": paths, "terminal_state_matches_evaluation": True,
            "batchnorm_layers": len(counters), "batchnorm_update_counts": sorted(set(counters.values())),
            "optimizer_step_values": clocks, "all_model_state_finite": True,
            "all_running_variances_positive": True,
            "minimum_running_variance": min(float(v.min()) for v in variances)}


def draw_terminal(groups, terminal_packets, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    limits = []
    first_method = next(iter(groups))
    for i, record in enumerate(groups[first_method]):
        cloud = [record["target_xyz"]]
        meta = record["metadata"]
        cloud += [np.asarray([meta["box"]["start_xyz"], meta["box"]["stop_xyz"]], dtype=float)-.5]
        for packet in terminal_packets.values():
            cloud.extend(packet[f"box{i}_{mode}_centers"] for mode in ("ordinary_inference", "current_crop_diagnostic"))
        points = np.concatenate(cloud)
        low, high = points.min(0), points.max(0)
        margin = np.maximum((high-low)*.05, 1.)
        limits.append((low-margin, high+margin))
    figures = []
    for method, packet in terminal_packets.items():
        fig = plt.figure(figsize=(18, 9))
        for row, mode in enumerate(("ordinary_inference", "current_crop_diagnostic")):
            for i, record in enumerate(groups[method]):
                ax = fig.add_subplot(2, 4, row*4+i+1, projection="3d")
                for points, _ in record["curve_samples"]:
                    ax.plot(*points.T, color="#147d42", linewidth=1.6)
                xyz = record["target_xyz"]
                ax.scatter(*xyz.T, color="#147d42", s=8, label="Reference targets / curve")
                centers, scores = packet[f"box{i}_{mode}_centers"], packet[f"box{i}_{mode}_scores"]
                ax.scatter(*centers.T, color="#4b78a8", s=3, alpha=.18, label="All 512 queries")
                accepted = centers[scores >= .5]
                if len(accepted):
                    ax.scatter(*accepted.T, color="#e57b19", s=10, alpha=.7, label="Accepted at 0.5")
                count = matched_count(xyz, accepted, 2.)
                available = matched_count(xyz, centers, 2.)
                low, high = limits[i]
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
                ax.set_box_aspect(high-low)
                ax.view_init(elev=24, azim=-60)
                ax.set_xlabel("X voxel", fontsize=8); ax.set_ylabel("Y voxel", fontsize=8); ax.set_zlabel("Z voxel", fontsize=8)
                ax.tick_params(labelsize=7)
                alias, tile = record["metadata"]["alias"], record["metadata"]["tile_id"]
                ax.set_title(f"{alias}/{tile}: matched {count}/{len(xyz)}, extras {len(accepted)-count}\nAll-query geometric matches: {available}/{len(xyz)}", fontsize=10)
        fig.suptitle(f"{method}: saved terminal predictions, update 2000\nTop: ordinary saved-BN inference | Bottom: current-crop statistics diagnostic (same weights)", fontsize=14)
        handles, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.text(.5, .045, "Shared per-box view limits across methods/modes. Reference geometry is for evaluation; no predicted edges are drawn.", ha="center", fontsize=9)
        fig.subplots_adjust(top=.84, bottom=.12, hspace=.25, wspace=.12)
        path = output / f"{method}_terminal.png"
        fig.savefig(path, dpi=150)
        plt.close(fig); figures.append(path.name)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder, output = Path(args.run).resolve(), Path(args.output).resolve()
    root_report = json.loads((folder / "report.json").read_text())
    recipe = common.load_recipe(folder / "recipe.json")
    if (root_report["status"] != "completed" or root_report["stage"] != "train"
            or root_report["recipe_fingerprint"] != common.fingerprint(recipe)
            or root_report["code_identity"] != common.code_identity()):
        raise ValueError("Expected completed current-source paired training control")
    output.mkdir(parents=True, exist_ok=False)
    groups = common.load_tasks(recipe)
    add_curve_indices(groups, recipe)
    summary = {"scope": "Saved-fit review only: no model forward, optimization or policy change",
               "run": str(folder.relative_to(common.ROOT)), "run_report_sha256": common.digest(folder / "report.json"),
               "review_script_sha256": common.digest(__file__), "recipe_fingerprint": common.fingerprint(recipe), "arms": []}
    summary["review_geometry_source_sha256"] = common.digest(common.ROOT / "src/pn2/evaluation/geometry.py")
    terminal_packets, initial_states = {}, []
    started = time.perf_counter()
    for arm in root_report["records"]:
        method = arm["method"]; directory = folder / method
        journal = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
        if [r["step"] for r in journal] != list(range(1, 2001)) or not all(np.isfinite([r[k] for k in ("loss", "classification", "center", "gradient_norm_before_clip", "seconds")]).all() for r in journal):
            raise ValueError("Training journal incomplete/nonfinite")
        print(f"REVIEW {method}: 2000 finite updates; recomputing saved node metrics", flush=True)
        history = []
        for evaluation in arm["evaluations"]:
            path = directory / evaluation["prediction_packet"]
            if common.digest(path) != evaluation["prediction_packet_sha256"]:
                raise ValueError("Saved prediction hash mismatch")
            with np.load(path, allow_pickle=False) as stored:
                packet = {k: stored[k].copy() for k in stored.files}
            if not all(np.isfinite(v).all() for v in packet.values()):
                raise ValueError("Nonfinite saved prediction")
            boxes = []
            for i, record in enumerate(groups[method]):
                saved = evaluation["records"][i]
                row = {"alias": saved["alias"], "tile_id": saved["tile_id"], "modes": {}}
                for mode in ("ordinary_inference", "current_crop_diagnostic"):
                    row["modes"][mode] = summarize(packet, i, mode, record, saved["modes"][mode])
                delta = np.linalg.norm(packet[f"box{i}_ordinary_inference_centers"]-packet[f"box{i}_current_crop_diagnostic_centers"], axis=1)
                row["same_query_mode_displacement_voxel"] = {"median": float(np.median(delta)), "mean": float(delta.mean()), "maximum": float(delta.max())}
                boxes.append(row)
            history.append({"step": evaluation["step"], "boxes": boxes})
            if evaluation["step"] == 2000:
                terminal_packets[method] = packet
        initial_states.append(arm["evaluations"][0]["model_state_sha256"])
        state = inspect_checkpoints(directory, arm, recipe)
        result = {"method": method, "status": arm["status"], "updates": arm["updates"], "gate": arm["learnability_gate"],
                  "elapsed_minutes_after_initial_evaluation": arm["elapsed_after_initial_evaluation_seconds"]/60,
                  "peak_allocated_bytes": arm["peak_allocated_bytes"], "prefix_identity": arm["prefix_identity"],
                  "checkpoint_verification": state, "history": history}
        summary["arms"].append(result)
        common.write(output / f"{method}_review.json", result)
        for box in history[-1]["boxes"]:
            a, b = (box["modes"][mode] for mode in ("ordinary_inference", "current_crop_diagnostic"))
            print(f"TERMINAL {box['alias']}/{box['tile_id']} ordinary={a['matched_2']}/{a['target_count']}+{a['extras_2']} diagnostic={b['matched_2']}/{b['target_count']}+{b['extras_2']} geometry_all={a['all_query_matched_2']}->{b['all_query_matched_2']}", flush=True)
    if len(set(initial_states)) != 1:
        raise ValueError("Arms started from different models")
    summary["same_initial_model"] = True
    summary["figures"] = draw_terminal(groups, terminal_packets, output)
    summary["seconds"] = time.perf_counter()-started
    common.write(output / "report.json", summary)
    print(f"COMPLETE read-only CPU review: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

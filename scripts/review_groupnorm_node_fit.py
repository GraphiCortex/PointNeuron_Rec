"""Audit saved GN fit and compare with saved BN; no model forward or fitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import sampler_node_common as common
import groupnorm_node_experiment as gn
from review_sampler_node_fit import summarize, add_curve_indices, matched_count


def verify_checkpoints(folder, recipe, report):
    import torch
    torch.set_num_threads(1)
    markers = []
    for step in recipe["evaluation_steps"]:
        path = folder / f"checkpoint_step{step:06d}.pt"
        marker = json.loads(path.with_suffix(".complete.json").read_text())
        if (marker["sha256"] != common.digest(path) or marker["step"] != step
                or marker["recipe_fingerprint"] != common.fingerprint(recipe)):
            raise ValueError("GN checkpoint completion marker mismatch")
        markers.append(marker)
    state = torch.load(folder / markers[-1]["checkpoint"], map_location="cpu", weights_only=False)
    if (state["step"] != 2000 or state["method"] != "pn2_sato"
            or state["recipe_fingerprint"] != common.fingerprint(recipe)
            or state["code_identity"] != common.code_identity()
            or state["recipe"]["normalization_control"] != recipe["normalization_control"]
            or common.state_digest(state["model"]) != report["evaluations"][-1]["model_state_sha256"]):
        raise ValueError("GN terminal checkpoint differs from scored source/recipe/state")
    if any(k.endswith(("running_mean", "running_var", "num_batches_tracked")) for k in state["model"]):
        raise ValueError("Unexpected BatchNorm running state in GN checkpoint")
    if not all(bool(torch.isfinite(v).all()) for v in state["model"].values()):
        raise ValueError("Nonfinite GN model state")
    clocks = sorted({int(v["step"]) for v in state["optimizer"]["state"].values()})
    if clocks != [2000]:
        raise ValueError("GN Adam did not reach the declared cap")
    return {"checkpoints": markers, "terminal_state_matches_evaluation": True,
            "optimizer_steps": clocks, "batchnorm_running_buffers": 0, "model_state_finite": True}


def score_conditioning(packet, records):
    scores = np.stack([packet[f"box{i}_ordinary_inference_scores"] for i in range(4)])
    labels = []
    for i in range(4):
        mask = np.zeros(512, dtype=bool)
        mask[packet[f"box{i}_ordinary_inference_assignment_predictions"].astype(int)] = True
        labels.append(mask)
    rows = []
    for a in range(4):
        for b in range(a+1, 4):
            rows.append({"box_a": [records[a]["metadata"]["alias"], records[a]["metadata"]["tile_id"]],
                         "box_b": [records[b]["metadata"]["alias"], records[b]["metadata"]["tile_id"]],
                         "pearson_query_score_correlation": float(np.corrcoef(scores[a], scores[b])[0, 1]),
                         "assigned_positive_query_intersection": int((labels[a] & labels[b]).sum()),
                         "assigned_positive_counts": [int(labels[a].sum()), int(labels[b].sum())]})
    return {"pairs": rows, "limitation": "Descriptive same-index score similarity, not proof of image neglect, assignment instability or a causal explanation"}


def query_location_prior(recipe, records, output):
    """Prospective coordinate prior only; no query head/model is implemented."""
    import probe_pointnext_capacity as probe
    torch = common.runtime(recipe, "cpu")
    config, source = probe.load(common.ROOT / recipe["encoder_config"])
    probe.imports(source, "compact_indices_v1")
    from pointneuron.models.pointnext_ops import farthest_point_indices
    rows = []
    for record in records:
        meta, packet = record["metadata"], record["packet"]
        indices = farthest_point_indices(torch.from_numpy(packet["local_xyz"])[None], 512)[0].numpy()
        if len(np.unique(indices)) != 512:
            raise ValueError("Prospective source indices are not distinct")
        xyz, target = packet["native_xyz"], record["target_xyz"]
        anchors = xyz[indices]
        distance = np.linalg.norm(target[:, None]-anchors[None], axis=-1)
        a, b = linear_sum_assignment(distance)
        path = output / f"{meta['alias']}_tile{meta['tile_id']}_prospective_locations.npz"
        np.savez_compressed(path, input_indices=indices, native_xyz=anchors)
        rows.append({"alias": meta["alias"], "tile_id": meta["tile_id"], "target_count": len(target),
            "input_points": len(xyz), "locations": len(anchors), "input_packet_sha256": meta["packet_sha256"],
            "full_input_matches_2": matched_count(target, xyz, 2.), "location_matches_2": matched_count(target, anchors, 2.),
            "full_input_matches_6": matched_count(target, xyz, 6.), "location_matches_6": matched_count(target, anchors, 6.),
            "mean_minimum_assignment_distance_voxel": float(distance[a, b].mean()),
            "maximum_minimum_assignment_distance_voxel": float(distance[a, b].max()),
            "packet": path.name, "packet_sha256": common.digest(path)})
    return {"scope": "Coordinate-only prospective FPS512 starting locations; no model/offset head or training",
            "selection_reference_access": False, "selection": "Pinned native FPS on fixed hash-ordered local XYZ only",
            "limitations": "Locations alone are not skeleton proposals or a reconstruction; all-point encoder/attention memory would remain available; offsets not implemented",
            "source_sha256": config["source_sha256"], "records": rows}


def plot_comparison(records, baseline, current, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    packets = [("Saved Sato/BatchNorm: ordinary inference", baseline),
               ("Fresh Sato/GroupNorm: ordinary inference", current)]
    fig = plt.figure(figsize=(18, 9))
    for row, (label, packet) in enumerate(packets):
        for i, record in enumerate(records):
            points = [record["target_xyz"], baseline[f"box{i}_ordinary_inference_centers"], current[f"box{i}_ordinary_inference_centers"]]
            all_points = np.concatenate(points)
            low, high = all_points.min(0), all_points.max(0)
            margin = np.maximum((high-low)*.05, 1.)
            low, high = low-margin, high+margin
            ax = fig.add_subplot(2, 4, row*4+i+1, projection="3d")
            for curve, _ in record["curve_samples"]:
                ax.plot(*curve.T, color="#147d42", linewidth=1.6)
            ax.scatter(*record["target_xyz"].T, color="#147d42", s=8, label="Reference targets / curve")
            centers, scores = packet[f"box{i}_ordinary_inference_centers"], packet[f"box{i}_ordinary_inference_scores"]
            ax.scatter(*centers.T, color="#4b78a8", s=3, alpha=.18, label="All 512 queries")
            accepted = centers[scores >= .5]
            ax.scatter(*accepted.T, color="#e57b19", s=10, alpha=.7, label="Accepted at 0.5")
            count = matched_count(record["target_xyz"], accepted, 2.)
            raw = matched_count(record["target_xyz"], centers, 2.)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
            ax.set_box_aspect(high-low); ax.view_init(elev=24, azim=-60)
            for axis in ("x", "y", "z"):
                getattr(ax, f"set_{axis}label")(f"{axis.upper()} voxel", fontsize=8)
            ax.tick_params(labelsize=7)
            meta = record["metadata"]
            ax.set_title(f"{meta['alias']}/{meta['tile_id']}: matched {count}/{len(record['target_xyz'])}, extras {len(accepted)-count}\nAll-query geometric matches: {raw}/{len(record['target_xyz'])}", fontsize=10)
    fig.suptitle("Same frozen Sato inputs, targets, optimizer and cap; normalization differs\nTop: BatchNorm ordinary inference | Bottom: GroupNorm ordinary inference | Terminal update 2000", fontsize=14)
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.text(.5, .045, "Shared per-box view limits. Reference geometry is for evaluation; no predicted edges are drawn.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.12, hspace=.25, wspace=.12)
    path = output / "groupnorm_vs_batchnorm_terminal.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder, output = Path(args.run).resolve(), Path(args.output).resolve()
    recipe = gn.load_recipe(gn.CONFIG)
    report = json.loads((folder / "report.json").read_text())
    if (report["status"] != "completed" or report["stage"] != "train" or report["updates"] != 2000
            or any(report.get(k) != v for k, v in gn.identity(recipe).items())):
        raise ValueError("Expected completed current-recipe GN control")
    output.mkdir(parents=True, exist_ok=False)
    groups = common.load_tasks(recipe); add_curve_indices(groups, recipe)
    records = groups["pn2_sato"]
    journal = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
    if ([r["step"] for r in journal] != list(range(1, 2001))
            or not all(np.isfinite([r[k] for k in ("loss", "classification", "center", "gradient_norm_before_clip", "seconds")]).all() for r in journal)):
        raise ValueError("Incomplete/nonfinite GN journal")
    print("GN REVIEW: 2000 finite updates; verifying all saved mode comparisons and node counts", flush=True)
    started, history, terminal = time.perf_counter(), [], None
    for evaluation in report["evaluations"]:
        path = folder / evaluation["prediction_packet"]
        if common.digest(path) != evaluation["prediction_packet_sha256"]:
            raise ValueError("GN prediction packet changed")
        with np.load(path, allow_pickle=False) as saved:
            packet = {k: saved[k].copy() for k in saved.files}
        if not all(np.isfinite(v).all() for v in packet.values()):
            raise ValueError("Nonfinite saved GN prediction")
        boxes = []
        for i, record in enumerate(records):
            source = evaluation["records"][i]
            row = {"alias": source["alias"], "tile_id": source["tile_id"], "modes": {}}
            for mode in ("ordinary_inference", "current_crop_diagnostic"):
                row["modes"][mode] = summarize(packet, i, mode, record, source["modes"][mode])
            for field in ("centers", "logits", "scores"):
                if not np.array_equal(packet[f"box{i}_ordinary_inference_{field}"], packet[f"box{i}_current_crop_diagnostic_{field}"]):
                    raise ValueError("Saved GN mode mismatch")
            boxes.append(row)
        history.append({"step": evaluation["step"], "boxes": boxes})
        if evaluation["step"] == 2000:
            terminal = packet
    state = verify_checkpoints(folder, recipe, report)
    baseline_path = common.ROOT / recipe["normalization_control"]["baseline_report"]
    baseline = json.loads(baseline_path.read_text())
    arm = next(r for r in baseline["records"] if r["method"] == "pn2_sato")
    ev = arm["evaluations"][-1]
    p = baseline_path.parent / "pn2_sato" / ev["prediction_packet"]
    if common.digest(p) != ev["prediction_packet_sha256"]:
        raise ValueError("Baseline prediction packet changed")
    with np.load(p, allow_pickle=False) as data:
        previous = {k: data[k].copy() for k in data.files}
    figure = plot_comparison(records, previous, terminal, output)
    summary = {"scope": "Saved GN/BN comparison only; no model forward, fitting or policy change",
        "run": str(folder.relative_to(common.ROOT)), "run_report_sha256": common.digest(folder / "report.json"),
        "review_script_sha256": common.digest(__file__), "normalization_consistency_passed": True,
        "learnability_gate": report["learnability_gate"], "checkpoint_verification": state,
        "history": history, "baseline_comparison": report["baseline_comparison"],
        "score_conditioning": score_conditioning(terminal, records),
        "elapsed_minutes_after_initial_evaluation": report["elapsed_after_initial_evaluation_seconds"]/60,
        "figure": figure, "seconds": time.perf_counter()-started}
    summary["query_location_feasibility"] = query_location_prior(recipe, records, output)
    common.write(output / "report.json", summary)
    for b in history[-1]["boxes"]:
        m = b["modes"]["ordinary_inference"]
        print(f"TERMINAL GN {b['alias']}/{b['tile_id']}: matched={m['matched_2']}/{m['target_count']} extras={m['extras_2']} all-query={m['all_query_matched_2']}/{m['target_count']} assigned_mean_error={m['assignment_mean_distance_voxel']:.3f}", flush=True)
    print(f"COMPLETE read-only GN review: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

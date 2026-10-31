"""Audit saved geometric-query fit versus GN; no model forward or optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np

import sampler_node_common as common
import geometric_query_experiment as experiment
import geometric_query_train_controller as launcher
from review_sampler_node_fit import summarize, add_curve_indices, matched_count
from review_groupnorm_node_fit import verify_checkpoints


def read_packet(folder, evaluation):
    path = folder/evaluation["prediction_packet"]
    if common.digest(path) != evaluation["prediction_packet_sha256"]:
        raise ValueError("Saved prediction packet changed")
    with np.load(path, allow_pickle=False) as data:
        packet = {k: data[k].copy() for k in data.files}
    if not all(np.isfinite(v).all() for v in packet.values()):
        raise ValueError("Nonfinite saved predictions")
    return packet


def plot_comparison(records, previous, current, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 9))
    for row, packet in enumerate((previous, current)):
        for i, record in enumerate(records):
            accepted_sets = [p[f"box{i}_ordinary_inference_centers"][p[f"box{i}_ordinary_inference_scores"] >= .5]
                             for p in (previous, current)]
            xyz = np.concatenate([record["target_xyz"], *accepted_sets])
            low, high = xyz.min(0), xyz.max(0)
            margin = np.maximum((high-low)*.05, 1.)
            low -= margin; high += margin
            ax = fig.add_subplot(2, 4, row*4+i+1, projection="3d")
            for curve, _ in record["curve_samples"]:
                ax.plot(*curve.T, color="#147d42", linewidth=1.6)
            ax.scatter(*record["target_xyz"].T, color="#147d42", s=10, label="Reference nodes / curve")
            accepted = accepted_sets[row]
            ax.scatter(*accepted.T, color="#e57b19", s=15, alpha=.8, label="Predictions accepted at 0.5")
            count = matched_count(record["target_xyz"], accepted, 2.)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
            ax.set_box_aspect(high-low); ax.view_init(elev=24, azim=-60)
            for axis in ("x", "y", "z"):
                getattr(ax, f"set_{axis}label")(f"{axis.upper()} voxel", fontsize=8)
            ax.tick_params(labelsize=7)
            meta = record["metadata"]
            ax.set_title(f"{meta['alias']}/{meta['tile_id']}: matched {count}/{len(record['target_xyz'])}, extras {len(accepted)-count}", fontsize=11)
    fig.suptitle("Matched four-box TRAINING comparison at update 2000\nTop: GN learned tokens | Bottom: GN geometric queries | Same clouds, initial parameters, loss and optimizer", fontsize=14)
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.text(.5, .045, "Shared per-box view limits. Green lines are the reference, not predicted connectivity. Rejected queries omitted.", ha="center", fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.12, hspace=.25, wspace=.12)
    path = folder/"geometric_vs_learned_tokens_terminal.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder, output = Path(args.run).resolve(), Path(args.output).resolve()
    recipe = experiment.load_recipe(experiment.CONFIG)
    report = json.loads((folder/"report.json").read_text())
    if (report["status"] != "completed" or report["stage"] != "train" or report["updates"] != 2000
            or any(report.get(k) != v for k, v in experiment.identity(recipe).items())):
        raise ValueError("Expected completed current-source geometric-query fit")
    launch = report["launch"]
    if (launch["controller_sha256"] != common.digest(launcher.__file__)
            or launch["powershell_launcher_sha256"] != common.digest(common.ROOT/"scripts/run_geometric_query_fit.ps1")
            or launch["update_cap"] != 2000 or launch["external_wall_timeout_seconds"] != 5400):
        raise ValueError("Launch metadata does not match the reviewed controller")
    timing_folder = common.ROOT/"tmp/geometric_query_timing_20261031_155707"
    timing = experiment.require_timing(recipe, timing_folder)
    if (common.digest(timing_folder/"report.json") != launch["timing_report_sha256"]
            or report["prefix_identity"] != timing["prefix_identity"]):
        raise ValueError("Fit does not replay the passed CUDA prefix")
    journal = [json.loads(line) for line in (folder/"updates.jsonl").read_text().splitlines()]
    if ([r["step"] for r in journal] != list(range(1,2001)) or not all(np.isfinite(
            [r[k] for k in ("loss","classification","center","gradient_norm_before_clip","seconds")]).all() for r in journal)):
        raise ValueError("Incomplete/nonfinite update journal")
    if [ev["step"] for ev in report["evaluations"]] != recipe["evaluation_steps"]:
        raise ValueError("Unexpected evaluation steps")
    output.mkdir(parents=True, exist_ok=False)
    groups = common.load_tasks(recipe); add_curve_indices(groups, recipe)
    records = groups["pn2_sato"]
    history = []
    for evaluation in report["evaluations"]:
        packet = read_packet(folder, evaluation)
        boxes = []
        for i, record in enumerate(records):
            saved = evaluation["records"][i]
            if (saved["alias"],saved["tile_id"]) != (record["metadata"]["alias"],record["metadata"]["tile_id"]):
                raise ValueError("Scored box order changed")
            modes = {mode: summarize(packet, i, mode, record, saved["modes"][mode])
                     for mode in ("ordinary_inference","current_crop_diagnostic")}
            for key in ("centers","scores","logits"):
                if not np.array_equal(packet[f"box{i}_ordinary_inference_{key}"],packet[f"box{i}_current_crop_diagnostic_{key}"]):
                    raise ValueError("Saved train/eval mismatch")
            boxes.append({"alias":saved["alias"],"tile_id":saved["tile_id"],"ordinary_inference":modes["ordinary_inference"]})
        history.append({"step":evaluation["step"],"boxes":boxes})
    terminal = packet
    state = verify_checkpoints(folder, recipe, report)
    control = recipe["geometric_query_control"]
    baseline_path = common.ROOT/control["baseline_report"]
    baseline = common.checked_json(control["baseline_report"],control["baseline_report_sha256"])
    previous = read_packet(baseline_path.parent,baseline["evaluations"][-1])
    baseline_boxes = [summarize(previous,i,"ordinary_inference",record,
                      baseline["evaluations"][-1]["records"][i]["modes"]["ordinary_inference"]) for i,record in enumerate(records)]
    passed = all(b["ordinary_inference"]["precision_2"] >= .95 and b["ordinary_inference"]["recall_2"] >= .95 for b in history[-1]["boxes"])
    if passed != report["learnability_gate"]:
        raise ValueError("Independent terminal verdict differs from report")
    summary = {"scope":"Saved TRAINING predictions and CPU checkpoint audit; no model forward, fitting, policy change or new-data access",
        "run":str(folder.relative_to(common.ROOT)),"run_report_sha256":common.digest(folder/"report.json"),
        "review_script_sha256":common.digest(__file__),"recipe_fingerprint":common.fingerprint(recipe),
        "normalization_consistency_passed":True,"learnability_gate":passed,"checkpoint_verification":state,
        "exact_cuda_prefix_replay":True,"finite_update_rows":2000,"node_count_comparisons":44,
        "exact_mode_array_pairs":60,"history":history,"baseline_terminal":baseline_boxes,
        "elapsed_minutes_after_initial_evaluation":report["elapsed_after_initial_evaluation_seconds"]/60,
        "figure":plot_comparison(records,previous,terminal,output)}
    common.write(output/"report.json",summary)
    for box in history[-1]["boxes"]:
        m=box["ordinary_inference"]
        print(f"VERIFIED {box['alias']}/{box['tile_id']}: {m['matched_2']}/{m['target_count']}+{m['extras_2']} mean_error={m['assignment_mean_distance_voxel']:.4f} positive_min={m['assigned_positive_score_range'][0]:.4f} negative_max={m['strongest_assigned_unused_score']:.4f}",flush=True)
    print(f"COMPLETE saved geometric-query review: {output/'report.json'}",flush=True)


if __name__ == "__main__":
    main()

"""Saved-weight CPU normalization control and query-score audit; zero updates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from scripts.pointneuron2 import pointnext_query_experiment as experiment
from pointneuron.models.pointnext_query_cropnorm import convert_checkpoint, load_cropnorm_checkpoint


def summarize(prediction, record, recipe):
    loss = experiment.query_set_loss(prediction, record["nodes"], record["mask"], **recipe["query_loss"])
    scores = prediction.objectness_logits.softmax(-1)[0, :, 1].double().numpy()
    xyz = prediction.center_proposals[0].double().numpy()
    targets = record["nodes"][0, :, 1:4].double().numpy()
    distances = np.linalg.norm(xyz[:, None] - targets[None], axis=-1)
    nearest = distances.argmin(axis=1)
    curve_distance = experiment.reference_distances(xyz, record["geometry"]["full_segments"])
    assignments = {int(q): int(t) for q, t in zip(*loss.assignments[0])}
    rows = []
    for q in range(len(xyz)):
        target = assignments.get(q)
        rows.append({"query": q, "score": float(scores[q]), "accepted": bool(scores[q] >= .5),
                     "matched_target": target,
                     "matched_distance": None if target is None else float(distances[q, target]),
                     "nearest_target": int(nearest[q]), "nearest_target_distance": float(distances[q, nearest[q]]),
                     "curve_distance": float(curve_distance[q])})
    matched = sorted((r for r in rows if r["matched_target"] is not None), key=lambda r: r["matched_target"])
    extras = [r for r in rows if r["matched_target"] is None and r["accepted"]]
    support = experiment.measure_support(xyz[scores >= .5], record["geometry"])
    match = experiment.inspect_matches(prediction, record["nodes"], record["mask"], loss)[0]
    curve6 = support["reference_to_points"]["within_voxels"]["6"]
    passed = (match["max_matched_distance"] <= 1 and match["matched_accepted_within_6"] == match["targets"]
              and match["unused_accepted"] == 0 and curve6 >= .95
              and bool(torch.isfinite(prediction.radius).all()) and bool((prediction.radius > 0).all()))
    summary = {"crop": record["metadata"]["patch_index"], "input_count": len(record["points"][0]),
               "loss": float(loss.total), "classification_loss": float(loss.objectness),
               "center_loss": float(loss.center), "radius_loss": float(loss.radius),
               "matching": match, "matched_queries": matched, "unmatched_accepted_queries": extras,
               "accepted_count": int((scores >= .5).sum()), "accepted_curve_coverage_6": curve6,
               "learnability_gate_passed": passed}
    return summary, rows


def maximum_delta(a, b):
    return max(float((getattr(a, key) - getattr(b, key)).abs().max())
               for key in ("center_proposals", "objectness_logits", "radius"))


def main(source_directory, output_directory):
    if output_directory.exists():
        raise FileExistsError("Preserve control outputs; select a fresh output directory")
    torch.set_num_threads(2)
    args = experiment.parser().parse_args([
        "--action", "preflight", "--device", "cpu", "--experiment-dir", str(source_directory),
        "--native-manifest", str(ROOT / "tmp/pointneuron2/local_runs/field_pilot_20261012/cache/manifest.json"),
        "--field-models", str(ROOT / "tmp/pointneuron2/local_runs/field_attention_20261013/baseline/model"),
        "--data-root", str(ROOT.parent / "data/gold166"),
    ])
    recipe, manifest, records, protected = experiment.load_cache(args)
    fit_path = source_directory / "model/fit_report.json"
    fit = experiment.read_json(fit_path)
    if (fit["status"] != "completed" or fit["optimizer_updates"] != 2000
            or fit["cache_manifest_sha256"] != experiment.file_digest(source_directory / "cache/manifest.json")):
        raise ValueError("Expected the completed declared fit/cache")
    protected.update(fit["source_files_sha256"])
    protected[str(fit_path.resolve())] = experiment.file_digest(fit_path)
    for step in (500, 1000, 2000):
        path = source_directory / f"model/step_{step:04d}.pt"
        protected[str(path.resolve())] = experiment.file_digest(path)
    if protected[str((source_directory / "model/step_2000.pt").resolve())] != fit["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoint identity differs")
    experiment.protected_unchanged(protected)
    output_directory.mkdir(parents=True)
    report_path = output_directory / "comparison.json"
    report = {"status": "running", "scope": "batch-one saved-weight normalization control; two training crops",
              "device": "CPU", "optimizer_updates": 0, "field_inference_passes": 0,
              "source_fit": str(fit_path.resolve()), "protected_files_sha256": protected,
              "rows": [], "train_eval_max_delta": 0., "reload_max_delta": 0., "crop_order_max_delta": 0.}
    report["control_source_sha256"] = {str(p.resolve()): experiment.file_digest(p) for p in
        (Path(__file__), ROOT / "src/pointneuron/models/pointnext_query_cropnorm.py")}
    previous = {}
    try:
        for step in (500, 1000, 2000):
            path = source_directory / f"model/step_{step:04d}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["step"] != step or checkpoint["recipe"] != recipe:
                raise ValueError("Saved checkpoint step/recipe differs")
            converted, model = convert_checkpoint(checkpoint, source_sha256=experiment.file_digest(path))
            model.requires_grad_(False)
            predictions = []
            for record in records:
                with torch.no_grad():
                    prediction = experiment.forward(model, record)
                summary, queries = summarize(prediction, record, recipe)
                summary["step"] = step
                mapping = {r["matched_target"]: r["query"] for r in summary["matched_queries"]}
                crop = summary["crop"]
                summary["matched_slot_changes_since_previous_snapshot"] = (None if crop not in previous else
                    sum(mapping[t] != previous[crop][t] for t in mapping))
                previous[crop] = mapping
                report["rows"].append(summary)
                for row in queries:
                    row.update(step=step, crop=crop)
                with (output_directory / f"step_{step:04d}_crop_{crop}_scores.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(queries[0])); writer.writeheader(); writer.writerows(queries)
                np.savez_compressed(output_directory / f"step_{step:04d}_crop_{crop}_predictions.npz",
                    centers=prediction.center_proposals.numpy(), logits=prediction.objectness_logits.numpy(),
                    radii=prediction.radius.numpy())
                predictions.append(prediction)
                print(f"step={step} crop={crop} mean_match={summary['matching']['mean_matched_distance']:.4f} "
                      f"accepted_matches={summary['matching']['matched_accepted_within_6']}/{summary['matching']['targets']} "
                      f"extras={summary['matching']['unused_accepted']} gate={summary['learnability_gate_passed']}", flush=True)
            if step == 2000:
                candidate_path = output_directory / "cropnorm_step_2000.pt"
                torch.save(converted, candidate_path)
                reloaded = load_cropnorm_checkpoint(torch.load(candidate_path, map_location="cpu", weights_only=True))
                for i in (1, 0):
                    record = records[i]
                    model.train()
                    with torch.no_grad():
                        trained_mode = experiment.forward(model, record)
                        loaded_mode = experiment.forward(reloaded, record)
                    model.eval()
                    with torch.no_grad():
                        repeated = experiment.forward(model, record)
                    report["train_eval_max_delta"] = max(report["train_eval_max_delta"], maximum_delta(predictions[i], trained_mode))
                    report["reload_max_delta"] = max(report["reload_max_delta"], maximum_delta(predictions[i], loaded_mode))
                    report["crop_order_max_delta"] = max(report["crop_order_max_delta"], maximum_delta(predictions[i], repeated))
                report["converted_checkpoint_sha256"] = experiment.file_digest(candidate_path)
                report["converted_model_spec"] = converted["model_spec"]
                report["discarded_running_buffers"] = converted["provenance"]["discarded_running_buffers"]
                report["unchanged_learned_parameters"] = True
        if report["train_eval_max_delta"] > 1e-4 or report["reload_max_delta"] > 1e-5 or report["crop_order_max_delta"] > 1e-5:
            raise ValueError("Normalization control consistency/reload checks failed")
        experiment.protected_unchanged(protected)
        final = [r for r in report["rows"] if r["step"] == 2000]
        report.update(status="completed", source_files_unchanged=True,
                      learnability_gate_passed=all(r["learnability_gate_passed"] for r in final),
                      next_step="Review per-query score/assignment findings; no fit or threshold change starts automatically.")
        experiment.write_json(report_path, report)
        print(f"Status: completed; optimizer updates=0; original source/checkpoints unchanged\nreport: {report_path}")
    except Exception as exc:
        report.update(status="failed", error=str(exc)); experiment.write_json(report_path, report); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=ROOT / "tmp/pointneuron2/local_runs/pointnext_query_20261013")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "tmp/pointneuron2/local_runs/pointnext_cropnorm_20261013")
    options = parser.parse_args()
    main(options.source_directory, options.output_directory)

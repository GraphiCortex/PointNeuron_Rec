"""Read-only diagnosis of a completed field fit: label excess, loss and thinning.

No optimizer, threshold search, learned settings, graph reconstruction or gate
change. Thinning is a secondary diagnostic, not promotion of failed predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from pointneuron.data.centerline_field import SCHEMA
from pointneuron.data.query_cache import file_digest
from pointneuron.models.centerline_unet import CenterlineUNet, field_loss_terms
from scripts.pointneuron2.train_centerline_fit import load_records, field_metrics
from scripts.pointneuron2.centerline_duration_comparison import completed_fit_config


def run(args):
    torch.set_num_threads(2)
    manifest, records = load_records(args.manifest, ROOT / "configs/pointneuron2/partition_seed0.json")
    fit = json.loads((args.models / "fit_report.json").read_text())
    cfg = completed_fit_config(manifest["config"], fit, ROOT / "configs/pointneuron2/centerline_duration.json")
    if fit["status"] != "completed" or fit["config"] != cfg or fit["manifest_sha256"] != file_digest(args.manifest):
        raise ValueError("Completed fit/cache contract mismatch")
    checkpoint_path = args.models / f"field_step_{cfg['steps']:04d}.pt"
    checkpoint_sha = file_digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    if (checkpoint["schema"] != SCHEMA or checkpoint["step"] != cfg["steps"] or
            checkpoint["config"] != cfg or checkpoint["manifest_sha256"] != file_digest(args.manifest)):
        raise ValueError("Final checkpoint contract mismatch")
    model = CenterlineUNet(cfg["base_channels"]).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    loss_spec = checkpoint.get("loss_spec", {"negative_aggregation": "all", "negative_to_positive_ratio": 3})
    if "loss_spec" in fit and fit["loss_spec"] != loss_spec:
        raise ValueError("Checkpoint and report loss definitions differ")
    loss_kwargs = {k: loss_spec[k] for k in ("negative_aggregation", "negative_to_positive_ratio")}
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    thinning_error, thinning_function = None, None
    try:
        from skimage.morphology import skeletonize
        thinning_function = skeletonize
    except (ImportError, ValueError) as exc:
        thinning_error = f"{type(exc).__name__}: {exc}"
        print(f"Optional thinning unavailable: {thinning_error}. Environment unchanged; continuing voxel/loss audit.", flush=True)
    report = {"scope": "read-only fixed training-crop diagnosis; no validation, optimizer or gate change",
              "fit_report_sha256": file_digest(args.models / "fit_report.json"), "checkpoint_sha256": checkpoint_sha,
              "manifest_sha256": file_digest(args.manifest), "device": args.device,
              "primary_gate_passed": fit["final_gate_passed"],
              "loss_spec": loss_spec,
              "threshold": cfg["threshold"], "thinning_method": "skimage skeletonize Lee, 3D, fixed threshold mask",
              "thinning_error": thinning_error,
              "trajectory": [], "crops": []}
    for evaluation in fit["evaluations"]:
        rows = evaluation["crops"]
        report["trajectory"].append({"step": evaluation["step"],
            **{k: float(np.mean([r[k] for r in rows])) for k in
               ("known_voxel_dice", "reference_coverage_6", "reference_proximity_6")}})
    for r, original in zip(records, fit["evaluations"][-1]["crops"]):
        with torch.no_grad():
            logits = model(torch.from_numpy(r["image"])[None, None].to(args.device))[0, 0]
            probability = logits.sigmoid().cpu().numpy()
            target = torch.from_numpy(r["target"]).to(args.device)
            known = torch.from_numpy(r["known"]).to(args.device)
            pos_term, neg_term, dice_term, selected_negatives = field_loss_terms(logits, target, known, **loss_kwargs)
            pos_bce, neg_bce, soft_dice_loss = .5*float(pos_term), .5*float(neg_term), float(dice_term)
        measured = field_metrics(probability, r)
        for key in ("reference_coverage_6", "reference_proximity_6", "known_voxel_dice"):
            if abs(measured[key] - original[key]) > 1e-5:
                raise ValueError("Final inference does not reproduce saved metrics")
        predicted = (probability >= .5) & r["valid"]
        pos, neg = r["known"] & r["target"], r["known"] & ~r["target"]
        near = neg & (r["distance"] <= 6)
        far = neg & ~near
        true_positive = int((predicted & pos).sum())
        false_positive = int((predicted & neg).sum())
        predicted_xyz = np.argwhere(predicted)[:, ::-1]
        reference = r["reference_local_xyz"]
        reference_distance = np.full(len(reference), np.inf) if not len(predicted_xyz) else cKDTree(predicted_xyz).query(reference)[0]
        reference_score = map_coordinates(probability, reference[:, ::-1].T, order=1, mode="nearest")
        thinned = None
        if thinning_function is not None:
            skeleton = thinning_function(predicted, method="lee").astype(bool)
            thinned = field_metrics(skeleton.astype(float), r)
        row = {"sample_index": r["row"]["sample_index"], "patch_index": r["row"]["patch_index"],
               "primary": measured,
               "target_voxels": int(pos.sum()), "known_negative_voxels": int(neg.sum()),
               "known_tube_recall": float(true_positive / pos.sum()),
               "known_tube_precision": float(true_positive / max(1, true_positive+false_positive)),
               "reference_sample_distance_to_accepted_voxel": {
                   "mean": float(reference_distance.mean()), "median": float(np.median(reference_distance)),
                   "p95": float(np.quantile(reference_distance, .95)),
                   "coverage_by_tolerance_voxels": {str(t): float((reference_distance <= t).mean()) for t in (1, 2, 4, 6)}},
               "interpolated_reference_center_score_mean": float(reference_score.mean()),
               "known_false_positive_voxels": false_positive,
               "known_false_positive_near_2_to_6": int((predicted & near).sum()),
               "known_false_positive_farther_than_6": int((predicted & far).sum()),
               "accepted_to_target_voxel_ratio": float(predicted.sum()/pos.sum()),
               "mean_known_negative_score": float(probability[neg].mean()),
               "known_positive_probability_mass": float(probability[pos].sum()),
               "known_negative_probability_mass": float(probability[neg].sum()),
               "selected_negative_bce_voxels": selected_negatives,
               "bce_positive_to_selected_negative_per_voxel_weight": float(selected_negatives/pos.sum()),
               "near_negative_fraction_of_all_known_negatives": float(near.sum()/neg.sum()),
               "loss": {"positive_bce_contribution": pos_bce, "negative_bce_contribution": neg_bce,
                        "soft_dice_loss": soft_dice_loss, "total": pos_bce+neg_bce+soft_dice_loss},
               "thinning_diagnostic": None if thinned is None else
                    {k: thinned[k] for k in ("reference_coverage_6", "reference_proximity_6", "accepted_voxels")}}
        report["crops"].append(row)
        if thinned is not None:
            np.savez_compressed(args.output_dir / (Path(r["row"]["path"]).stem + "_thinning.npz"),
                                skeleton=skeleton, origin_xyz=np.array(r["metadata"]["origin_xyz"]))
        print(f"neuron={row['sample_index']} crop={row['patch_index']}: tube_recall={row['known_tube_recall']:.3f} "
              f"tube_precision={row['known_tube_precision']:.3f} predicted/target={row['accepted_to_target_voxel_ratio']:.1f} "
              f"far_FP={row['known_false_positive_farther_than_6']} "
              f"thinning={'unavailable' if thinned is None else str(row['thinning_diagnostic'])}", flush=True)
    if file_digest(checkpoint_path) != checkpoint_sha:
        raise AssertionError("Checkpoint changed during read-only audit")
    report.update(status="completed", checkpoint_unchanged=True, inference_reproduction_checks="passed")
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2))
    print(f"report: {args.output_dir / 'audit.json'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())

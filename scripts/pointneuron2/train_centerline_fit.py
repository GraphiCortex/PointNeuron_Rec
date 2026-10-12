"""Bounded native-image field learnability check; user launches the full fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pointneuron.data.centerline_field import SCHEMA
from pointneuron.data.query_cache import file_digest
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.models.centerline_unet import CenterlineUNet, field_loss
from scripts.pointneuron2.prepare_centerline_fit import inspection
from scripts.pointneuron2.centerline_loss_comparison import check_baseline, compare_final, assert_baseline_unchanged, assert_initial_metrics
from scripts.pointneuron2.centerline_duration_comparison import duration_config, check_duration_baseline, compare_duration


def load_records(path, partition_path):
    manifest = json.loads(path.read_text())
    partition = json.loads(partition_path.read_text())
    if manifest.get("status") != "completed" or manifest.get("schema") != SCHEMA:
        raise ValueError("Incomplete or invalid field cache")
    if manifest.get("volume_reader_version") != VOLUME_READER_VERSION or manifest["partition_sha256"] != file_digest(partition_path):
        raise ValueError("Reader/partition mismatch")
    allowed = {(r["sample_index"], r["sample_id"]) for r in partition["partitions"]["train"]}
    reserved = {(r["sample_index"], r["sample_id"]) for role in ("val", "test") for r in partition["partitions"][role]}
    if {i for i, _ in allowed} & {i for i, _ in reserved} or {s for _, s in allowed} & {s for _, s in reserved}:
        raise ValueError("Crossed neuron partitions")
    records, seen = [], set()
    for row in manifest["records"]:
        if row["partition"] != "train" or (row["sample_index"], row["sample_id"]) not in allowed:
            raise ValueError("Learnability uses training neurons only")
        record_path = (path.parent / row["path"]).resolve()
        if not record_path.is_relative_to(path.parent.resolve()) or record_path in seen or file_digest(record_path) != row["sha256"]:
            raise ValueError("Invalid, duplicate or changed field record")
        seen.add(record_path)
        with np.load(record_path, allow_pickle=False) as z:
            r = {key: z[key].copy() for key in z.files if key != "metadata"}
            r["metadata"] = json.loads(str(z["metadata"]))
        if r["metadata"].get("field_schema") != SCHEMA:
            raise ValueError("Field schema mismatch")
        for key in ("sample_index", "sample_id", "partition", "patch_index"):
            if r["metadata"][key] != row[key]:
                raise ValueError("Record identity mismatch")
        shape = r["image"].shape
        if len(shape) != 3 or not np.isfinite(r["image"]).all() or r["image"].min() < 0 or r["image"].max() > 1:
            raise ValueError("Invalid normalized image")
        if any(r[k].shape != shape for k in ("target", "known", "valid", "distance", "unknown_bright")):
            raise ValueError("Misaligned voxel fields")
        if any(r[k].dtype != np.bool_ for k in ("target", "known", "valid", "unknown_bright")):
            raise ValueError("Expected boolean label masks")
        if np.any(r["known"] & ~r["valid"]) or np.any(r["known"] & r["unknown_bright"]):
            raise ValueError("Unknown/padding contributes direct supervision")
        if not np.any(r["target"] & r["known"]) or not np.any(~r["target"] & r["known"]):
            raise ValueError("Unsupervised fit crop")
        if np.isnan(r["distance"]).any() or (r["distance"] < 0).any() or not np.isfinite(r["reference_local_xyz"]).all():
            raise ValueError("Invalid geometry")
        r["row"] = row
        records.append(r)
    cfg = manifest["config"]
    actual = [(r["row"]["sample_index"], r["row"]["patch_index"]) for r in records]
    if actual != [tuple(x) for x in cfg["crops"]] or len(actual) != 8 or len(set(actual)) != 8:
        raise ValueError("Fixed crop contract mismatch")
    return manifest, records


def field_metrics(probability, record, threshold=.5, tolerance=6.):
    """Reference agreement, not biological precision; count unknown predictions too."""
    accepted = (probability >= threshold) & record["valid"]
    xyz = np.argwhere(accepted)[:, ::-1].astype(float)
    coverage = 0. if not len(xyz) else float((cKDTree(xyz).query(record["reference_local_xyz"])[0] <= tolerance).mean())
    proximity = 0. if not len(xyz) else float((record["distance"][accepted] <= tolerance).mean())
    predicted_known = accepted & record["known"]
    target = record["target"] & record["known"]
    dice = float(2*(predicted_known & target).sum() / max(1, predicted_known.sum() + target.sum()))
    return {"reference_coverage_6": coverage, "reference_proximity_6": proximity,
            "known_voxel_dice": dice, "accepted_voxels": int(accepted.sum()),
            "accepted_unknown_bright_voxels": int((accepted & record["unknown_bright"]).sum()),
            "mean_reference_tube_score": float(probability[target].mean())}


def gate(metrics, cfg):
    g = cfg["gate"]
    return (all(m["accepted_voxels"] > 0 and m["reference_coverage_6"] >= g["minimum_crop_reference_coverage"]
                and m["reference_proximity_6"] >= g["minimum_crop_reference_proximity"] for m in metrics)
            and float(np.mean([m["known_voxel_dice"] for m in metrics])) >= g["minimum_mean_known_dice"])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run(args):
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; full fit requires explicit device choice")
    device = torch.device(args.device)
    manifest, records = load_records(args.manifest, args.partition)
    source_cfg = manifest["config"]
    duration_recipe = getattr(args, "duration_config", None)
    duration_models = getattr(args, "duration_baseline_models", None)
    if bool(duration_recipe) != bool(duration_models):
        raise ValueError("Duration-only replay requires both duration recipe and completed baseline")
    cfg = source_cfg if duration_recipe is None else duration_config(source_cfg, duration_recipe)
    if cfg["schema"] != "pointneuron2-centerline-fit-v1" or cfg["threshold"] != .5 or cfg["coverage_tolerance_voxels"] != 6:
        raise ValueError("Unsupported frozen protocol")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = CenterlineUNet(cfg["base_channels"]).to(device)
    baseline_models = getattr(args, "baseline_models", None)
    loss_config = getattr(args, "loss_config", None)
    if duration_recipe and (baseline_models or not loss_config):
        raise ValueError("Duration replay needs the frozen loss config and cannot combine with a loss-only control")
    if not duration_recipe and bool(baseline_models) != bool(loss_config):
        raise ValueError("Loss-only control requires both baseline models and loss config")
    pairing = None if baseline_models is None else check_baseline(
        baseline_models, args.manifest, cfg, device, model, loss_config)
    if duration_recipe:
        pairing = check_duration_baseline(duration_models, args.manifest, source_cfg, cfg,
                                          device, model, loss_config, duration_recipe)
    loss_spec = {"negative_aggregation": "all", "negative_to_positive_ratio": 3} if pairing is None else pairing["loss_spec"]
    loss_kwargs = {key: loss_spec[key] for key in ("negative_aggregation", "negative_to_positive_ratio")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    inputs = [torch.from_numpy(r["image"])[None, None].to(device) for r in records]
    targets = [torch.from_numpy(r["target"])[None, None].to(device) for r in records]
    masks = [torch.from_numpy(r["known"])[None, None].to(device) for r in records]
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'} | native full-resolution 3D image", flush=True)
    print(f"Training-only crops: 8 from 4 neurons | batch=1 | U-Net base={cfg['base_channels']} | "
          f"parameters={sum(p.numel() for p in model.parameters()):,} | steps={cfg['steps']}", flush=True)
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)
    print(f"Negative BCE aggregation: {loss_spec['negative_aggregation']} | known negatives only; positive BCE and Dice unchanged", flush=True)
    if pairing:
        print("Baseline pairing checks passed: exact initial model tensors, cache, training settings, device and loss.", flush=True)
    if duration_recipe:
        print("Duration-only fresh replay: 1600 uninterrupted updates; AdamW state retained throughout; saved checks 0/400/800/1600.", flush=True)

    def update(i):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = field_loss(model(inputs[i]), targets[i], masks[i], **loss_kwargs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite field loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())

    if args.preflight:
        if pairing:
            model.eval()
            initial_metrics = []
            with torch.no_grad():
                for i, r in enumerate(records):
                    initial_metrics.append(field_metrics(model(inputs[i]).sigmoid()[0, 0].cpu().numpy(), r))
            assert_initial_metrics(initial_metrics, pairing)
            print("All eight step-zero inference results match the saved baseline.", flush=True)
        print("Timing only: two warm-up and two measured throwaway updates; no model checkpoint retained.", flush=True)
        for i in range(2):
            update(i)
        synchronize(device)
        started = time.perf_counter()
        for i in (2, 3):
            update(i)
        synchronize(device)
        update_seconds = (time.perf_counter() - started)/2
        model.eval()
        started = time.perf_counter()
        with torch.no_grad():
            probability = model(inputs[0]).sigmoid()[0, 0].cpu().numpy()
        field_metrics(probability, records[0])
        synchronize(device)
        eval_seconds = time.perf_counter() - started
        estimate = cfg["steps"] * update_seconds + len(cfg["eval_steps"]) * 8 * eval_seconds
        report = {"status": "completed", "scope": "timing only; throwaway weights",
                  "manifest_sha256": file_digest(args.manifest), "device": str(device),
                  "seconds_per_update": update_seconds, "seconds_per_crop_eval": eval_seconds,
                  "estimated_fit_seconds": estimate, "parameters": sum(p.numel() for p in model.parameters()),
                  "loss_spec": loss_spec, "config": cfg,
                  "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None}
        if pairing:
            assert_baseline_unchanged(pairing)
            report["pairing_checks"] = {**pairing["checks"], "step_zero_inference_matches": True,
                                       "baseline_files_unchanged": True}
        if duration_recipe:
            report["duration_config_sha256"] = pairing["duration_config_sha256"]
        (args.output_dir / "timing.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return report

    started = time.perf_counter()
    report = {"status": "running", "scope": "fixed-training-crop learnability; not validation or SWC quality",
              "manifest_sha256": file_digest(args.manifest), "partition_sha256": manifest["partition_sha256"],
              "config": cfg, "device": str(device), "torch_version": torch.__version__, "evaluations": [],
              "loss_spec": loss_spec,
              "raw_image_cutoff_baseline": []}
    if pairing:
        report["baseline_pairing"] = {key: value for key, value in pairing.items() if key != "baseline_report"}
    if duration_recipe:
        report["duration_config_sha256"] = pairing["duration_config_sha256"]
    for r in records:
        cutoff = r["metadata"]["threshold"] / r["metadata"]["image_scale"]
        report["raw_image_cutoff_baseline"].append({"sample_index": r["row"]["sample_index"], "patch_index": r["row"]["patch_index"],
                                                   **field_metrics((r["image"] > cutoff).astype(float), r)})

    def evaluate(step):
        model.eval()
        metrics = []
        for i, r in enumerate(records):
            with torch.no_grad():
                probability = model(inputs[i]).sigmoid()[0, 0].cpu().numpy()
            measured = {"sample_index": r["row"]["sample_index"], "patch_index": r["row"]["patch_index"],
                        **field_metrics(probability, r, cfg["threshold"], cfg["coverage_tolerance_voxels"])}
            metrics.append(measured)
            name = Path(r["row"]["path"]).stem
            snapshot = args.output_dir / f"step_{step:04d}"
            snapshot.mkdir(exist_ok=True)
            np.savez_compressed(snapshot / (name + ".npz"), probability=probability.astype(np.float16),
                                origin_xyz=np.array(r["metadata"]["origin_xyz"]))
            inspection(r["image"], r["distance"], r["valid"], r["unknown_bright"],
                       f"Training neuron {measured['sample_index']}, crop {measured['patch_index']} | update {step}",
                       snapshot / (name + ".png"), probability)
            print(f"eval step={step} neuron={measured['sample_index']} crop={measured['patch_index']} "
                  f"coverage_6={measured['reference_coverage_6']:.4f} proximity_6={measured['reference_proximity_6']:.4f} "
                  f"known_dice={measured['known_voxel_dice']:.4f} accepted={measured['accepted_voxels']} "
                  f"accepted_unknown={measured['accepted_unknown_bright_voxels']}", flush=True)
        passed = gate(metrics, cfg)
        if step == 0 and pairing:
            assert_initial_metrics(metrics, pairing)
            print("Step-zero inference matches the saved baseline before the first optimizer update.", flush=True)
        report["evaluations"].append({"step": step, "crops": metrics, "gate_passed": bool(passed)})
        torch.save({"schema": SCHEMA, "step": step, "config": cfg, "loss_spec": loss_spec, "manifest_sha256": report["manifest_sha256"],
                    "model": model.state_dict()}, args.output_dir / f"field_step_{step:04d}.pt")
        (args.output_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
        print(f"step={step}: learnability_gate_passed={passed}", flush=True)

    evaluate(0)
    rng, order, visits = np.random.default_rng(cfg["seed"]), [], []
    for step in range(1, cfg["steps"] + 1):
        if not order:
            order = rng.permutation(8).tolist()
        index = order.pop()
        visits.append(index)
        loss = update(index)
        if step == 1 or step % 20 == 0:
            print(f"step={step}/{cfg['steps']} loss={loss:.5f} elapsed={time.perf_counter()-started:.1f}s "
                  f"cuda_allocated={torch.cuda.memory_allocated()/2**20 if device.type == 'cuda' else 0:.1f}MiB", flush=True)
        if step in cfg["eval_steps"]:
            evaluate(step)
    # Reload the exact final checkpoint and verify one inference crop. This is
    # an execution check, separate from the learnability gate.
    checkpoint = torch.load(args.output_dir / f"field_step_{cfg['steps']:04d}.pt", map_location=device, weights_only=True)
    reloaded = CenterlineUNet(cfg["base_channels"]).to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    reloaded.eval()
    model.eval()
    with torch.no_grad():
        delta = float((model(inputs[0]) - reloaded(inputs[0])).abs().max())
    if delta > 1e-6:
        raise AssertionError("Final checkpoint reload changes inference")
    report.update(status="completed", elapsed_seconds=time.perf_counter()-started,
                  checkpoint_reload_max_logit_delta=delta,
                  final_gate_passed=report["evaluations"][-1]["gate_passed"],
                  crop_index_order=visits,
                  next_step="review all crop overlays; validation only if final gate passes")
    if pairing:
        comparison = compare_duration(report, pairing) if duration_recipe else compare_final(report, pairing)
        comparison_path = args.output_dir / ("duration_comparison.json" if duration_recipe else "loss_comparison.json")
        comparison_path.write_text(json.dumps(comparison, indent=2))
        if duration_recipe:
            print(f"Duration-only pairing checks passed; common-budget metrics reproduced="
                  f"{comparison['common_budget_metrics_match_within_tolerance']} | gates={comparison['gate_passed']}", flush=True)
        else:
            print(f"Loss-only comparison: paired checks passed; baseline gate={comparison['baseline_gate_passed']} "
                  f"control gate={comparison['control_gate_passed']}", flush=True)
        for arm, summary in comparison["summary"].items():
            print(f"{arm}: mean coverage_6={summary['reference_coverage_6']:.4f} "
                  f"proximity_6={summary['reference_proximity_6']:.4f} known_dice={summary['known_voxel_dice']:.4f}", flush=True)
        print(f"comparison: {comparison_path}", flush=True)
    (args.output_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
    print(f"Status: completed; checkpoint reload passed; learnability gate: {report['final_gate_passed']}", flush=True)
    print(f"report: {args.output_dir / 'fit_report.json'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", type=Path, default=ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--baseline-models", type=Path)
    parser.add_argument("--loss-config", type=Path)
    parser.add_argument("--duration-baseline-models", type=Path)
    parser.add_argument("--duration-config", type=Path)
    run(parser.parse_args())

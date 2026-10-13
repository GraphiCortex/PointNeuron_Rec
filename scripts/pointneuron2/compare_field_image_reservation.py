"""Cached-field quota comparison; no inference, optimizer or architecture change."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from pointneuron.data.field_guided_sampling import FieldSamplingConfig, SelectedCloud, candidate_pools, local_xyz
from pointneuron.data.field_image_reservation import image_reserved_cloud
from pointneuron.data.field_pilot import load_manifest, write_json
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from scripts.pointneuron2.compare_field_sampling import EXPECTED_CONFIG, matching_source, protected_unchanged, summarize
from scripts.pointneuron2.field_sampling_metrics import cloud_metrics, reference_distances, reference_geometry

EXPECTED_RECIPE = {
    "schema": "pointneuron2-field-image-reservation-v1", "source_model": "fresh_baseline_epoch_30",
    "group": "matched", "budgets": [512, 2048], "sampling_seeds": [0, 1, 2, 3, 4],
    "foreground_fraction": .5, "foreground_rounding": "floor",
    "foreground_selection": "uniform_without_replacement", "fill_selection": "uniform_expanded_excluding_reserved",
    "field_threshold": .5, "comparison_control": "expanded_random", "model_inference_passes": 0, "optimizer_updates": 0,
}
CONTROL, RESERVED = "expanded_random", "image_reserved_uniform"
ENDPOINTS = ("reference_coverage_2", "reference_proximity_6", "worst_chain_coverage_2",
             "longest_unsupported_run_2", "critical_reference_coverage_2")


def safe_path(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Saved artifact path escapes source directory")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_context(args):
    print("Verifying native cache, frozen baseline reports and saved fields; no model inference.", flush=True)
    recipe = read_json(args.recipe)
    if recipe != EXPECTED_RECIPE:
        raise ValueError("Recipe differs from the agreed image-reservation control")
    manifest, cfg = load_manifest(args.manifest, args.partition, args.config)
    fit_path, source_path = args.models / "pilot_report.json", args.source_comparison / "comparison.json"
    fit, source = read_json(fit_path), read_json(source_path)
    checkpoint = args.models / "checkpoints/field_epoch_030.pt"
    if (fit.get("status") != "completed" or fit.get("config") != cfg
            or fit.get("model_spec", {}).get("kind") != "centerline_unet_v1"
            or not fit.get("model_spec", {}).get("shared_initialization_sha256")
            or fit.get("manifest_sha256") != file_digest(args.manifest)
            or fit.get("partition_sha256") != file_digest(args.partition)
            or fit.get("final_checkpoint_sha256") != file_digest(checkpoint)
            or fit.get("checkpoint_reload_max_logit_delta", float("inf")) > 1e-6
            or not fit["evaluations"] or fit["evaluations"][-1]["epoch"] != 30):
        raise ValueError("Expected the completed fresh-baseline epoch-30 fit")
    if (source.get("status") != "completed" or source.get("config") != EXPECTED_CONFIG
            or source.get("source_fit_sha256") != file_digest(fit_path)
            or source.get("source_checkpoint_sha256") != file_digest(checkpoint)
            or source.get("manifest_sha256") != fit["manifest_sha256"]
            or source.get("model_spec") != fit["model_spec"]
            or source.get("optimizer_updates") != 0 or source.get("reserved_test_records") != 0
            or source.get("saved_field_max_score_delta") != 0):
        raise ValueError("Saved sampling comparison does not match the fixed baseline")
    records = [r for r in manifest["records"] if r["partition"] == "val"]
    protected = dict(source["protected_files_sha256"])
    protected.update(source["source_files_sha256"])
    protected.update({str(p.resolve()): file_digest(p) for p in
                      (args.manifest, args.partition, args.config, args.recipe, fit_path, source_path, checkpoint)})
    controls = {}
    for row in source["rows"]:
        if row["group"] == "matched" and row["arm"] == CONTROL:
            key = (row["sample_index"], row["patch_index"], row["nominal_budget"], row["sampling_seed"])
            if key in controls:
                raise ValueError("Duplicate saved control identity")
            controls[key] = row
    expected = {(r["sample_index"], r["patch_index"], budget, seed) for r in records
                for budget in recipe["budgets"] for seed in recipe["sampling_seeds"]}
    if set(controls) != expected:
        raise ValueError("Saved control lacks the exact development/budget/seed pairs")
    fields = fit["final_development_fields_sha256"]
    if set(fields) != {Path(r["path"]).name for r in records}:
        raise ValueError("Saved fit has incomplete development fields")
    for number, row in enumerate(records, 1):
        name = Path(row["path"]).name
        field_path = safe_path(args.models / "development_fields", name)
        if file_digest(field_path) != fields[name]:
            raise ValueError("Saved field hash changed")
        protected[str(field_path)] = fields[name]
        raw, valid, probability, origin, threshold, meta = read_inputs(args, row)
        pools = candidate_pools(raw, valid, probability, threshold)
        if len(pools.foreground) != meta["foreground_count"]:
            raise ValueError("Foreground availability differs from the source")
        for budget in recipe["budgets"]:
            for seed in recipe["sampling_seeds"]:
                saved = controls[row["sample_index"], row["patch_index"], budget, seed]
                if saved["actual_unique_count"] != min(budget, len(pools.foreground)):
                    raise ValueError("Saved matched budget changed")
                path = safe_path(args.source_comparison, saved["cloud_path"])
                if str(path) not in protected:
                    raise ValueError("Saved control cloud lacks a frozen hash")
        print(f"checked inputs {number}/{len(records)} neuron={row['sample_index']} crop={row['patch_index']}", flush=True)
    print("Checking protected source/cloud hashes before any new output.", flush=True)
    protected_unchanged(protected)
    return records, fit, source, controls, protected


def read_inputs(args, row):
    native = safe_path(args.manifest.parent, row["path"])
    with np.load(native, allow_pickle=False) as z:
        # Selection reads image/validity/metadata only; label arrays are not opened.
        raw, valid, image = (z[k].copy() for k in ("raw_image", "valid", "image"))
        meta = json.loads(str(z["metadata"]))
    name = Path(row["path"]).name
    path = safe_path(args.source_comparison / "inputs", name)
    with np.load(path, allow_pickle=False) as z:
        if (not np.array_equal(raw, z["raw_image"]) or not np.array_equal(valid, z["valid"])
                or not np.array_equal(image, z["image"]) or not np.array_equal(meta["origin_xyz"], z["origin_xyz"])
                or float(z["intensity_threshold"]) != meta["threshold"]):
            raise ValueError("Saved inputs differ from the native image cache")
        probability = z["probability"].copy()
    with np.load(safe_path(args.models / "development_fields", name), allow_pickle=False) as z:
        if not np.array_equal(probability, z["probability"]) or not np.array_equal(meta["origin_xyz"], z["origin_xyz"]):
            raise ValueError("Saved comparison differs from the final field/native origin")
    if (not np.array_equal(raw.astype(np.float32) / meta["image_scale"], image)
            or any(meta[k] != row[k] for k in ("sample_index", "sample_id", "patch_index", "partition"))):
        raise ValueError("Invalid native image normalization or identity")
    return raw, valid, probability, np.asarray(meta["origin_xyz"]), meta["threshold"], meta


def metrics_close(left, right):
    for key in ("reference_coverage_2", "reference_proximity_6", "worst_chain_coverage",
                "longest_unsupported_run_voxels", "critical_reference_coverage"):
        a, b = left[key], right[key]
        items = [(a[k], b[k]) for k in a] if isinstance(a, dict) else [(a, b)]
        if any((x is None) != (y is None) or (x is not None and not np.isclose(x, y, atol=1e-8, rtol=1e-8))
               for x, y in items):
            raise ValueError("Saved control measurements were not reproduced")


def flatten(metrics):
    return {"reference_coverage_2": metrics["reference_coverage_2"],
            "reference_proximity_6": metrics["reference_proximity_6"],
            "worst_chain_coverage_2": metrics["worst_chain_coverage"]["2"],
            "longest_unsupported_run_2": metrics["longest_unsupported_run_voxels"]["2"],
            "critical_reference_coverage_2": metrics["critical_reference_coverage"]["2"]}


def paired_summary(rows):
    by_key = {(r["sample_index"], r["patch_index"], r["nominal_budget"], r["sampling_seed"], r["arm"]): r for r in rows}
    pairs = []
    for key, reserved in sorted(by_key.items()):
        if key[-1] != RESERVED:
            continue
        control = by_key[(*key[:-1], CONTROL)]
        if reserved["actual_unique_count"] != control["actual_unique_count"]:
            raise AssertionError("Reservation contrast has unequal unique counts")
        pairs.append({k: reserved[k] for k in ("sample_index", "patch_index", "nominal_budget", "sampling_seed", "actual_unique_count")}
                     | {k + "_delta": None if reserved[k] is None else reserved[k] - control[k] for k in ENDPOINTS})
    per_neuron = []
    for budget, neuron in sorted({(r["nominal_budget"], r["sample_index"]) for r in pairs}):
        own = [r for r in pairs if (r["nominal_budget"], r["sample_index"]) == (budget, neuron)]
        per_neuron.append({"nominal_budget": budget, "sample_index": neuron} | {
            k + "_delta": float(np.mean(values)) if (values := [r[k + "_delta"] for r in own if r[k + "_delta"] is not None]) else None
            for k in ENDPOINTS})
    return {"difference": "image_reserved_uniform minus expanded_random", "pairs": pairs,
            "per_neuron": per_neuron, "gap_interpretation": "Negative unsupported-run deltas are improvements; coverage/proximity improvements are positive."}


def inspection(raw, valid, origin, geometry, clouds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    scale = float(np.quantile(raw[valid], .995)) if valid.any() else 1.
    display = np.clip(raw / max(scale, 1), 0, 1)
    reference = np.concatenate([p["probe"]["points"] for p in geometry["chains"]]) if geometry["chains"] else np.empty((0, 3))
    for col, (arm, cloud, mask, score) in enumerate(clouds):
        xyz = cloud.points[:, :3].astype(float) - origin
        for ax, axis, dims, name in [(axes[0, col], 0, (0, 1), "XY"), (axes[1, col], 1, (0, 2), "XZ")]:
            ax.imshow(display.max(axis=axis), cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.scatter(reference[:, dims[0]], reference[:, dims[1]], s=1, c="cyan", alpha=.6)
            for chosen, color in [(~mask, "orange"), (mask, "lime")]:
                ax.scatter(xyz[chosen, dims[0]], xyz[chosen, dims[1]], s=3, c=color, alpha=.6)
            cov = score["reference_coverage_2"]
            ax.set_title(f"{arm} | {name}\nN={cloud.actual_count}; cov2={cov:.3f}" if cov is not None else f"{arm} | no local reference")
            ax.set_xlabel("Local X voxel"); ax.set_ylabel("Local Y / Z voxel")
    fig.suptitle("Matched budget 2,048, seed 0; frozen fresh-baseline field\nCyan: reference; green: reserved image quota; orange: union fill; projections are not edges", fontsize=10)
    fig.savefig(output, dpi=120); plt.close(fig)


def run(args):
    if args.action == "compare" and args.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory; preserve previous results")
    started = time.perf_counter()
    records, fit, source, controls, protected = load_context(args)
    if args.action == "check":
        print(f"Status: ready; development neurons={len({r['sample_index'] for r in records})}; crops={len(records)}; matched pairs={len(controls)}", flush=True)
        print("CPU cached fields; inference passes=0; optimizer updates=0; no output clouds created.", flush=True)
        return {"status": "ready", "development_crops": len(records), "matched_pairs": len(controls), "inference_passes": 0, "optimizer_updates": 0}
    for folder in ("clouds", "measurements", "inspection"):
        (args.output_dir / folder).mkdir(parents=True)
    sources = [Path(__file__), ROOT / "src/pointneuron/data/field_image_reservation.py",
               ROOT / "scripts/pointneuron2/field_sampling_metrics.py", ROOT / "scripts/pointneuron2/reference_gaps.py",
               ROOT / "src/pointneuron/data/field_guided_sampling.py", ROOT / "src/pointneuron/data/curve_targets.py"]
    protected.update({str(p.resolve()): file_digest(p) for p in sources})
    report_path = args.output_dir / "comparison.json"
    report = {"schema": EXPECTED_RECIPE["schema"], "status": "running", "recipe": EXPECTED_RECIPE,
              "phase": "freezing all new image-only selections before reference scoring", "optimizer_updates": 0,
              "inference_passes": 0, "reserved_test_records": 0, "development_crops": len(records),
              "source_model_spec": fit["model_spec"], "source_comparison_sha256": file_digest(args.source_comparison / "comparison.json"),
              "source_checkpoint_sha256": fit["final_checkpoint_sha256"], "manifest_sha256": file_digest(args.manifest)}
    write_json(report_path, report)
    frozen = []
    try:
        for i, row in enumerate(records, 1):
            raw, valid, prob, origin, threshold, meta = read_inputs(args, row)
            pools = candidate_pools(raw, valid, prob, threshold)
            unique = {}
            for budget in EXPECTED_RECIPE["budgets"]:
                n = min(budget, len(pools.foreground))
                for seed in EXPECTED_RECIPE["sampling_seeds"]:
                    if (n, seed) not in unique:
                        selection_seed = seed + row["sample_index"] * 10000 + row["patch_index"]
                        cloud, mask = image_reserved_cloud(raw, valid, prob, origin, threshold, n, selection_seed, pools=pools)
                        if cloud.actual_count != n or int(mask.sum()) != n // 2:
                            raise AssertionError("Matched reservation did not fill the quota/budget")
                        name = f"sample_{row['sample_index']:04d}_patch_{row['patch_index']:03d}_{RESERVED}_n{n:04d}_seed{seed}.npz"
                        path = args.output_dir / "clouds" / name
                        cm = {k: row[k] for k in ("sample_index", "sample_id", "patch_index", "partition")}
                        cm.update(arm=RESERVED, actual_unique_count=n, selection_seed=selection_seed,
                                  reserved_foreground_count=int(mask.sum()), origin_xyz=origin.tolist(),
                                  selection_is_reference_free=True, field_fallback=False, coordinate_units="native voxels",
                                  reservation_role="reserved_foreground_mask is quota provenance; selection_role is not a guide")
                        np.savez_compressed(path, points=cloud.points, voxel_indices=cloud.voxel_indices,
                                            field_scores=cloud.field_scores, selection_role=cloud.selection_role,
                                            reserved_foreground_mask=mask, metadata=np.array(json.dumps(cm)))
                        protected[str(path.resolve())] = file_digest(path)
                        unique[n, seed] = path
                    frozen.append({k: row[k] for k in ("sample_index", "patch_index")} | {
                        "nominal_budget": budget, "sampling_seed": seed, "group": "matched", "availability_only": False,
                        "arm": RESERVED, "actual_unique_count": n, "cloud_path": str(unique[n, seed].relative_to(args.output_dir))})
            print(f"frozen {i}/{len(records)} neuron={row['sample_index']} crop={row['patch_index']} F={len(pools.foreground)} C={len(pools.expanded)}", flush=True)
        protected_unchanged(protected)
        report.update(phase="all new clouds frozen; reference-only scoring", frozen_rows=len(frozen))
        write_json(report_path, report)
        measured = []
        for i, row in enumerate(records, 1):
            raw, valid, prob, origin, threshold, meta = read_inputs(args, row)
            directory = safe_path(args.data_root, row["sample_id"])
            swc = matching_source(directory, {".swc"}, meta["swc_sha256"])
            protected[str(swc.resolve())] = meta["swc_sha256"]
            geometry = reference_geometry(parse_swc(swc), origin, raw.shape)
            unique_metrics, overlays = {}, []
            own = [r for r in frozen if (r["sample_index"], r["patch_index"]) == (row["sample_index"], row["patch_index"])]
            for reserved_row in own:
                key = (row["sample_index"], row["patch_index"], reserved_row["nominal_budget"], reserved_row["sampling_seed"])
                control = controls[key]
                for arm, item, base in [(CONTROL, control, args.source_comparison), (RESERVED, reserved_row, args.output_dir)]:
                    path = safe_path(base, item["cloud_path"])
                    if path not in unique_metrics:
                        with np.load(path, allow_pickle=False) as z:
                            idx, pts, scores, roles = (z[k].copy() for k in ("voxel_indices", "points", "field_scores", "selection_role"))
                            mask = z["reserved_foreground_mask"].copy() if arm == RESERVED else np.zeros(len(idx), dtype=bool)
                            cm = json.loads(str(z["metadata"]))
                        if (len(idx) != item["actual_unique_count"] or len(np.unique(idx)) != len(idx)
                                or cm["partition"] != "val" or not np.all(valid.ravel()[idx])
                                or not np.array_equal(pts[:, :3], local_xyz(idx, raw.shape) + origin)
                                or not np.array_equal(pts[:, 3], raw.ravel()[idx]) or not np.array_equal(scores, prob.ravel()[idx])
                                or not np.all((raw.ravel()[idx] > threshold) | (scores >= .5))
                                or (arm == RESERVED and (mask.dtype != np.bool_ or mask.shape != idx.shape
                                    or int(mask.sum()) != len(idx)//2 or not np.all(raw.ravel()[idx[mask]] > threshold)))):
                            raise ValueError("Cloud count, coordinate, score or reservation provenance differs")
                        cloud = SelectedCloud(pts, idx, scores, roles, len(idx), len(idx), cm["field_fallback"])
                        distances = reference_distances(local_xyz(idx, raw.shape), geometry["full_segments"])
                        metric = cloud_metrics(cloud, raw, origin, threshold, geometry, idx, distances)
                        if metric["reference_coverage_2"] is None:
                            metric["reference_proximity_6"] = None
                        metric.update(reserved_foreground_count=int(mask.sum()), foreground_point_count=int((raw.ravel()[idx] > threshold).sum()))
                        if arm == CONTROL:
                            metrics_close(metric, item)
                        details = args.output_dir / "measurements" / path.with_suffix(".json").name
                        write_json(details, metric)
                        unique_metrics[path] = (metric, cloud, mask, details)
                    metric, cloud, mask, details = unique_metrics[path]
                    result = {k: reserved_row[k] for k in ("sample_index", "patch_index", "nominal_budget", "sampling_seed", "group", "availability_only")}
                    result.update(arm=arm, actual_unique_count=cloud.actual_count, cloud_path=str(path.resolve()),
                                  details_path=str(details.relative_to(args.output_dir)), **flatten(metric),
                                  reserved_foreground_count=metric["reserved_foreground_count"], foreground_point_count=metric["foreground_point_count"])
                    measured.append(result)
                    if reserved_row["nominal_budget"] == 2048 and reserved_row["sampling_seed"] == 0:
                        overlays.append((arm, cloud, mask, metric))
            inspection(raw, valid, origin, geometry, overlays, args.output_dir / "inspection" / (Path(row["path"]).stem + ".png"))
            print(f"scored {i}/{len(records)} neuron={row['sample_index']} crop={row['patch_index']}; matched coverage/proximity/gaps", flush=True)
        protected_unchanged(protected)
        summaries, _ = summarize(measured)
        report.update(status="completed", phase="completed", rows=measured, summaries=summaries,
                      paired=paired_summary(measured), elapsed_seconds=time.perf_counter()-started,
                      protected_files_sha256=protected, checks={"sources_and_clouds_unchanged": True,
                      "all_new_selections_frozen_before_reference_scoring": True, "matched_counts_and_reservation_provenance": True,
                      "saved_control_geometry_measurements_reproduced": True},
                      next_step="Review per-neuron/crop coverage, proximity and path gaps; no automatic sampler promotion or additional training.")
        write_json(report_path, report)
        columns = [k for k in measured[0] if k not in ("cloud_path", "details_path")]
        with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, columns, extrasaction="ignore"); writer.writeheader(); writer.writerows(measured)
        for summary in summaries:
            print(f"budget={summary['nominal_budget']} arm={summary['arm']} {summary['neuron_macro_mean']}", flush=True)
        print(f"Status: completed; paired execution checks passed; inference passes=0; optimizer updates=0\nreport: {report_path}", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter()-started)
        write_json(report_path, report)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("check", "compare"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--partition", type=Path, default=ROOT / "configs/pointneuron2/partition_seed0.json")
    p.add_argument("--config", type=Path, default=ROOT / "configs/pointneuron2/field_pilot.json")
    p.add_argument("--recipe", type=Path, default=ROOT / "configs/pointneuron2/field_image_reservation.json")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--source-comparison", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

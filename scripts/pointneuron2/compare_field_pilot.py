"""Freeze all development input clouds before scoring; no optimizer updates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pointneuron.data.field_guided_sampling import FieldSamplingConfig, candidate_pools, local_xyz
from pointneuron.data.field_pilot import SCHEMA, NativeRecords, field_metrics, load_manifest, write_json
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from pointneuron.models.centerline_unet import CenterlineUNet
from scripts.pointneuron2.compare_field_sampling import EXPECTED_CONFIG, matching_source, protected_unchanged, save_clouds, summarize, inspect_record
from scripts.pointneuron2.field_sampling_metrics import cloud_metrics, measure_support, reference_distances, reference_geometry


def paired_neuron_deltas(rows, left, right, group="expanded", budget=2048):
    lookup = {(r["sample_index"], r["patch_index"], r["sampling_seed"], r["arm"]): r for r in rows
              if r["group"] == group and r["nominal_budget"] == budget}
    pairs = []
    for sample, patch, seed, arm in sorted(lookup):
        if arm != left:
            continue
        a, b = lookup[sample, patch, seed, left], lookup[sample, patch, seed, right]
        if a["actual_unique_count"] != b["actual_unique_count"]:
            raise AssertionError("Paired contrast has unequal unique-point counts")
        row = {"sample_index": sample, "patch_index": patch, "sampling_seed": seed,
               "actual_unique_count": a["actual_unique_count"]}
        for key in ("reference_coverage_2", "reference_proximity_6"):
            row[key+"_delta"] = None if a[key] is None or b[key] is None else a[key]-b[key]
        aw, bw = a["worst_chain_coverage"]["2"], b["worst_chain_coverage"]["2"]
        row["worst_chain_coverage_2_delta"] = None if aw is None or bw is None else aw-bw
        pairs.append(row)
    per_neuron = []
    for neuron in sorted({r["sample_index"] for r in pairs}):
        row = {"sample_index": neuron}
        for key in ("reference_coverage_2_delta", "reference_proximity_6_delta", "worst_chain_coverage_2_delta"):
            values = [r[key] for r in pairs if r["sample_index"] == neuron and r[key] is not None]
            row[key] = float(np.mean(values)) if values else None
        per_neuron.append(row)
    return {"left_minus_right": f"{left} minus {right}", "group": group, "budget": budget,
            "pairs": pairs, "per_neuron": per_neuron}


def load_contract(args):
    manifest, cfg = load_manifest(args.manifest, args.partition, args.config)
    sampling = json.loads(args.sampling_config.read_text())
    if sampling != EXPECTED_CONFIG:
        raise ValueError("Only the fixed four-control settings are supported")
    report_path = args.models / "pilot_report.json"
    fit = json.loads(report_path.read_text())
    checkpoint_path = args.models / "checkpoints" / f"field_epoch_{cfg['checkpoint_for_sampling']:03d}.pt"
    if (fit.get("schema") != SCHEMA or fit.get("status") != "completed" or fit.get("config") != cfg
            or fit.get("manifest_sha256") != file_digest(args.manifest)
            or fit.get("partition_sha256") != file_digest(args.partition)
            or fit.get("backend", {}).get("torch_version") != str(torch.__version__)
            or fit.get("checkpoint_reload_max_logit_delta", float("inf")) > 1e-6
            or fit.get("final_checkpoint_sha256") != file_digest(checkpoint_path)):
        raise ValueError("Expected completed, reload-verified fixed final pilot checkpoint")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_updates = cfg["epochs"]*sum(r["supervised"] for r in manifest["records"] if r["partition"] == "train")
    if (cp.get("schema") != SCHEMA or cp.get("config") != cfg or cp.get("epoch") != 30
            or cp.get("manifest_sha256") != fit["manifest_sha256"] or cp.get("partition_sha256") != fit["partition_sha256"]
            or cp.get("optimizer_updates") != expected_updates or fit.get("optimizer_updates") != expected_updates):
        raise ValueError("Pilot checkpoint/config/update accounting mismatch")
    if [e["epoch"] for e in fit["evaluations"]] != cfg["snapshot_epochs"]:
        raise ValueError("Pilot lacks the agreed training/development snapshots")
    val_rows = [r for r in manifest["records"] if r["partition"] == "val"]
    identities = [(r["sample_index"], r["patch_index"]) for r in val_rows]
    if any([(c["sample_index"], c["patch_index"]) for c in e["val"]["crops"]] != identities for e in fit["evaluations"]):
        raise ValueError("Development evaluation identities changed")
    expected_fields = {Path(r["path"]).name for r in val_rows}
    if set(fit["final_development_fields_sha256"]) != expected_fields:
        raise ValueError("Incomplete full-precision development inference cache")
    model = CenterlineUNet(cfg["base_channels"]).to(args.device).eval().requires_grad_(False)
    model.load_state_dict(cp["model"], strict=True)
    protected = {str(p.resolve()): file_digest(p) for p in (args.manifest, args.partition, args.config, args.sampling_config, report_path, checkpoint_path)}
    for row in manifest["records"]:
        protected[str((args.manifest.parent/row["path"]).resolve())] = row["sha256"]
    for name, digest in fit["final_development_fields_sha256"].items():
        path = args.models / "development_fields" / name
        if file_digest(path) != digest:
            raise ValueError("Saved development field changed")
        protected[str(path.resolve())] = digest
    return manifest, cfg, sampling, fit, model, protected


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose the device explicitly")
    manifest, cfg, config, fit, model, protected = load_contract(args)
    torch.set_num_threads(cfg["torch_threads"])
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh comparison directory")
    for folder in ("inputs", "clouds", "measurements", "inspection"):
        (args.output_dir / folder).mkdir(parents=True)
    source_paths = [Path(__file__), ROOT / "src/pointneuron/data/field_pilot.py",
                    ROOT / "src/pointneuron/data/field_guided_sampling.py", ROOT / "scripts/pointneuron2/compare_field_sampling.py",
                    ROOT / "scripts/pointneuron2/field_sampling_metrics.py", ROOT / "scripts/pointneuron2/reference_gaps.py",
                    ROOT / "scripts/pointneuron2/audit_field_centerpaths.py", ROOT / "src/pointneuron/data/curve_targets.py",
                    ROOT / "src/pointneuron/models/centerline_unet.py"]
    source_hashes = {str(p.resolve()): file_digest(p) for p in source_paths}
    protected.update(source_hashes)
    sampling_cfg = FieldSamplingConfig(config["field_threshold"], config["cell_size"], config["guide_fraction"])
    val_rows = [r for r in manifest["records"] if r["partition"] == "val"]
    started = time.perf_counter()
    report_path = args.output_dir / "comparison.json"
    report = {"status": "running", "scope": "seven development neurons, image-defined crop regions; not reserved test or final SWC",
              "config": config, "pilot_config": cfg, "optimizer_updates": 0, "phase": "freezing every development input cloud",
              "source_fit_sha256": file_digest(args.models/"pilot_report.json"),
              "source_checkpoint_sha256": fit["final_checkpoint_sha256"], "manifest_sha256": file_digest(args.manifest),
              "source_files_sha256": source_hashes, "reserved_test_records": 0}
    write_json(report_path, report)
    frozen_rows, maximum_field_delta, geometry_loads = [], 0., 0
    try:
        print(f"Development crops: {len(val_rows)} | fresh frozen inference; no optimizer | controls A/B/C/D, budgets 512/2048, seeds 0-4", flush=True)
        for number, row in enumerate(val_rows, 1):
            # Only image/coordinate arrays are opened here; labels and SWCs wait until every cloud is sealed.
            with np.load(args.manifest.parent/row["path"], allow_pickle=False) as z:
                image, raw, valid = z["image"].copy(), z["raw_image"].copy(), z["valid"].copy()
                meta = json.loads(str(z["metadata"]))
            with torch.inference_mode():
                probability = model(torch.from_numpy(image)[None, None].to(args.device)).sigmoid()[0, 0].cpu().numpy().copy()
            with np.load(args.models/"development_fields"/Path(row["path"]).name, allow_pickle=False) as z:
                saved = z["probability"]
                if saved.dtype != np.float32 or not np.array_equal(z["origin_xyz"], meta["origin_xyz"]):
                    raise ValueError("Saved field precision/coordinate mismatch")
                delta = float(np.abs(saved-probability).max())
                maximum_field_delta = max(maximum_field_delta, delta)
                if delta > 1e-6 or not np.array_equal(saved >= .5, probability >= .5):
                    raise ValueError("Reloaded field differs from saved scores or threshold membership")
            pools = candidate_pools(raw, valid, probability, meta["threshold"], sampling_cfg)
            if len(pools.foreground) != meta["foreground_count"]:
                raise ValueError("Historical foreground count not reproduced")
            record = {"name": Path(row["path"]).stem, "row": row, "metadata": meta, "origin": np.array(meta["origin_xyz"]),
                      "threshold": meta["threshold"], "raw": raw, "image": image, "valid": valid,
                      "probability": probability, "pools": pools}
            input_path = args.output_dir/"inputs"/Path(row["path"]).name
            np.savez_compressed(input_path, raw_image=raw, image=image, valid=valid, probability=probability,
                                origin_xyz=record["origin"], intensity_threshold=meta["threshold"])
            protected[str(input_path.resolve())] = file_digest(input_path)
            rows, unique = save_clouds(record, args.output_dir, config, sampling_cfg, protected, False)
            # Retain filenames only; large arrays/cloud objects are released between crops.
            for item in rows:
                item["selection_seconds"] = unique[item.pop("unique_key")]["selection_seconds"]
            frozen_rows.extend(rows)
            print(f"frozen {number}/{len(val_rows)} neuron={row['sample_index']} crop={row['patch_index']} "
                  f"F={len(pools.foreground)} C={len(pools.expanded)}; all selection reference-free", flush=True)
        protected_unchanged(protected)
        report.update(phase="all clouds frozen; reference-only scoring", frozen_rows=len(frozen_rows))
        write_json(report_path, report)
        cache = NativeRecords(args.manifest, cfg["cpu_cache_records"])
        candidates, measured = [], []
        for number, row in enumerate(val_rows, 1):
            label_record = cache.get(row)
            meta = label_record["metadata"]
            name = Path(row["path"]).stem
            with np.load(args.output_dir/"inputs"/Path(row["path"]).name, allow_pickle=False) as z:
                raw, image, valid, probability = (z[k].copy() for k in ("raw_image", "image", "valid", "probability"))
            actual = field_metrics(probability, label_record)
            original = next(c for c in fit["evaluations"][-1]["val"]["crops"]
                            if (c["sample_index"], c["patch_index"]) == (row["sample_index"], row["patch_index"]))
            for key, value in actual.items():
                if value is None:
                    if original[key] is not None:
                        raise ValueError("Unannotated field metric changed")
                elif original[key] is None or abs(value-original[key]) > 1e-6:
                    raise ValueError("Saved development measurements not reproduced")
            directory = (args.data_root/row["sample_id"]).resolve()
            if not directory.is_relative_to(args.data_root.resolve()):
                raise ValueError("Reference path escapes dataset")
            swc_path = matching_source(directory, {".swc"}, meta["swc_sha256"])
            protected[str(swc_path)] = meta["swc_sha256"]
            geometry = reference_geometry(parse_swc(swc_path), np.array(meta["origin_xyz"]), raw.shape)
            geometry_loads += 1
            pools = candidate_pools(raw, valid, probability, meta["threshold"], sampling_cfg)
            union = pools.expanded
            distances = reference_distances(local_xyz(union, raw.shape), geometry["full_segments"])
            candidate = {"sample_index": row["sample_index"], "patch_index": row["patch_index"],
                         "foreground_count": len(pools.foreground), "guide_count": len(pools.guide), "expanded_count": len(union),
                         "zero_intensity_additions": int((raw.ravel()[union] == 0).sum()),
                         "supervised": row["supervised"], "field_metrics": actual}
            for key, indices in (("foreground", pools.foreground), ("expanded", union)):
                support = measure_support(local_xyz(indices, raw.shape), geometry)
                candidate[key+"_ceiling"] = {k: v for k, v in support.items() if k != "chains"}
            candidates.append(candidate)
            own_rows = [r for r in frozen_rows if (r["sample_index"], r["patch_index"]) == (row["sample_index"], row["patch_index"])]
            clouds = {}
            for item in own_rows:
                cloud_path = args.output_dir/item["cloud_path"]
                if item["cloud_path"] not in clouds:
                    from pointneuron.data.field_guided_sampling import SelectedCloud
                    with np.load(cloud_path, allow_pickle=False) as z:
                        cm = json.loads(str(z["metadata"]))
                        cloud = SelectedCloud(z["points"].copy(), z["voxel_indices"].copy(), z["field_scores"].copy(),
                                              z["selection_role"].copy(), cm["actual_unique_count"],
                                              cm["actual_unique_count"], cm["field_fallback"])
                    metrics = cloud_metrics(cloud, raw, np.array(meta["origin_xyz"]), meta["threshold"], geometry, union, distances)
                    if metrics["duplicate_count"] != 0 or cm["partition"] != "val":
                        raise AssertionError("Development cloud provenance/count check failed")
                    # A crop without an in-box reference is not a verified biological negative.
                    if metrics["reference_coverage_2"] is None:
                        metrics["reference_proximity_6"] = None
                    metric_path = args.output_dir/"measurements"/(cloud_path.stem+".json")
                    write_json(metric_path, metrics)
                    clouds[item["cloud_path"]] = {"cloud": cloud, "path": cloud_path, "metrics": metrics, "metric_path": metric_path}
                data = clouds[item["cloud_path"]]
                result = dict(item)
                result.update({k: v for k, v in data["metrics"].items() if k not in ("chains", "roles")})
                result["details_path"] = str(data["metric_path"].relative_to(args.output_dir))
                measured.append(result)
            record = {"name": name, "row": row, "metadata": meta, "origin": np.array(meta["origin_xyz"]),
                      "image": image, "valid": valid, "geometry": geometry, "unique_clouds": clouds}
            inspect_record(record, measured, args.output_dir)
            print(f"scored {number}/{len(val_rows)} neuron={row['sample_index']} crop={row['patch_index']} "
                  f"reference_length={candidate['expanded_ceiling']['reference_length_voxels']:.2f}; saved XY/XZ overlay", flush=True)
        summaries, primary_pairs = summarize(measured)
        contrasts = {
            "expanded_random_minus_threshold_matched": paired_neuron_deltas(measured, "expanded_random", "threshold_random", "matched"),
            "guided_minus_expanded_random": paired_neuron_deltas(measured, "guided_context", "expanded_random"),
            "guided_minus_spatial": paired_neuron_deltas(measured, "guided_context", "expanded_coverage"),
        }
        with (args.output_dir/"metrics.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["sample_index", "patch_index", "group", "nominal_budget", "sampling_seed", "arm", "actual_unique_count",
                      "availability_only", "reference_coverage_2", "reference_proximity_6", "cloud_path", "details_path"]
            writer = csv.DictWriter(f, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(measured)
        protected_unchanged(protected)
        report.update(status="completed", phase="completed", candidate_availability=candidates, rows=measured, summaries=summaries,
                      primary_guided_minus_coverage_pairs=primary_pairs, contrasts=contrasts,
                      inference_passes=len(val_rows), reference_geometry_loads=geometry_loads,
                      saved_field_max_score_delta=maximum_field_delta, elapsed_seconds=time.perf_counter()-started,
                      protected_files_sha256=protected,
                      checks={"all_clouds_frozen_before_reference_scoring": True, "sources_and_clouds_unchanged": True,
                              "matched_unique_counts_and_no_duplicates": True, "saved_final_development_fields_reproduced": True},
                      next_step="Review per-neuron coverage/proximity, missing chains and training fit; no automatic sampler promotion, extension or architecture replacement.")
        write_json(report_path, report)
        for summary in summaries:
            if summary["group"] == "expanded" and summary["nominal_budget"] == 2048:
                print(f"{summary['arm']}: {summary['neuron_macro_mean']}", flush=True)
        print(f"Status: completed; execution checks passed; optimizer updates=0\nreport: {report_path.resolve()}", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/pointneuron2/field_pilot.json")
    p.add_argument("--sampling-config", type=Path, default=ROOT / "configs/pointneuron2/field_sampling_comparison.json")
    p.add_argument("--partition", type=Path, default=ROOT / "configs/pointneuron2/partition_seed0.json")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(p.parse_args())

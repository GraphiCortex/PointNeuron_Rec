"""User-visible frozen-field Phase 1 comparison; no optimizer or later PN2 modules."""
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

from pointneuron.data.centerline_field import SCHEMA, crop_volume
from pointneuron.data.field_guided_sampling import ARMS, FieldSamplingConfig, candidate_pools, select_cloud, local_xyz
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import volume_data_array
from pointneuron.data.vaa3d_raw import read_volume, VOLUME_READER_VERSION
from pointneuron.models.centerline_unet import CenterlineUNet
from scripts.pointneuron2.centerline_duration_comparison import completed_fit_config
from scripts.pointneuron2.train_centerline_fit import field_metrics
from scripts.pointneuron2.field_sampling_metrics import reference_geometry, reference_distances, cloud_metrics, measure_support

EXPECTED_CONFIG = {"schema": "pointneuron2-field-sampling-comparison-v1", "field_threshold": .5,
                   "cell_size": 8, "guide_fraction": .5, "budgets": [512, 2048], "sampling_seeds": [0, 1, 2, 3, 4],
                   "probe_step_voxels": 1., "tolerances_voxels": [1, 2, 4, 6], "primary_group": "expanded",
                   "primary_budget": 2048, "primary_endpoint": "reference_coverage_2"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def protected_unchanged(protected):
    if any(file_digest(p) != value for p, value in protected.items()):
        raise AssertionError("Protected source or frozen cloud changed")


def matching_source(directory, suffixes, sha):
    matches = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes and file_digest(p) == sha]
    if len(matches) != 1:
        raise ValueError("Cannot recover exactly one original hashed source")
    return matches[0]


def load_contract(args):
    manifest = json.loads(args.manifest.read_text())
    partition = json.loads(args.partition.read_text())
    config = json.loads(args.config.read_text())
    if config != EXPECTED_CONFIG:
        raise ValueError("Only the reviewed four-arm settings are supported")
    if (manifest.get("status") != "completed" or manifest.get("schema") != SCHEMA
            or manifest.get("volume_reader_version") != VOLUME_READER_VERSION
            or manifest["partition_sha256"] != file_digest(args.partition)):
        raise ValueError("Field cache/reader/partition mismatch")
    allowed = {(r["sample_index"], r["sample_id"]) for r in partition["partitions"]["train"]}
    reserved = {(r["sample_index"], r["sample_id"]) for role in ("val", "test") for r in partition["partitions"][role]}
    if {i for i, _ in allowed} & {i for i, _ in reserved} or {s for _, s in allowed} & {s for _, s in reserved}:
        raise ValueError("Crossed neuron partitions")
    identities = [(r["sample_index"], r["patch_index"]) for r in manifest["records"]]
    if identities != [tuple(c) for c in manifest["config"]["crops"]] or len(set(identities)) != 8:
        raise ValueError("Expected the immutable eight-crop learnability cache")
    paths, seen = [], set()
    for row in manifest["records"]:
        if row["partition"] != "train" or (row["sample_index"], row["sample_id"]) not in allowed:
            raise ValueError("Reserved or non-training input")
        path = (args.manifest.parent / row["path"]).resolve()
        if not path.is_relative_to(args.manifest.parent.resolve()) or path in seen or file_digest(path) != row["sha256"]:
            raise ValueError("Invalid or changed input record")
        seen.add(path)
        paths.append(path)
    fit_path = args.models / "fit_report.json"
    fit = json.loads(fit_path.read_text())
    cfg = completed_fit_config(manifest["config"], fit, ROOT / "configs/pointneuron2/centerline_duration.json")
    if (fit["status"] != "completed" or not fit["final_gate_passed"] or cfg["steps"] != 1600
            or fit["manifest_sha256"] != file_digest(args.manifest) or fit["torch_version"] != str(torch.__version__)
            or fit.get("checkpoint_reload_max_logit_delta", float("inf")) > 1e-6):
        raise ValueError("Expected the passing completed duration checkpoint")
    if [e["step"] for e in fit["evaluations"]] != cfg["eval_steps"]:
        raise ValueError("Completed field report has missing evaluations")
    for evaluation in fit["evaluations"]:
        if [(c["sample_index"], c["patch_index"]) for c in evaluation["crops"]] != identities:
            raise ValueError("Completed field evaluation identities differ from input cache")
    checkpoint_path = args.models / "field_step_1600.pt"
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (cp["schema"] != SCHEMA or cp["step"] != 1600 or cp["config"] != cfg
            or cp["manifest_sha256"] != fit["manifest_sha256"] or cp["loss_spec"] != fit["loss_spec"]):
        raise ValueError("Frozen checkpoint contract mismatch")
    model = CenterlineUNet(cfg["base_channels"]).to(args.device).eval().requires_grad_(False)
    model.load_state_dict(cp["model"], strict=True)
    protected = {str(p.resolve()): file_digest(p) for p in
                 [args.manifest, args.partition, args.config, fit_path, checkpoint_path, *paths]}
    return manifest, fit, model, protected


class RawImages:
    def __init__(self, data_root, protected):
        self.root = data_root.resolve()
        self.protected = protected
        self.identity = None
        self.volume = None

    def crop(self, metadata, image, valid):
        started = time.perf_counter()
        directory = (self.root / metadata["sample_id"]).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("Image path escapes dataset")
        if self.identity != metadata["sample_id"]:
            path = matching_source(directory, {".v3draw", ".v3dpbd"}, metadata["volume_sha256"])
            decoded = read_volume(path)
            x, y, z, channels = decoded.dimensions
            if channels != 1:
                raise ValueError("Expected a single image channel")
            self.volume = volume_data_array(decoded).reshape(z, y, x)
            self.identity = metadata["sample_id"]
            self.protected[str(path)] = metadata["volume_sha256"]
            if float(self.volume.max()) != metadata["image_scale"]:
                raise ValueError("Full-image normalization scale differs from cached input")
        raw, actual_valid, origin = crop_volume(self.volume, metadata["patch_center"], metadata["config"]["patch_radius"])
        if (not np.array_equal(actual_valid, valid) or not np.array_equal(origin, metadata["origin_xyz"])
                or not np.array_equal(raw.astype(np.float32) / metadata["image_scale"], image)):
            raise ValueError("Raw crop does not reproduce cached image/coordinates/validity")
        restored = np.rint(image.astype(np.float64) * metadata["image_scale"]).astype(raw.dtype)
        if not np.array_equal(restored, raw):
            raise ValueError("Intensity restoration is not exact for the decoded source")
        return raw, time.perf_counter() - started


def save_clouds(record, output, config, sampling_config, protected, preflight):
    rows, unique = [], {}
    budgets = [2048] if preflight else config["budgets"]
    seeds = [0] if preflight else config["sampling_seeds"]
    for budget in budgets:
        for seed in seeds:
            selection_seed = seed + record["row"]["sample_index"]*10000 + record["row"]["patch_index"]
            for group, desired in (("matched", min(budget, len(record["pools"].foreground))),
                                   ("expanded", min(budget, len(record["pools"].expanded)))):
                for arm in ARMS:
                    n = min(budget, len(record["pools"].foreground)) if arm == ARMS[0] else desired
                    key = (arm, n, seed)
                    if key not in unique:
                        started = time.perf_counter()
                        cloud = select_cloud(record["raw"], record["valid"], record["probability"], record["origin"],
                                             record["threshold"], arm, n, selection_seed, sampling_config, pools=record["pools"])
                        if cloud.actual_count != n:
                            raise AssertionError("Matched-count allocation did not fill its unique budget")
                        elapsed = time.perf_counter() - started
                        path = output / "clouds" / f"{record['name']}_{arm}_n{n:04d}_seed{seed}.npz"
                        metadata = {"schema": config["schema"], "sample_index": record["row"]["sample_index"],
                                    "patch_index": record["row"]["patch_index"], "partition": "train", "arm": arm,
                                    "actual_unique_count": n, "seed": selection_seed,
                                    "origin_xyz": record["origin"].tolist(), "image_scale": record["metadata"]["image_scale"],
                                    "raw_intensity_dtype": str(record["raw"].dtype), "field_fallback": cloud.fallback,
                                    "coordinate_units": "native voxels", "selection_is_reference_free": True}
                        np.savez_compressed(path, points=cloud.points, voxel_indices=cloud.voxel_indices,
                                            field_scores=cloud.field_scores, selection_role=cloud.selection_role,
                                            metadata=np.array(json.dumps(metadata)))
                        protected[str(path.resolve())] = file_digest(path)
                        unique[key] = {"cloud": cloud, "path": path, "selection_seconds": elapsed}
                    item = unique[key]
                    rows.append({"sample_index": record["row"]["sample_index"], "patch_index": record["row"]["patch_index"],
                                 "group": group, "nominal_budget": budget, "sampling_seed": seed, "arm": arm,
                                 "actual_unique_count": n, "availability_only": group == "expanded" and arm == ARMS[0]
                                    and n != desired,
                                 "cloud_path": str(item["path"].relative_to(output)), "unique_key": key})
    return rows, unique


def freeze_inputs(args, manifest, model, protected, config, sampling_config):
    records, all_rows = [], []
    images = RawImages(args.data_root, protected)
    source_rows = [r for r in manifest["records"] if not args.preflight or (r["sample_index"], r["patch_index"]) == (96, 3)]
    for number, row in enumerate(source_rows, 1):
        path = args.manifest.parent / row["path"]
        with np.load(path, allow_pickle=False) as z:
            image, valid = z["image"].copy(), z["valid"].copy()
            meta = json.loads(str(z["metadata"]))
        if meta["field_schema"] != SCHEMA or any(meta[k] != row[k] for k in ("sample_index", "patch_index", "sample_id", "partition")):
            raise ValueError("Input record identity mismatch")
        print(f"[{number}/{len(source_rows)}] CPU image verification: neuron={row['sample_index']} crop={row['patch_index']}", flush=True)
        raw, raw_seconds = images.crop(meta, image, valid)
        started = time.perf_counter()
        with torch.inference_mode():
            probability = model(torch.from_numpy(image)[None, None].to(args.device)).sigmoid()[0, 0].cpu().numpy().copy()
        infer_seconds = time.perf_counter() - started
        pools = candidate_pools(raw, valid, probability, meta["threshold"], sampling_config)
        if len(pools.foreground) != meta["foreground_count"]:
            raise ValueError("Exact raw cutoff does not reproduce the historical foreground count")
        record = {"name": path.stem, "row": row, "metadata": meta, "origin": np.array(meta["origin_xyz"]),
                  "threshold": meta["threshold"], "raw": raw, "image": image, "valid": valid,
                  "probability": probability, "pools": pools, "raw_seconds": raw_seconds,
                  "inference_seconds": infer_seconds}
        input_path = args.output_dir / "inputs" / path.name
        np.savez_compressed(input_path, raw_image=raw, image=image, valid=valid, probability=probability,
                            origin_xyz=record["origin"], intensity_threshold=meta["threshold"])
        protected[str(input_path.resolve())] = file_digest(input_path)
        rows, unique = save_clouds(record, args.output_dir, config, sampling_config, protected, args.preflight)
        record["unique_clouds"] = unique
        records.append(record)
        all_rows.extend(rows)
        print(f"Frozen clouds: F={len(pools.foreground)} P={len(pools.guide)} C={len(pools.expanded)} "
              f"unique cloud files={len(unique)} | no reference geometry supplied", flush=True)
    images.volume = None
    return records, all_rows


def score_records(args, records, rows, fit, protected):
    evaluation_started = time.perf_counter()
    candidate_reports, measured_rows = [], []
    for record in records:
        row, meta = record["row"], record["metadata"]
        # Only now, after every input cloud is frozen, load label arrays and SWC geometry.
        with np.load(args.manifest.parent / row["path"], allow_pickle=False) as z:
            labels = {k: z[k] for k in ("target", "known", "distance", "unknown_bright", "reference_local_xyz")}
        labels["valid"] = record["valid"]
        actual = field_metrics(record["probability"], labels)
        original = next(c for c in fit["evaluations"][-1]["crops"]
                        if (c["sample_index"], c["patch_index"]) == (row["sample_index"], row["patch_index"]))
        if any(abs(actual[k]-original[k]) > 1e-5 for k in ("reference_coverage_6", "reference_proximity_6", "known_voxel_dice")):
            raise ValueError("Frozen inference does not reproduce completed field measurements")
        if actual["accepted_voxels"] != original["accepted_voxels"]:
            raise ValueError("Full-precision threshold membership differs from saved final inference")
        directory = (args.data_root / row["sample_id"]).resolve()
        swc_path = matching_source(directory, {".swc"}, meta["swc_sha256"])
        protected[str(swc_path)] = meta["swc_sha256"]
        started = time.perf_counter()
        geometry = reference_geometry(parse_swc(swc_path), record["origin"], record["raw"].shape)
        union = record["pools"].expanded
        distances = reference_distances(local_xyz(union, record["raw"].shape), geometry["full_segments"])
        record["reference_setup_seconds"] = time.perf_counter() - started
        record["geometry"] = geometry
        candidate_report = {"sample_index": row["sample_index"], "patch_index": row["patch_index"],
                            "raw_image_and_intensity_restoration_exact": True, "frozen_field_metrics_reproduced": True,
                            "foreground_count": len(record["pools"].foreground), "guide_count": len(record["pools"].guide),
                            "expanded_count": len(union), "image_datatype": str(record["raw"].dtype),
                            "below_threshold_additions": len(union)-len(record["pools"].foreground),
                            "zero_intensity_additions": int((record["raw"].ravel()[union] == 0).sum())}
        for name, indices in (("foreground", record["pools"].foreground), ("expanded", union)):
            support = measure_support(local_xyz(indices, record["raw"].shape), geometry)
            candidate_report[f"{name}_ceiling"] = {k: v for k, v in support.items() if k != "chains"}
        candidate_reports.append(candidate_report)
        for item in record["unique_clouds"].values():
            started = time.perf_counter()
            metrics = cloud_metrics(item["cloud"], record["raw"], record["origin"], record["threshold"], geometry, union, distances)
            item["scoring_seconds"] = time.perf_counter() - started
            item["metrics"] = metrics
            metric_path = args.output_dir / "measurements" / (item["path"].stem + ".json")
            write_json(metric_path, metrics)
            item["metric_path"] = metric_path
        for original_row in rows:
            if (original_row["sample_index"], original_row["patch_index"]) != (row["sample_index"], row["patch_index"]):
                continue
            item = record["unique_clouds"][original_row["unique_key"]]
            result = {k: v for k, v in original_row.items() if k != "unique_key"}
            result.update({k: v for k, v in item["metrics"].items() if k not in ("chains", "roles")})
            result.update(selection_seconds=item["selection_seconds"], scoring_seconds=item["scoring_seconds"],
                          details_path=str(item["metric_path"].relative_to(args.output_dir)))
            measured_rows.append(result)
        print(f"Scored neuron={row['sample_index']} crop={row['patch_index']} | frozen field reproduced | "
              f"reference chains={len(geometry['chains'])} | original files unchanged", flush=True)
    protected_unchanged(protected)
    return candidate_reports, measured_rows, time.perf_counter() - evaluation_started


def summarize(rows):
    metrics = ("reference_coverage_2", "reference_proximity_6")
    summaries = []
    keys = sorted({(r["group"], r["nominal_budget"], r["arm"]) for r in rows})
    for group, budget, arm in keys:
        relevant = [r for r in rows if (r["group"], r["nominal_budget"], r["arm"]) == (group, budget, arm)]
        macro = {}
        for metric in metrics:
            neuron_values = []
            for neuron in sorted({r["sample_index"] for r in relevant}):
                values = [r[metric] for r in relevant if r["sample_index"] == neuron and r[metric] is not None]
                if values:
                    neuron_values.append(float(np.mean(values)))
            macro[metric] = float(np.mean(neuron_values)) if neuron_values else None
        summaries.append({"group": group, "nominal_budget": budget, "arm": arm,
                          "neuron_macro_mean": macro, "availability_only_pairs": sum(r["availability_only"] for r in relevant)})
    pairs = []
    lookup = {(r["sample_index"], r["patch_index"], r["sampling_seed"], r["arm"]): r
              for r in rows if r["group"] == "expanded" and r["nominal_budget"] == 2048}
    for sample, patch, seed, arm in sorted(lookup):
        if arm != "guided_context":
            continue
        guided, control = lookup[sample, patch, seed, arm], lookup[sample, patch, seed, "expanded_coverage"]
        if guided["actual_unique_count"] != control["actual_unique_count"]:
            raise AssertionError("Primary comparison has mismatched unique-point counts")
        pairs.append({"sample_index": sample, "patch_index": patch, "sampling_seed": seed,
                      "actual_unique_count": guided["actual_unique_count"],
                      "reference_coverage_2_delta": None if guided["reference_coverage_2"] is None else
                          guided["reference_coverage_2"]-control["reference_coverage_2"],
                      "reference_proximity_6_delta": None if guided["reference_proximity_6"] is None else
                          guided["reference_proximity_6"]-control["reference_proximity_6"],
                      "worst_chain_coverage_2_delta": None if guided["worst_chain_coverage"]["2"] is None else
                          guided["worst_chain_coverage"]["2"]-control["worst_chain_coverage"]["2"]})
    return summaries, pairs


def inspect_record(record, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r["group"] == "expanded" and r["nominal_budget"] == 2048
                and r["sampling_seed"] == 0 and (r["sample_index"], r["patch_index"]) ==
                (record["row"]["sample_index"], record["row"]["patch_index"])]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    scale = float(np.quantile(record["image"][record["valid"]], .995)) or float(record["image"].max())
    display = np.clip(record["image"] / max(scale, 1e-6), 0, 1)
    segments = np.concatenate([p["probe"]["points"] for p in record["geometry"]["chains"]]) if record["geometry"]["chains"] else np.empty((0, 3))
    for column, arm in enumerate(ARMS):
        row = next(r for r in selected if r["arm"] == arm)
        item = next(x for x in record["unique_clouds"].values() if str(x["path"].relative_to(output)) == row["cloud_path"])
        xyz = item["cloud"].points[:, :3].astype(float)-record["origin"]
        roles = item["cloud"].selection_role
        for projection, (axis, dims, plane) in enumerate(((0, (0, 1), "XY"), (1, (0, 2), "XZ"))):
            ax = axes[projection, column]
            ax.imshow(display.max(axis=axis), cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.scatter(segments[:, dims[0]], segments[:, dims[1]], s=1, color="cyan", alpha=.6)
            for mask, color in ((roles == 0, "orange"), (roles == 1, "lime")):
                ax.scatter(xyz[mask, dims[0]], xyz[mask, dims[1]], s=3, color=color, alpha=.6)
            value = row["reference_coverage_2"]
            ax.set_title(f"{arm.replace('_', ' ')} | {plane}\nunique N={len(xyz)} | cov2={value:.3f}" if value is not None else arm)
            ax.set_xlabel("local X voxel")
            ax.set_ylabel("local Y / Z voxel")
    fig.suptitle(f"Training crop {record['row']['sample_index']}/{record['row']['patch_index']} | nominal 2048, seed 0\n"
                 "cyan: reference; orange: context; green: guide; A may have fewer unique points; projections are not connections")
    fig.savefig(output / "inspection" / (record["name"] + ".png"), dpi=120)
    plt.close(fig)


def run(args):
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose the device explicitly")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    started = time.perf_counter()
    manifest, fit, model, protected = load_contract(args)
    config = json.loads(args.config.read_text())
    sampling_config = FieldSamplingConfig(config["field_threshold"], config["cell_size"], config["guide_fraction"])
    source_files = [Path(__file__), ROOT / "src/pointneuron/data/field_guided_sampling.py",
                    ROOT / "scripts/pointneuron2/field_sampling_metrics.py", ROOT / "scripts/pointneuron2/reference_gaps.py",
                    ROOT / "scripts/pointneuron2/audit_field_centerpaths.py", ROOT / "src/pointneuron/data/curve_targets.py",
                    ROOT / "src/pointneuron/data/vaa3d_raw.py", ROOT / "src/pointneuron/data/centerline_field.py",
                    ROOT / "src/pointneuron/models/centerline_unet.py"]
    source_hashes = {str(p.resolve()): file_digest(p) for p in source_files}
    protected.update(source_hashes)
    for folder in ("inputs", "clouds", "measurements", "inspection"):
        (args.output_dir / folder).mkdir(parents=True)
    report_path = args.output_dir / ("timing.json" if args.preflight else "comparison.json")
    report = {"status": "running", "scope": "eight fitted training crops; isolated Phase 1 inputs; not validation or complete PN2",
              "config": config, "device": args.device, "torch_version": str(torch.__version__),
              "source_fit_sha256": file_digest(args.models / "fit_report.json"),
              "source_checkpoint_sha256": file_digest(args.models / "field_step_1600.pt"),
              "manifest_sha256": file_digest(args.manifest), "source_files_sha256": source_hashes,
              "phase": "freezing image-derived inputs"}
    write_json(report_path, report)
    print(f"Device: {torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU'} | frozen field inference only", flush=True)
    print("No optimizer; no PointNeXt, Transformer or GATv2 execution. References are loaded only after input clouds are frozen.", flush=True)
    try:
        records, rows = freeze_inputs(args, manifest, model, protected, config, sampling_config)
        if not records:
            raise ValueError("No approved input crop found")
        protected_unchanged(protected)
        report["phase"] = "reference-only scoring; all input clouds frozen"
        write_json(report_path, report)
        candidates, measured, scoring_seconds = score_records(args, records, rows, fit, protected)
        summaries, pairs = summarize(measured)
        if args.preflight:
            record = records[0]
            selection_cycle = sum(x["selection_seconds"] for x in record["unique_clouds"].values())
            score_cycle = sum(x["scoring_seconds"] for x in record["unique_clouds"].values())
            estimate = (4*record["raw_seconds"] + 8*record["inference_seconds"] +
                        80*(selection_cycle+score_cycle) + 8*record["reference_setup_seconds"])
            report.update(status="completed", scope="one-crop timing/execution check only; not the full comparison",
                          crop=[96, 3], inference_passes=1, optimizer_updates=0,
                          raw_source_seconds=record["raw_seconds"], inference_seconds=record["inference_seconds"],
                          selection_cycle_seconds=selection_cycle, scoring_cycle_seconds=score_cycle,
                          reference_setup_seconds=record["reference_setup_seconds"],
                          representative_compute_estimate_seconds=estimate,
                          estimate_note="One representative crop, one budget, one seed; different images/counts and plotting/startup add variability.")
        else:
            for record in records:
                inspect_record(record, measured, args.output_dir)
            report.update(status="completed", scope="eight fitted training crops; no generalization claim",
                          candidate_availability=candidates, rows=measured, summaries=summaries,
                          primary_guided_minus_coverage_pairs=pairs, inference_passes=len(records), optimizer_updates=0,
                          next_step="review all arms/chain gaps; no automatic sampler promotion or new training")
            with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                fields = ["sample_index", "patch_index", "group", "nominal_budget", "sampling_seed", "arm",
                          "actual_unique_count", "availability_only", "reference_coverage_2", "reference_proximity_6", "cloud_path", "details_path"]
                writer = csv.DictWriter(handle, fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(measured)
            for summary in summaries:
                if summary["group"] == "expanded" and summary["nominal_budget"] == 2048:
                    print(f"{summary['arm']}: {summary['neuron_macro_mean']}", flush=True)
        protected_unchanged(protected)
        report.update(elapsed_seconds=time.perf_counter()-started, scoring_seconds=scoring_seconds,
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else None,
                      retained_input_array_mib=sum(sum(r[k].nbytes for k in ("raw", "image", "valid", "probability")) for r in records)/2**20,
                      protected_files_sha256=protected, phase="completed",
                      checks={"sources_and_frozen_clouds_unchanged": True, "raw_images_and_intensity_restoration_exact": True,
                              "all_inputs_frozen_before_reference_scoring": True, "saved_field_metrics_reproduced": True,
                              "matched_unique_counts_and_zero_duplicates": True})
        write_json(report_path, report)
        print(f"Status: completed; execution checks passed; optimizer updates=0\nreport: {report_path.resolve()}", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pointneuron2/field_sampling_comparison.json")
    parser.add_argument("--partition", type=Path, default=ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())

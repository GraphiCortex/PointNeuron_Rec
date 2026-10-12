"""User-visible CPU native-image caching for the agreed 33/7 field pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pointneuron.data.centerline_field import SCHEMA as FIELD_SCHEMA, crop_volume, segment_distance_field, weak_labels
from pointneuron.data.field_pilot import CACHE_SCHEMA, SCHEMA, load_config, partition_contract, write_json
from pointneuron.data.query_cache import file_digest, load_query_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import volume_data_array
from pointneuron.data.vaa3d_raw import read_volume, VOLUME_READER_VERSION
from scripts.pointneuron2.compare_field_sampling import matching_source
from scripts.pointneuron2.field_sampling_metrics import reference_geometry
from scripts.pointneuron2.prepare_centerline_fit import inspection


def prepare(args):
    started = time.perf_counter()
    cfg = load_config(args.config)
    allowed = partition_contract(args.partition, expected_counts={"train": 33, "val": 7, "test": 8})
    source = load_query_manifest(args.manifest, args.partition)
    if any({(r["sample_index"], r["sample_id"]) for r in source["records"] if r["partition"] == role}
           != allowed[role] for role in ("train", "val")):
        raise ValueError("Source cache lacks one or more agreed train/development neurons")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh native cache directory")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    protected = {str(p.resolve()): file_digest(p) for p in (args.manifest, args.partition, args.config)}
    preparation_sources = [Path(__file__), ROOT / "src/pointneuron/data/centerline_field.py",
                           ROOT / "src/pointneuron/data/vaa3d_raw.py", ROOT / "src/pointneuron/data/curve_targets.py",
                           ROOT / "scripts/pointneuron2/field_sampling_metrics.py"]
    preparation_hashes = {str(p.resolve()): file_digest(p) for p in preparation_sources}
    protected.update(preparation_hashes)
    manifest = {"schema": CACHE_SCHEMA, "status": "running", "config": cfg,
                "config_sha256": file_digest(args.config), "partition_sha256": file_digest(args.partition),
                "source_manifest_sha256": file_digest(args.manifest), "volume_reader_version": VOLUME_READER_VERSION,
                "scope": "existing image-defined native crop boxes; train/development only; no whole-neuron claim",
                "preparation_source_files_sha256": preparation_hashes, "records": []}
    write_json(manifest_path, manifest)
    previous, volume, tree, first_crop = None, None, None, set()
    try:
        for number, row in enumerate(source["records"], 1):
            with np.load(row["resolved_path"], allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
            protected[row["resolved_path"]] = row["sha256"]
            directory = (args.data_root / row["sample_id"]).resolve()
            if not directory.is_relative_to(args.data_root.resolve()):
                raise ValueError("Dataset path escapes data root")
            if previous != row["sample_id"]:
                print(f"CPU decoding {number}/{len(source['records'])}: neuron {row['sample_index']} ({row['partition']})", flush=True)
                image_path = matching_source(directory, {".v3draw", ".v3dpbd"}, meta["volume_sha256"])
                swc_path = matching_source(directory, {".swc"}, meta["swc_sha256"])
                protected[str(image_path)] = meta["volume_sha256"]
                protected[str(swc_path)] = meta["swc_sha256"]
                decoded = read_volume(image_path)
                x, y, z, channels = decoded.dimensions
                if channels != 1:
                    raise ValueError("Only single-channel images are supported")
                volume = volume_data_array(decoded).reshape(z, y, x)
                tree = parse_swc(swc_path)
                previous = row["sample_id"]
                image_scale = float(volume.max())
                if image_scale <= 0:
                    raise ValueError("Empty source image")
            raw, valid, origin = crop_volume(volume, meta["patch_center"], meta["config"]["patch_radius"])
            if int(((raw > meta["threshold"]) & valid).sum()) != meta["foreground_count"]:
                raise ValueError("Native crop does not reproduce the original image-only foreground count")
            distance = segment_distance_field(tree, origin, raw.shape, cfg["uncertainty_distance_voxels"])
            target, known, unknown = weak_labels(raw, valid, distance, meta["threshold"],
                                                cfg["tube_radius_voxels"], cfg["uncertainty_distance_voxels"])
            geometry = reference_geometry(tree, origin, raw.shape)
            probes = [p["probe"]["points"] for p in geometry["chains"]]
            weights = [p["probe"]["weights"] for p in geometry["chains"]]
            reference = np.concatenate(probes) if probes else np.empty((0, 3), dtype=np.float64)
            reference_weights = np.concatenate(weights) if weights else np.empty(0, dtype=np.float64)
            image = raw.astype(np.float32)/image_scale
            record_meta = {**meta, "field_schema": FIELD_SCHEMA, "pilot_schema": SCHEMA,
                           "origin_xyz": origin.tolist(), "shape_zyx": list(raw.shape), "image_scale": image_scale,
                           "coordinate_units": "native acquisition voxels, unrescaled",
                           "annotation_policy": "full-segment tube-positive; distant bright/padding unknown; no local trace -> entirely unsupervised"}
            name = f"sample_{row['sample_index']:04d}_patch_{row['patch_index']:03d}"
            path = args.output_dir / (name + ".npz")
            np.savez_compressed(path, raw_image=raw, image=image, valid=valid, distance=distance, target=target,
                                known=known, unknown_bright=unknown, reference_local_xyz=reference,
                                reference_weights=reference_weights, metadata=np.array(json.dumps(record_meta)))
            supervised = bool((known & target).any() and (known & ~target).any())
            record_row = {k: row[k] for k in ("sample_index", "sample_id", "patch_index", "partition")}
            record_row.update(path=path.name, sha256=file_digest(path), supervised=supervised,
                              valid_voxels=int(valid.sum()), positive_voxels=int(target.sum()), known_voxels=int(known.sum()),
                              unknown_bright_voxels=int(unknown.sum()), reference_length_voxels=float(reference_weights.sum()),
                              volume_sha256=meta["volume_sha256"], swc_sha256=meta["swc_sha256"])
            manifest["records"].append(record_row)
            # One alignment overlay per neuron at preparation, all development crops after fitting.
            if row["sample_index"] not in first_crop:
                inspection(image, distance, valid, unknown, f"{row['partition']} neuron {row['sample_index']}, crop {row['patch_index']} | native coordinates",
                           args.output_dir / "inspection" / (name + ".png"))
                first_crop.add(row["sample_index"])
            write_json(manifest_path, manifest)
            print(f"prepared {number}/{len(source['records'])} neuron={row['sample_index']} crop={row['patch_index']} "
                  f"role={row['partition']} supervised={supervised} positives={target.sum()} "
                  f"reference_length={reference_weights.sum():.2f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        for path, digest in protected.items():
            if file_digest(path) != digest:
                raise AssertionError("Protected source changed during preparation")
        counts = {}
        for role in ("train", "val"):
            rows = [r for r in manifest["records"] if r["partition"] == role]
            per_neuron = Counter(r["sample_index"] for r in rows if r["supervised"])
            counts[role] = {"neurons": len({r["sample_index"] for r in rows}), "crops": len(rows),
                            "supervised_crops": sum(r["supervised"] for r in rows),
                            "unsupervised_crops": sum(not r["supervised"] for r in rows),
                            "neurons_without_supervised_crops": sorted({r["sample_index"] for r in rows} - set(per_neuron))}
            if not counts[role]["supervised_crops"]:
                raise ValueError(f"No usable {role} supervision")
        manifest.update(status="completed", counts=counts, protected_files_sha256=protected,
                        elapsed_seconds=time.perf_counter()-started, reserved_test_records=0)
        write_json(manifest_path, manifest)
        print(json.dumps(counts, indent=2), flush=True)
        print(f"Status: completed; CPU preparation only; no optimizer or model inference\nmanifest: {manifest_path.resolve()}", flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/pointneuron2/field_pilot.json")
    p.add_argument("--partition", type=Path, default=ROOT / "configs/pointneuron2/partition_seed0.json")
    prepare(p.parse_args())

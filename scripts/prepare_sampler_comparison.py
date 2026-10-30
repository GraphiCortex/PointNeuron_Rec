"""Freeze eight existing clouds into the audited PointNeXt input interface.

CPU only: verify original image bytes and attributes, preserve sampled IDs and
duplicates, change only point order, and save local XYZ plus XYZI attributes.
No reference is opened, no sampler/filter is rerun, and no torch is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pn2.inputs.cases import ImageCase
from pn2.inputs.image import decode_image
from pn2.evaluation.branch_evidence import source_priority_order


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked_json(relative, expected):
    path = ROOT / relative
    if digest(path) != expected:
        raise ValueError(f"Pinned input changed: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(config_path, output):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["schema_version"] != 1 or config["requested_slots"] != 2048
            or config["cloud_seed"] != 0 or config["coordinate_scale_voxel"] != 64.
            or config["ordering_seed"] != 0
            or config["methods"] != ["pn1_range_random", "pn2_sato"]):
        raise ValueError("This preparation supports only the declared two-arm input audit")
    prior = checked_json(config["comparison_report"], config["comparison_report_sha256"])
    inventory = checked_json(config["training_inventory"], config["training_inventory_sha256"])
    directory = Path(output).resolve()
    # The check above is read-only; an existing output is never overwritten.
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "config.json", config)
    manifest = {"schema_version": 1, "scope": config["scope"],
                "status": "inputs_verified_not_model_ready", "config_sha256": digest(config_path),
                "script_sha256": digest(__file__), "numpy": np.__version__, "records": []}
    for selected in config["cases"]:
        case = next(c for c in prior["cases"] if c["alias"] == selected["alias"])
        source = next(c for c in inventory["cases"] if c["case_id"] == case["case_id"])
        if source["partition"] != "training":
            raise ValueError("Only explicitly declared training images are allowed")
        # Only the image path is consumed; the inventory's reference field is not read.
        image_path = (ROOT / config["training_inventory"]).parent / source["image_path"]
        decoded = decode_image(ImageCase(case["case_id"], image_path, "training"))
        image = decoded.array_zyx
        if (decoded.provenance["source_sha256"] != case["source_sha256"]
                or decoded.provenance["decoded_sha256"] != case["verified_decode_sha256"]):
            raise ValueError("Original image or verified decode changed")
        minimum, maximum = float(image.min()), float(image.max())
        for tile_id in selected["tile_ids"]:
            box = next(b for b in case["boxes"] if b["tile_id"] == tile_id)
            low = np.asarray(box["tile"]["start_xyz"], dtype=np.float64)
            high = np.asarray(box["tile"]["stop_xyz"], dtype=np.float64)
            center = (low + high - 1.) / 2.
            for method in config["methods"]:
                arm = next(a for a in box["arms"] if a["method"] == method
                           and a["seed"] == config["cloud_seed"]
                           and a["requested_slots"] == config["requested_slots"])
                name = f"tile{tile_id}_{method}_slots2048_seed0.npz"
                original = (ROOT / config["comparison_report"]).parent / case["alias"] / name
                with np.load(original, allow_pickle=False) as packet:
                    ids = packet["flat_indices"].copy()
                    raw = packet["verified_intensity"].copy()
                    stored = packet["stored_intensity"].copy()
                if (ids.dtype != np.int64 or ids.shape != (2048,) or raw.shape != ids.shape
                        or not np.array_equal(raw, stored)
                        or np.any(ids < 0) or np.any(ids >= image.size)
                        or not np.array_equal(raw, image.ravel()[ids])):
                    raise ValueError(f"Point IDs or verified attributes disagree: {original}")
                unique = len(np.unique(ids))
                if (unique != arm["unique_count"] or len(ids) != arm["actual_slots"]
                        or len(ids) - unique != arm["duplicate_slots"]):
                    raise ValueError("Slot/distinct counts disagree with frozen comparison")
                order = source_priority_order(ids, config["ordering_seed"])
                ids, raw = ids[order], raw[order]
                xyz = np.asarray(np.unravel_index(ids, image.shape))[::-1].T.astype(np.float64)
                if not np.all((xyz >= low) & (xyz < high)):
                    raise ValueError("Selected source point escapes its actual clipped box")
                local = ((xyz - center) / config["coordinate_scale_voxel"]).astype(np.float32)
                intensity = (np.zeros_like(raw, dtype=np.float32) if maximum == minimum
                             else ((raw.astype(np.float64)-minimum)/(maximum-minimum)).astype(np.float32))
                attributes = np.column_stack((local, intensity)).astype(np.float32)
                if (not np.isfinite(attributes).all() or np.any(intensity < 0)
                        or np.any(intensity > 1) or not np.allclose(local * 64. + center, xyz, atol=1e-5)):
                    raise ValueError("Input normalization does not preserve source coordinates/range")
                target = directory / f"{case['alias']}_{name}"
                np.savez_compressed(target, flat_indices=ids, native_xyz=xyz,
                                    raw_intensity=raw, local_xyz=local, attributes=attributes)
                with np.load(target, allow_pickle=False) as replay:
                    if not all(np.array_equal(replay[key], value) for key, value in
                               (("flat_indices", ids), ("local_xyz", local), ("attributes", attributes))):
                        raise ValueError("Saved model inputs did not round-trip exactly")
                row = {"case_id": case["case_id"], "alias": case["alias"], "partition": "training",
                       "tile_id": tile_id, "method": method, "box": box["tile"],
                       "center_xyz": center.tolist(), "coordinate_scale_voxel": 64.,
                       "image_minimum": minimum, "image_maximum": maximum,
                       "source_sha256": case["source_sha256"],
                       "decoded_sha256": case["verified_decode_sha256"],
                       "source_packet": str(original.relative_to(ROOT)),
                       "source_packet_sha256": digest(original), "packet": target.name,
                       "packet_sha256": digest(target), "slots": len(ids), "distinct_points": unique,
                       "duplicate_slots": len(ids)-unique, "attributes_shape": list(attributes.shape)}
                manifest["records"].append(row)
                print(f"{case['alias']}/{tile_id} {method}: slots={len(ids)} distinct={unique} duplicates={len(ids)-unique}", flush=True)
    if len(manifest["records"]) != 8:
        raise ValueError("Expected exactly four paired boxes")
    write(directory / "inputs.json", manifest)
    print(f"CPU input preparation complete; no targets/model/training: {directory / 'inputs.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sampler_learnability_inputs.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)

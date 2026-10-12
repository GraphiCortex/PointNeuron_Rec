"""Verified native field caches, bounded CPU loading and reference-only metrics."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pointneuron.data.centerline_field import SCHEMA as FIELD_SCHEMA
from pointneuron.data.query_cache import file_digest
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION

SCHEMA = "pointneuron2-field-pilot-v1"
CACHE_SCHEMA = "pointneuron2-field-pilot-cache-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg.get("schema") != SCHEMA or cfg.get("batch_size") != 1 or cfg.get("augment") is not False
            or cfg.get("precision") != "float32" or cfg.get("field_threshold") != .5
            or cfg.get("negative_aggregation") != "topk" or cfg.get("negative_to_positive_ratio") != 3
            or cfg.get("tube_radius_voxels") != 2 or cfg.get("uncertainty_distance_voxels") != 6
            or cfg.get("epochs") != 30 or cfg.get("snapshot_epochs") != [0, 5, 15, 30]
            or cfg.get("checkpoint_for_sampling") != 30):
        raise ValueError("Settings differ from the agreed bounded field pilot")
    if (cfg["base_channels"] != 8 or cfg["learning_rate"] != .001 or cfg["weight_decay"] != .0001
            or cfg["gradient_clip"] != 10 or cfg["cpu_cache_records"] < 1 or cfg["torch_threads"] < 1):
        raise ValueError("Invalid or changed pilot model/optimization settings")
    return cfg


def partition_contract(path, *, expected_counts=None):
    partition = json.loads(Path(path).read_text())
    roles = partition["partitions"]
    if set(roles) != {"train", "val", "test"}:
        raise ValueError("Expected train/development/reserved test roles")
    allowed = {role: {(r["sample_index"], r["sample_id"]) for r in rows} for role, rows in roles.items()}
    for role, rows in roles.items():
        if (len(allowed[role]) != len(rows) or len({i for i, _ in allowed[role]}) != len(rows)
                or len({s for _, s in allowed[role]}) != len(rows)):
            raise ValueError("Duplicate partition identity")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ({i for i, _ in allowed[a]} & {i for i, _ in allowed[b]}
                or {s for _, s in allowed[a]} & {s for _, s in allowed[b]}):
            raise ValueError("Crossed neuron partitions")
    if expected_counts and any(len(allowed[k]) != n for k, n in expected_counts.items()):
        raise ValueError("Pilot requires the agreed 33/7/8 partition")
    return allowed


def load_manifest(path, partition_path, config_path):
    path = Path(path)
    cfg = load_config(config_path)
    allowed = partition_contract(partition_path)
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != CACHE_SCHEMA or manifest.get("status") != "completed"
            or manifest.get("volume_reader_version") != VOLUME_READER_VERSION
            or manifest.get("partition_sha256") != file_digest(partition_path)
            or manifest.get("config_sha256") != file_digest(config_path) or manifest.get("config") != cfg):
        raise ValueError("Native pilot cache/config/partition mismatch")
    paths, identities, included = set(), set(), {"train": set(), "val": set()}
    for row in manifest["records"]:
        role = row["partition"]
        identity = (row["sample_index"], row["sample_id"])
        crop = (*identity, row["patch_index"])
        if role not in included or identity not in allowed[role]:
            raise ValueError("Reserved/test or invalid cache identity")
        record_path = (path.parent / row["path"]).resolve()
        if (not record_path.is_relative_to(path.parent.resolve()) or record_path in paths or crop in identities
                or file_digest(record_path) != row["sha256"]):
            raise ValueError("Invalid, duplicate or changed native crop")
        paths.add(record_path)
        identities.add(crop)
        included[role].add(identity)
    if any(included[role] != allowed[role] for role in included):
        raise ValueError("Native cache does not cover every agreed train/development neuron")
    if not any(r["supervised"] and r["partition"] == "train" for r in manifest["records"]):
        raise ValueError("No usable training supervision")
    if not any(r["supervised"] and r["partition"] == "val" for r in manifest["records"]):
        raise ValueError("No usable development supervision")
    return manifest, cfg


def validate_record(record, row):
    meta = record["metadata"]
    if meta.get("field_schema") != FIELD_SCHEMA or meta.get("pilot_schema") != SCHEMA:
        raise ValueError("Unsupported native field record")
    if any(meta[k] != row[k] for k in ("sample_index", "sample_id", "patch_index", "partition")):
        raise ValueError("Record identity mismatch")
    image, raw = record["image"], record["raw_image"]
    shape = image.shape
    if (image.dtype != np.float32 or len(shape) != 3 or not np.isfinite(image).all()
            or image.min() < 0 or image.max() > 1 or raw.dtype not in (np.uint8, np.uint16)
            or raw.shape != shape or not np.array_equal(raw.astype(np.float32) / meta["image_scale"], image)):
        raise ValueError("Native raw intensity/normalization mismatch")
    for key in ("target", "known", "valid", "unknown_bright"):
        if record[key].dtype != np.bool_ or record[key].shape != shape:
            raise ValueError("Misaligned or invalid label mask")
    target, known, valid = (record[k] for k in ("target", "known", "valid"))
    distance = record["distance"]
    if (distance.shape != shape or np.isnan(distance).any() or (distance < 0).any()
            or np.any(known & ~valid) or np.any(known & record["unknown_bright"])
            or not np.array_equal(target, (distance <= 2) & valid)):
        raise ValueError("Invalid geometry or unknown/padding supervision")
    if not np.array_equal(record["unknown_bright"], valid & (raw > meta["threshold"]) & (distance > 6)):
        raise ValueError("Bright unannotated structures lost their uncertainty mask")
    has_trace = bool(np.any(valid & (distance <= .9)))
    expected_known = valid & ~record["unknown_bright"] if has_trace else np.zeros(shape, dtype=bool)
    if not np.array_equal(known, expected_known):
        raise ValueError("Crop without a trace must not become a background training example")
    supervised = bool((known & target).any() and (known & ~target).any())
    if supervised != row["supervised"] or int(target.sum()) != row["positive_voxels"]:
        raise ValueError("Supervision summary differs from native labels")
    reference, weights = record["reference_local_xyz"], record["reference_weights"]
    if (reference.ndim != 2 or reference.shape[1] != 3 or weights.shape != (len(reference),)
            or not np.isfinite(reference).all() or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Invalid reference probes")
    return record


class NativeRecords:
    """LRU CPU cache; at most one crop is transferred to CUDA by the runner."""
    def __init__(self, manifest_path, capacity=8):
        self.root = Path(manifest_path).parent.resolve()
        self.capacity = capacity
        self.cache = OrderedDict()
        self.peak_array_bytes = 0

    def get(self, row):
        key = row["path"]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or file_digest(path) != row["sha256"]:
            raise ValueError("Changed native record")
        with np.load(path, allow_pickle=False) as z:
            record = {k: z[k].copy() for k in z.files if k != "metadata"}
            record["metadata"] = json.loads(str(z["metadata"]))
        validate_record(record, row)
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = record
        self.peak_array_bytes = max(self.peak_array_bytes, sum(
            v.nbytes for r in self.cache.values() for v in r.values() if isinstance(v, np.ndarray)))
        return record


def field_metrics(probability, record, threshold=.5):
    """Arc-weighted continuous reference coverage; annotation agreement, not precision."""
    accepted = (probability >= threshold) & record["valid"]
    points = np.argwhere(accepted)[:, ::-1].astype(float)
    refs, weights = record["reference_local_xyz"], record["reference_weights"]
    weight = float(weights.sum())
    distances = cKDTree(points).query(refs, workers=1)[0] if len(points) and len(refs) else np.full(len(refs), np.inf)
    known, target = record["known"], record["known"] & record["target"]
    prediction_known = accepted & known
    has_supervision = bool(target.any() and (known & ~target).any())
    return {
        "reference_length_voxels": weight,
        "reference_coverage_2": float(weights[distances <= 2].sum()/weight) if weight else None,
        "reference_coverage_6": float(weights[distances <= 6].sum()/weight) if weight else None,
        "reference_proximity_6": float((record["distance"][accepted] <= 6).mean()) if accepted.any() and weight else (0. if weight else None),
        "known_voxel_dice": float(2*(prediction_known & target).sum()/max(1, prediction_known.sum()+target.sum())) if has_supervision else None,
        "accepted_voxels": int(accepted.sum()),
        "accepted_fraction_valid": float(accepted.sum()/max(1, record["valid"].sum())),
        "accepted_zero_intensity_voxels": int((accepted & (record["raw_image"] == 0)).sum()),
        "accepted_unknown_bright_voxels": int((accepted & record["unknown_bright"]).sum()),
        "supervised": has_supervision,
    }


def macro_metrics(rows):
    keys = ("reference_coverage_2", "reference_coverage_6", "reference_proximity_6", "known_voxel_dice")
    neurons = []
    for neuron in sorted({r["sample_index"] for r in rows}):
        relevant = [r for r in rows if r["sample_index"] == neuron]
        item = {"sample_index": neuron, "crop_count": len(relevant), "supervised_crops": sum(r["supervised"] for r in relevant)}
        for key in keys:
            values = [r[key] for r in relevant if r[key] is not None]
            item[key] = float(np.mean(values)) if values else None
        neurons.append(item)
    macro = {}
    for key in keys:
        values = [r[key] for r in neurons if r[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    return {"neuron_macro_mean": macro, "per_neuron": neurons,
            "crops_without_supervision": sum(not r["supervised"] for r in rows),
            "crops_without_reference_length": sum(r["reference_coverage_2"] is None for r in rows)}

"""Immutable derived target caches for the paired query/pointwise comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pointneuron.data.curve_targets import TARGET_SCHEMA
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.data.torch_dataset import collate_training_records

CACHE_SCHEMA = "pointneuron2-query-cache-v1"


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_query_manifest(path, partition_path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    partition = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError("Incomplete or unsupported query target cache")
    if manifest.get("target_schema") != TARGET_SCHEMA or manifest.get("volume_reader_version") != VOLUME_READER_VERSION:
        raise ValueError("Query target/reader contract mismatch")
    if manifest["partition_sha256"] != file_digest(partition_path):
        raise ValueError("Partition hash mismatch")
    allowed = {role: {(r["sample_index"], r["sample_id"]) for r in rows}
               for role, rows in partition["partitions"].items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ({i for i, _ in allowed[a]} & {i for i, _ in allowed[b]} or
                {s for _, s in allowed[a]} & {s for _, s in allowed[b]}):
            raise ValueError("Crossed partitions")
    seen = set()
    for row in manifest["records"]:
        role = row["partition"]
        if role not in ("train", "val") or (row["sample_index"], row["sample_id"]) not in allowed[role]:
            raise ValueError("Reserved/test or invalid neuron identity")
        record = (path.parent / row["path"]).resolve()
        if not record.is_relative_to(path.parent.resolve()) or record in seen:
            raise ValueError("Invalid or duplicate cache path")
        seen.add(record)
        if file_digest(record) != row["sha256"]:
            raise ValueError("Query cache content hash mismatch")
        with np.load(record, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            for key in ("sample_index", "sample_id", "partition", "patch_index"):
                if metadata[key] != row[key]:
                    raise ValueError("Record identity mismatch")
            if metadata["target_schema"] != TARGET_SCHEMA or metadata["target_spacing"] != manifest["config"]["target_spacing"]:
                raise ValueError("Record target settings mismatch")
            if metadata["target_capacity"] != manifest["config"]["target_capacity"] or metadata["config"] != manifest["input_config"]:
                raise ValueError("Record input/capacity settings mismatch")
            nodes, points, curve = z["target_nodes"], z["points"], z["reference_curve"]
            if nodes.shape != (row["target_count"], 6) or len(nodes) > manifest["config"]["target_capacity"]:
                raise ValueError("Target shape/capacity mismatch")
            if points.shape != (manifest["input_config"]["candidate_points"], 4) or curve.ndim != 2 or curve.shape[1] != 3:
                raise ValueError("Input/reference shape mismatch")
            if not all(np.isfinite(t).all() for t in (nodes, points, curve)):
                raise ValueError("Nonfinite cached geometry")
            if np.any(nodes[:, 4] < 0) or z["target_roles"].shape != (len(nodes),):
                raise ValueError("Invalid radii or role flags")
        row["resolved_path"] = str(record)
    return manifest


class QueryCacheDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with np.load(row["resolved_path"], allow_pickle=False) as z:
            return {"points": torch.from_numpy(z["points"].astype(np.float32)),
                    "skeleton_nodes": torch.from_numpy(z["target_nodes"].astype(np.float32)),
                    "edge_index": torch.empty(0, 2, dtype=torch.long),
                    "reference_curve": z["reference_curve"].copy(),
                    "metadata": json.loads(str(z["metadata"])), "path": row["resolved_path"]}


def collate_query_records(records):
    result = collate_training_records(records)
    result["patch_centers"] = torch.tensor([r["metadata"]["patch_center"] for r in records], dtype=torch.float32)
    result["reference_curves"] = [r["reference_curve"] for r in records]
    return result

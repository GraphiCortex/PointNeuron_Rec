"""Bounded image candidates and optionally full guides for training/inference."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from pointneuron.data.medial_sampling import MedialSamplingConfig, image_medial_axis
from pointneuron.data.training_cache import indices_to_point_array
from pointneuron.data.torch_dataset import TrainingCacheDataset, collate_training_records
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION


@dataclass(frozen=True)
class SamplingPoolConfig:
    candidate_points: int = 1024
    guide_points: int | None = None

    def __post_init__(self):
        if isinstance(self.candidate_points, bool) or not isinstance(self.candidate_points, int) or self.candidate_points < 1:
            raise ValueError('Candidate pool size must be a positive integer')
        if self.guide_points is not None and (isinstance(self.guide_points, bool) or
                not isinstance(self.guide_points, int) or self.guide_points < 1):
            raise ValueError('Guide size must be a positive integer or None for the full guide')


def build_image_pools(indices: np.ndarray, data: np.ndarray, width: int, height: int,
                      pools: SamplingPoolConfig, medial: MedialSamplingConfig, seed: int) -> dict:
    """Extract from all patch foreground; bound only the network inputs afterward.

    SWCs are deliberately absent from this API. Candidate selection is uniform
    and independent of guide extraction. Undersized pools preserve all unique
    image voxels before padding; row indices can then represent repeated voxels.
    """
    if not len(indices):
        raise ValueError("Cannot build pools from empty foreground")
    rng = np.random.default_rng(seed)
    if len(indices) >= pools.candidate_points:
        candidates = np.sort(rng.choice(indices, pools.candidate_points, replace=False))
    else:
        candidates = np.sort(np.concatenate((indices, rng.choice(indices, pools.candidate_points - len(indices)))))
    guide = image_medial_axis(indices, width, height, medial)
    full_guide_count = len(guide)
    if pools.guide_points is not None and len(guide) > pools.guide_points:
        guide = guide[np.sort(rng.choice(len(guide), pools.guide_points, replace=False))]
    return {"points": indices_to_point_array(candidates, data, width, height),
            "candidate_indices": candidates, "image_guide_xyz": guide.astype(np.float32),
            "foreground_count": len(indices), "full_guide_count": full_guide_count}


class SamplingCacheDataset(TrainingCacheDataset):
    def __getitem__(self, index):
        record = super().__getitem__(index)
        import torch
        with np.load(self.record_paths[index], allow_pickle=False) as cached:
            if record["metadata"].get("sampling_cache_schema") not in (1, 2):
                raise ValueError("Expected PN 2.0 sampling cache schema 1 or 2")
            guide = cached["image_guide_xyz"].astype(np.float32)
            if guide.ndim != 2 or guide.shape[1] != 3 or not np.isfinite(guide).all():
                raise ValueError("Invalid image guide")
            if not np.isfinite(record["points"].numpy()).all():
                raise ValueError("Nonfinite candidate points")
            record["image_guide_xyz"] = torch.from_numpy(guide)
        return record


def collate_sampling_records(records):
    import torch
    batch = collate_training_records(records)
    max_guide = max(len(record["image_guide_xyz"]) for record in records)
    guide = torch.zeros(len(records), max_guide, 3)
    mask = torch.zeros(len(records), max_guide, dtype=torch.bool)
    for b, record in enumerate(records):
        n = len(record["image_guide_xyz"])
        guide[b, :n] = record["image_guide_xyz"]
        mask[b, :n] = True
    batch.update(image_guide_xyz=guide, image_guide_mask=mask)
    return batch


def load_sampling_manifest(path: str | Path, partition_path: str | Path, *, allow_historical_reader: bool = False) -> dict:
    """Check saved partition, file hashes, and actual NPZ identities before training."""
    import hashlib
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not allow_historical_reader and manifest.get('volume_reader_version') != VOLUME_READER_VERSION:
        raise ValueError('Cache predates the corrected Vaa3D decoder. Rebuild the image cache before training; '
                         'allow_historical_reader is only for read-only analysis of saved runs.')
    pools = SamplingPoolConfig(manifest['config']['candidate_points'], manifest['config']['guide_points'])
    expected_schema = 2 if pools.guide_points is None else 1
    partition = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    if manifest.get("partition_sha256") != hashlib.sha256(Path(partition_path).read_bytes()).hexdigest():
        raise ValueError("Partition hash mismatch")
    allowed = {role: {(row["sample_index"], row["sample_id"]) for row in rows}
               for role, rows in partition["partitions"].items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ({idx for idx, _ in allowed[a]} & {idx for idx, _ in allowed[b]} or
                {identity for _, identity in allowed[a]} & {identity for _, identity in allowed[b]}):
            raise ValueError("Crossed partition identities")
    seen = set()
    for row in manifest["records"]:
        role = row["partition"]
        if role not in ("train", "val") or (row["sample_index"], row["sample_id"]) not in allowed[role]:
            raise ValueError("Cache contains a reserved/test or misassigned neuron")
        record_path = (Path(path).parent / row["path"]).resolve()
        if record_path in seen:
            raise ValueError("Duplicate cache path")
        seen.add(record_path)
        if hashlib.sha256(record_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Cache content hash mismatch")
        with np.load(record_path, allow_pickle=False) as record:
            metadata = json.loads(str(record["metadata"]))
            if not allow_historical_reader and metadata.get('volume_reader_version') != VOLUME_READER_VERSION:
                raise ValueError('Cache record has an incompatible Vaa3D decoder version')
            for key in ("sample_id", "sample_index", "partition"):
                if metadata[key] != row[key]:
                    raise ValueError("Cache identity disagrees with manifest")
            if metadata["sampling_cache_schema"] != expected_schema or metadata["config"] != manifest["config"]:
                raise ValueError("Cache schema/config mismatch")
            if metadata["medial_config"] != manifest["medial_config"]:
                raise ValueError("Medial config mismatch")
            if record["points"].shape != (manifest["config"]["candidate_points"], 4):
                raise ValueError("Candidate pool shape mismatch")
            if pools.guide_points is not None and len(record["image_guide_xyz"]) > pools.guide_points:
                raise ValueError("Guide cap exceeded")
            if pools.guide_points is None and len(record['image_guide_xyz']) != metadata.get('full_guide_count'):
                raise ValueError('Full-guide cache discarded vertices')
        row["resolved_path"] = str(record_path)
    return manifest

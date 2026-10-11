from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
import re
from pathlib import Path


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.train, self.val, self.test)):
            raise ValueError("Split ratios must be finite")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("Split ratios must all be positive")


def split_records(record_paths: list[str], ratios: SplitRatios = SplitRatios(), seed: int = 0) -> dict[str, list[str]]:
    ratios.validate()
    shuffled = list(record_paths)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * ratios.train)
    val_count = int(total * ratios.val)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count == 0:
            test_count = 1
            train_count = max(1, train_count - 1)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def record_paths_from_cache_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [record["path"] for record in payload["records"]]


def neuron_identity(sample_id: str) -> str:
    """Normalize the patch suffix emitted by TrainingCacheRecord.sample_id."""
    return re.sub(r"#patch-\d+$", "", sample_id)


def write_split_file(
    cache_manifest: str | Path,
    output_path: str | Path,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 0,
    group_by: str = "sample_id",
    record_root: str | Path | None = None,
    dataset_id: str | None = None,
) -> dict[str, list[str]]:
    """Split neurons (or explicit specimen groups), never individual patches by default.

    Relative record paths retain the historical caller-CWD convention unless
    record_root is supplied. This permits worktrees to read existing caches.
    group_by='record' is an explicit compatibility mode for historical experiments.
    """
    records = json.loads(Path(cache_manifest).read_text(encoding="utf-8"))["records"]
    if not records:
        raise ValueError("Cannot split an empty cache manifest")
    metadata = []
    seen_paths: set[str] = set()
    neuron_groups: dict[str, str] = {}
    index_identities: dict[str, str] = {}
    for record in records:
        path = Path(record["path"])
        if record_root is not None and not path.is_absolute():
            path = Path(record_root).resolve() / path
        path_string = str(path)
        # Windows paths are case insensitive; resolve equivalent relative spellings.
        path_key = os.path.normcase(str(path.resolve()))
        if path_key in seen_paths:
            raise ValueError(f"Duplicate record path: {path_string}")
        seen_paths.add(path_key)
        source_sample_id = record.get("sample_id")
        sample_id = neuron_identity(str(source_sample_id)) if source_sample_id is not None else None
        if group_by == "record":
            group = path_string
        else:
            if sample_id is None or not str(sample_id).strip():
                raise ValueError(f"Missing sample_id for {path_string}; neuron identity is required")
            value = record.get(group_by)
            if value is None or not str(value).strip() or isinstance(value, (list, dict, bool)):
                raise ValueError(f"Missing or invalid {group_by} for {path_string}")
            group = sample_id if group_by == "sample_id" else str(value)
            if record.get("sample_index") is not None:
                index = str(record["sample_index"])
                previous_id = index_identities.setdefault(index, sample_id)
                if previous_id != sample_id:
                    raise ValueError(f"Sample index {index} maps to conflicting neuron identities")
            previous = neuron_groups.setdefault(str(sample_id), group)
            if previous != group:
                raise ValueError(f"Neuron {sample_id!r} belongs to conflicting {group_by} groups")
        metadata.append({"path": path_string, "sample_id": sample_id, "source_sample_id": source_sample_id, "sample_index": record.get("sample_index"), "group": group})
    groups = sorted({record["group"] for record in metadata})
    if group_by != "record" and len(groups) < 3:
        raise ValueError("At least three distinct groups are required for train/val/test")
    group_splits = split_records(
        [record["group"] for record in metadata] if group_by == "record" else groups,
        ratios=ratios, seed=seed,
    )
    assignments = {group: split for split, values in group_splits.items() for group in values}
    splits = {split: [] for split in group_splits}
    for record in metadata:
        splits[assignments[record["group"]]].append(record["path"])
    if group_by == "record":
        splits = split_records([record["path"] for record in metadata], ratios=ratios, seed=seed)
    payload = {
        "schema_version": 3,
        "identity_rule": "strip_legacy_patch_suffix_v1",
        "cache_manifest": str(Path(cache_manifest).resolve()),
        "dataset_id": dataset_id,
        "group_by": group_by,
        "split_unit": "record" if group_by == "record" else "neuron_group",
        "record_root": str(Path(record_root).resolve()) if record_root is not None else None,
        "seed": seed,
        "ratios": {
            "train": ratios.train,
            "val": ratios.val,
            "test": ratios.test,
        },
        "counts": {key: len(value) for key, value in splits.items()},
        "splits": splits,
        "group_splits": group_splits,
        "group_counts": {key: len(value) for key, value in group_splits.items()},
        "record_metadata": metadata,
    }
    validate_split_payload(payload, require_neuron_split=group_by != "record")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return splits


def validate_split_payload(payload: dict, require_neuron_split: bool = False) -> None:
    """Reject duplicated records, crossed neurons/groups, and stale metadata."""
    splits = payload.get("splits", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits")
    owners: dict[str, str] = {}
    for split, paths in splits.items():
        for path in paths:
            key = os.path.normcase(str(Path(path).resolve()))
            if key in owners:
                raise ValueError(f"Record appears more than once: {path}")
            owners[key] = split
    if require_neuron_split and payload.get("split_unit") != "neuron_group":
        raise ValueError("A verified neuron-group split is required; rebuild with --group-by sample_id")
    if payload.get("split_unit") != "neuron_group":
        return
    if payload.get("identity_rule") != "strip_legacy_patch_suffix_v1":
        raise ValueError("Neuron identities must be normalized; rebuild this split")
    group_owners: dict[str, str] = {}
    for split, groups in payload.get("group_splits", {}).items():
        if split not in splits:
            raise ValueError(f"Unknown group split {split!r}")
        for group in groups:
            if group in group_owners:
                raise ValueError(f"Group appears more than once: {group}")
            group_owners[group] = split
    metadata_owners: set[str] = set()
    neuron_owners: dict[str, str] = {}
    index_owners: dict[str, str] = {}
    used_groups: set[str] = set()
    for record in payload.get("record_metadata", []):
        key = os.path.normcase(str(Path(record["path"]).resolve()))
        if key not in owners or key in metadata_owners:
            raise ValueError("Record metadata does not match split paths")
        metadata_owners.add(key)
        split = owners[key]
        group = record["group"]
        if group_owners.get(group) != split:
            raise ValueError(f"Group assignment disagrees with record: {record['path']}")
        used_groups.add(group)
        sample_id = record.get("sample_id")
        if sample_id is None or not str(sample_id).strip():
            raise ValueError("Missing neuron identity in split metadata")
        if neuron_identity(str(sample_id)) != sample_id:
            raise ValueError("Patch identity was used instead of neuron identity")
        previous = neuron_owners.setdefault(str(sample_id), split)
        if previous != split:
            raise ValueError(f"Neuron {sample_id!r} crosses splits")
        if record.get("sample_index") is not None:
            index = str(record["sample_index"])
            previous = index_owners.setdefault(index, split)
            if previous != split:
                raise ValueError(f"Sample index {index} crosses splits")
    if metadata_owners != set(owners) or used_groups != set(group_owners):
        raise ValueError("Incomplete record or group metadata")
    if any(not paths for paths in splits.values()):
        raise ValueError("Neuron-group splits must all contain records")


def validate_split_file(path: str | Path, require_neuron_split: bool = False) -> None:
    validate_split_payload(json.loads(Path(path).read_text(encoding="utf-8")), require_neuron_split)


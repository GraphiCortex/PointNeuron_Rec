from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
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


def write_split_file(
    cache_manifest: str | Path,
    output_path: str | Path,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 0,
) -> dict[str, list[str]]:
    record_paths = record_paths_from_cache_manifest(cache_manifest)
    splits = split_records(record_paths, ratios=ratios, seed=seed)
    payload = {
        "cache_manifest": str(cache_manifest),
        "seed": seed,
        "ratios": {
            "train": ratios.train,
            "val": ratios.val,
            "test": ratios.test,
        },
        "counts": {key: len(value) for key, value in splits.items()},
        "splits": splits,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return splits


from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.splits import SplitRatios, write_split_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic train/val/test split from a cache manifest.")
    parser.add_argument("--cache-manifest", required=True, help="Path to cache_manifest.json.")
    parser.add_argument("--output", default="tmp/splits/gold166_clean_seed0.json", help="Output split JSON path.")
    parser.add_argument("--seed", type=int, default=0, help="Random split seed.")
    parser.add_argument("--group-by", default="sample_id", help="Manifest field used to keep neurons/specimens together; 'record' explicitly reproduces legacy patch splits.")
    parser.add_argument("--record-root", help="Root for manifest-relative record paths, e.g. the original checkout when using shared caches.")
    parser.add_argument("--dataset-id", help="Dataset identity saved with the split.")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio.")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio.")
    args = parser.parse_args()

    splits = write_split_file(
        args.cache_manifest,
        args.output,
        ratios=SplitRatios(train=args.train, val=args.val, test=args.test),
        seed=args.seed,
        group_by=args.group_by,
        record_root=args.record_root,
        dataset_id=args.dataset_id,
    )
    print(f"split_file: {args.output}")
    for name, paths in splits.items():
        print(f"{name}: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.torch_dataset import TrainingCacheDataset, collate_training_records, load_split_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch dataset batch from cached training records.")
    parser.add_argument("--split-file", required=True, help="Path to split JSON.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to inspect.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of records to collate.")
    args = parser.parse_args()

    paths = load_split_paths(args.split_file, args.split)
    dataset = TrainingCacheDataset(paths)
    try:
        batch = collate_training_records([dataset[index] for index in range(min(args.batch_size, len(dataset)))])
    except ModuleNotFoundError as exc:
        print(str(exc))
        return 2

    print(f"records: {len(dataset)}")
    print(f"points: {tuple(batch['points'].shape)} {batch['points'].dtype}")
    print(f"skeleton_nodes: {tuple(batch['skeleton_nodes'].shape)} {batch['skeleton_nodes'].dtype}")
    print(f"skeleton_mask: {tuple(batch['skeleton_mask'].shape)} true={int(batch['skeleton_mask'].sum().item())}")
    print(f"edge_index: {tuple(batch['edge_index'].shape)} {batch['edge_index'].dtype}")
    print(f"edge_mask: {tuple(batch['edge_mask'].shape)} true={int(batch['edge_mask'].sum().item())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

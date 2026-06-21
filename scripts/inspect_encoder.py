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
    parser = argparse.ArgumentParser(description="Run a shape check through the DGCNN/EdgeConv encoder.")
    parser.add_argument("--split-file", required=True, help="Path to split JSON.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to inspect.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of records to collate.")
    parser.add_argument("--k", type=int, default=20, help="kNN neighbors for EdgeConv.")
    parser.add_argument("--feature-dim", type=int, help="Optional projected output feature dimension.")
    parser.add_argument("--proposal", action="store_true", help="Also run the skeleton proposal head.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before running the encoder.")
        return 2
    from pointneuron.models.dgcnn import DGCNNEncoder
    from pointneuron.models.proposal import SkeletonProposalHead

    paths = load_split_paths(args.split_file, args.split)
    dataset = TrainingCacheDataset(paths)
    batch_size = min(args.batch_size, len(dataset))
    batch = collate_training_records([dataset[index] for index in range(batch_size)])

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    points = batch["points"].to(device)
    model = DGCNNEncoder(k=args.k, feature_dim=args.feature_dim).to(device)
    model.eval()
    with torch.no_grad():
        features = model(points)

    print(f"device: {device}")
    print(f"input_points: {tuple(points.shape)} {points.dtype}")
    print(f"geometric_feature_dim: {model.geometric_feature_dim}")
    print(f"encoded_features: {tuple(features.shape)} {features.dtype}")
    if args.proposal:
        proposal = SkeletonProposalHead(in_channels=features.shape[-1] + 3).to(device)
        proposal.eval()
        with torch.no_grad():
            output = proposal(points, features)
        print(f"proposal_offsets: {tuple(output.offsets.shape)} {output.offsets.dtype}")
        print(f"proposal_objectness_logits: {tuple(output.objectness_logits.shape)} {output.objectness_logits.dtype}")
        print(f"proposal_radius: {tuple(output.radius.shape)} {output.radius.dtype}")
        print(f"proposal_centers: {tuple(output.center_proposals.shape)} {output.center_proposals.dtype}")
    if device == "cuda":
        print(f"cuda_memory_allocated_mb: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}")
        print(f"cuda_memory_reserved_mb: {torch.cuda.memory_reserved() / 1024 / 1024:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

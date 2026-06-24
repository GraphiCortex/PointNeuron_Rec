from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from pointneuron.data.torch_dataset import TrainingCacheDataset, collate_training_records
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from pointneuron.models.proposal_loss import nearest_distances


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether a proposal checkpoint improves cached foreground samples.")
    parser.add_argument("--split-file", required=True, help="Training split JSON produced by build_split.py.")
    parser.add_argument("--checkpoint", required=True, help="Proposal checkpoint to audit.")
    parser.add_argument("--splits", nargs="+", default=["val"], help="Split names to sample from.")
    parser.add_argument("--records-per-split", type=int, default=40, help="Maximum records to audit from each split.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for the forward pass.")
    parser.add_argument("--hit-distance", type=float, default=8.0, help="Distance counted as a center hit.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required for checkpoint auditing.")
        return 2

    payload = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    record_paths: list[str] = []
    for split_name in args.splits:
        split_paths = payload["splits"].get(split_name)
        if split_paths is None:
            raise ValueError(f"Unknown split {split_name!r}; expected one of {sorted(payload['splits'])}")
        record_paths.extend(split_paths[: args.records_per_split])
    if not record_paths:
        raise ValueError("No records selected for audit.")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    k = int(checkpoint_args.get("k", 20))
    coordinate_mode = checkpoint_args.get("proposal_coordinate_mode", "raw")

    encoder = DGCNNEncoder(k=k).to(device)
    proposal_in_channels = int(checkpoint["proposal"]["mlp.0.weight"].shape[1])
    proposal = SkeletonProposalHead(
        in_channels=proposal_in_channels,
        include_xyz=(proposal_in_channels == encoder.output_dim + 3),
        coordinate_mode=coordinate_mode,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    proposal.load_state_dict(checkpoint["proposal"])
    encoder.eval()
    proposal.eval()

    rows: list[dict[str, float | str | int]] = []
    dataset = TrainingCacheDataset(record_paths)
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            batch = collate_training_records([item])
            points = batch["points"].to(device)
            skeleton_nodes = batch["skeleton_nodes"].to(device)
            skeleton_mask = batch["skeleton_mask"].to(device)
            features = encoder(points)
            output = proposal(points, features)

            gt_xyz = skeleton_nodes[0, skeleton_mask[0], 1:4]
            input_distance, _ = nearest_distances(points[0, :, :3], gt_xyz)
            proposal_distance, _ = nearest_distances(output.center_proposals[0], gt_xyz)
            scores = output.objectness_logits.softmax(dim=-1)[0, :, 1]
            radii = output.radius[0, :, 0]
            offset_norm = output.offsets[0].norm(dim=1)
            metadata = item["metadata"]

            rows.append(
                {
                    "domain": domain_name(str(metadata.get("sample_id", ""))),
                    "sample_index": int(metadata.get("sample_index", -1)),
                    "input_hit": float((input_distance <= args.hit_distance).float().mean().item()),
                    "proposal_hit": float((proposal_distance <= args.hit_distance).float().mean().item()),
                    "input_distance": float(input_distance.mean().item()),
                    "proposal_distance": float(proposal_distance.mean().item()),
                    "score_mean": float(scores.mean().item()),
                    "score_p90": float(scores.quantile(0.90).item()),
                    "score_p99": float(scores.quantile(0.99).item()),
                    "score_ge_04": float((scores >= 0.4).float().mean().item()),
                    "radius_mean": float(radii.mean().item()),
                    "radius_std": float(radii.std().item()),
                    "offset_mean": float(offset_norm.mean().item()),
                    "offset_p90": float(offset_norm.quantile(0.90).item()),
                }
            )

    print(f"checkpoint: {args.checkpoint}")
    print(f"coordinate_mode: {coordinate_mode}")
    print(f"loss_mode: {checkpoint_args.get('loss_mode')}")
    print(f"records: {len(rows)}")
    print(f"hit_distance: {args.hit_distance:g}")
    print()
    print_group("all", rows)
    for name in sorted({str(row["domain"]) for row in rows}):
        print_group(name, [row for row in rows if row["domain"] == name])

    worst = sorted(rows, key=lambda row: float(row["proposal_hit"]))[:10]
    print("worst_records:")
    for row in worst:
        print(
            f"  sample {row['sample_index']}: domain={row['domain']} "
            f"input_hit={float(row['input_hit']):.4f} proposal_hit={float(row['proposal_hit']):.4f} "
            f"input_dist={float(row['input_distance']):.2f} proposal_dist={float(row['proposal_distance']):.2f} "
            f"score_p90={float(row['score_p90']):.3f} offset={float(row['offset_mean']):.2f}"
        )

    return 0


def domain_name(sample_id: str) -> str:
    lower = sample_id.lower()
    if "janelia" in lower or "flylight" in lower:
        return "janelia_like"
    return "other_gold"


def print_group(name: str, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    print(f"{name}: n={len(rows)}")
    for key in [
        "input_hit",
        "proposal_hit",
        "input_distance",
        "proposal_distance",
        "score_mean",
        "score_p90",
        "score_p99",
        "score_ge_04",
        "radius_mean",
        "radius_std",
        "offset_mean",
        "offset_p90",
    ]:
        print(f"  mean_{key}: {mean(rows, key):.4f}")
    print()


def mean(rows: list[dict[str, float | str | int]], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return statistics.fmean(values) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())

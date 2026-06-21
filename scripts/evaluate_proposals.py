from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pointneuron.data.torch_dataset import TrainingCacheDataset, collate_training_records, load_split_paths
from scripts.visualize_proposals import select_proposals


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained skeleton proposal predictions against SWC labels.")
    parser.add_argument("--split-file", required=True, help="Path to split JSON.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to evaluate.")
    parser.add_argument("--checkpoint", required=True, help="Path to proposal checkpoint.")
    parser.add_argument("--k", type=int, help="Override kNN neighbors. Defaults to checkpoint value or 20.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum objectness probability before NMS.")
    parser.add_argument("--min-candidates", type=int, default=0, help="Lower the score threshold per record to include at least this many candidates before NMS.")
    parser.add_argument("--candidate-score-floor", type=float, default=0.0, help="Minimum score allowed when --min-candidates lowers the threshold.")
    parser.add_argument("--top-proposals", type=int, default=512, help="Maximum selected proposals per sample.")
    parser.add_argument("--nms-mode", default="sphere", choices=["sphere", "distance"], help="Proposal downsampling mode.")
    parser.add_argument("--nms-radius", type=float, default=8.0, help="Spherical NMS radius in voxels.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Sphere IoU threshold for spherical NMS.")
    parser.add_argument("--match-distance", type=float, default=6.0, help="Proposal-to-SWC distance counted as a hit.")
    parser.add_argument("--coverage-distance", type=float, default=8.0, help="SWC node distance counted as covered.")
    parser.add_argument("--max-records", type=int, help="Optional limit for a quick check.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N records. Set 0 to disable.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before evaluating proposals.")
        return 2
    from pointneuron.models.dgcnn import DGCNNEncoder
    from pointneuron.models.proposal import SkeletonProposalHead

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is false.")
        return 2

    paths = load_split_paths(args.split_file, args.split)
    if args.max_records is not None:
        paths = paths[: args.max_records]
    if not paths:
        print(f"No records found for split {args.split!r}.")
        return 2

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    k = args.k if args.k is not None else int(checkpoint_args.get("k", 20))

    encoder = DGCNNEncoder(k=k).to(device)
    proposal_in_channels = int(checkpoint["proposal"]["mlp.0.weight"].shape[1])
    proposal = SkeletonProposalHead(
        in_channels=proposal_in_channels,
        include_xyz=(proposal_in_channels == encoder.output_dim + 3),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    proposal.load_state_dict(checkpoint["proposal"])
    encoder.eval()
    proposal.eval()

    rows = []
    total_selected = 0
    total_hits = 0
    total_nodes = 0
    total_covered = 0
    distance_means = []
    coverage_rates = []
    score_quantile_rows = []

    with torch.no_grad():
        for record_index, path in enumerate(paths):
            dataset = TrainingCacheDataset([path])
            batch = collate_training_records([dataset[0]])
            points = batch["points"].to(device)
            skeleton_nodes = batch["skeleton_nodes"].to(device)
            skeleton_mask = batch["skeleton_mask"].to(device)
            metadata = batch["metadata"][0]

            features = encoder(points)
            output = proposal(points, features)
            scores = output.objectness_logits.softmax(dim=-1)[0, :, 1]
            centers = output.center_proposals[0]
            radii = output.radius[0, :, 0]
            valid_skeleton = skeleton_nodes[0, skeleton_mask[0]][:, 1:4]
            score_quantile_rows.append(scores.detach().float())
            score_quantiles = scores.detach().float().quantile(
                torch.tensor([0.5, 0.9, 0.99, 1.0], device=scores.device)
            )
            above_threshold = int((scores >= args.score_threshold).sum().item())

            selected_indices = select_proposals(
                centers=centers,
                radii=radii,
                scores=scores,
                score_threshold=args.score_threshold,
                min_candidates=args.min_candidates,
                candidate_score_floor=args.candidate_score_floor,
                top_proposals=args.top_proposals,
                nms_mode=args.nms_mode,
                nms_radius=args.nms_radius,
                iou_threshold=args.iou_threshold,
            )

            selected_centers = centers[selected_indices] if selected_indices else centers.new_zeros((0, 3))
            if selected_centers.shape[0] > 0:
                selected_to_swc = torch.cdist(selected_centers, valid_skeleton).min(dim=1).values
                swc_to_selected = torch.cdist(valid_skeleton, selected_centers).min(dim=1).values
                hits = int((selected_to_swc <= args.match_distance).sum().item())
                covered = int((swc_to_selected <= args.coverage_distance).sum().item())
                mean_distance = float(selected_to_swc.mean().item())
                median_distance = float(selected_to_swc.median().item())
            else:
                hits = 0
                covered = 0
                mean_distance = float("inf")
                median_distance = float("inf")

            selected_count = len(selected_indices)
            node_count = int(valid_skeleton.shape[0])
            precision = hits / selected_count if selected_count else 0.0
            coverage = covered / node_count if node_count else 0.0

            total_selected += selected_count
            total_hits += hits
            total_nodes += node_count
            total_covered += covered
            if selected_count:
                distance_means.append(mean_distance)
            coverage_rates.append(coverage)

            rows.append(
                (
                    coverage,
                    precision,
                    record_index,
                    metadata.get("sample_id", path),
                    selected_count,
                    mean_distance,
                    median_distance,
                    above_threshold,
                    tuple(float(value.item()) for value in score_quantiles),
                    node_count,
                )
            )
            if args.progress_every > 0 and (record_index + 1) % args.progress_every == 0:
                print(
                    f"evaluated {record_index + 1}/{len(paths)} "
                    f"candidates={above_threshold} selected={selected_count} "
                    f"precision={precision:.4f} coverage={coverage:.4f} "
                    f"score_p99={float(score_quantiles[2].item()):.4f} "
                    f"score_max={float(score_quantiles[3].item()):.4f}",
                    flush=True,
                )

    rows.sort(key=lambda row: row[0])
    print(f"device: {device}")
    print(f"records: {len(paths)}")
    print(f"score_threshold: {args.score_threshold}")
    print(f"min_candidates: {args.min_candidates}")
    print(f"candidate_score_floor: {args.candidate_score_floor}")
    print(f"top_proposals: {args.top_proposals}")
    print(f"nms_mode: {args.nms_mode}")
    print(f"nms_radius: {args.nms_radius}")
    print(f"iou_threshold: {args.iou_threshold}")
    print(f"proposal_precision@{args.match_distance:g}: {total_hits / total_selected if total_selected else 0.0:.4f} ({total_hits}/{total_selected})")
    print(f"skeleton_coverage@{args.coverage_distance:g}: {total_covered / total_nodes if total_nodes else 0.0:.4f} ({total_covered}/{total_nodes})")
    if distance_means:
        print(f"mean_selected_distance: {statistics.fmean(distance_means):.4f}")
    if coverage_rates:
        print(f"mean_sample_coverage: {statistics.fmean(coverage_rates):.4f}")
    if score_quantile_rows:
        all_scores = torch.cat(score_quantile_rows)
        quantiles = all_scores.quantile(torch.tensor([0.5, 0.9, 0.99, 1.0], device=all_scores.device))
        print(
            "score_quantiles: "
            f"p50={float(quantiles[0].item()):.4f} "
            f"p90={float(quantiles[1].item()):.4f} "
            f"p99={float(quantiles[2].item()):.4f} "
            f"max={float(quantiles[3].item()):.4f}"
        )
    nonzero_coverages = [coverage for coverage in coverage_rates if coverage > 0.0]
    zero_selected = sum(1 for row in rows if row[4] == 0)
    print(f"zero_selected_samples: {zero_selected}/{len(rows)}")
    if nonzero_coverages:
        print(f"mean_nonzero_sample_coverage: {statistics.fmean(nonzero_coverages):.4f}")
    print("lowest_coverage_samples:")
    for (
        coverage,
        precision,
        record_index,
        sample_id,
        selected_count,
        mean_distance,
        median_distance,
        above_threshold,
        quantiles,
        node_count,
    ) in rows[:5]:
        mean_text = "inf" if math.isinf(mean_distance) else f"{mean_distance:.3f}"
        median_text = "inf" if math.isinf(median_distance) else f"{median_distance:.3f}"
        print(
            f"  index={record_index} coverage={coverage:.4f} precision={precision:.4f} "
            f"candidates={above_threshold} selected={selected_count} "
            f"score_p50={quantiles[0]:.4f} score_p90={quantiles[1]:.4f} "
            f"score_p99={quantiles[2]:.4f} score_max={quantiles[3]:.4f} "
            f"nodes={node_count} mean_distance={mean_text} median_distance={median_text} sample_id={sample_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

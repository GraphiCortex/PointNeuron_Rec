from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
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


@dataclass(frozen=True)
class CachedPrediction:
    scores: object
    centers: object
    radii: object
    skeleton: object
    sample_id: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep proposal checkpoints and selection policies on a split.")
    parser.add_argument("--split-file", required=True, help="Path to split JSON.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to sweep.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path or label=checkpoint path. Repeat to compare multiple checkpoints.",
    )
    parser.add_argument("--thresholds", default="0.5,0.3,0.2", help="Comma-separated score thresholds.")
    parser.add_argument("--top-proposals", default="128,256,512", help="Comma-separated selected proposal caps.")
    parser.add_argument("--min-candidates", default="0,128,256,512", help="Comma-separated adaptive pre-NMS candidate floors.")
    parser.add_argument("--candidate-score-floor", type=float, default=0.0, help="Minimum score allowed by adaptive candidate floors.")
    parser.add_argument("--nms-mode", default="sphere", choices=["sphere", "distance"], help="Proposal downsampling mode.")
    parser.add_argument("--nms-radius", type=float, default=8.0, help="Distance NMS radius in voxels.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Sphere IoU threshold for spherical NMS.")
    parser.add_argument("--match-distance", type=float, default=6.0, help="Proposal-to-SWC distance counted as a hit.")
    parser.add_argument("--coverage-distance", type=float, default=8.0, help="SWC node distance counted as covered.")
    parser.add_argument("--beta", type=float, default=2.0, help="F-beta beta. Values >1 prioritize coverage.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of ranked rows to print.")
    parser.add_argument("--max-records", type=int, help="Optional limit for a quick sweep.")
    parser.add_argument("--csv", help="Optional CSV output path for all rows.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run model inference on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before sweeping proposals.")
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

    thresholds = parse_float_list(args.thresholds)
    top_proposals_values = parse_int_list(args.top_proposals)
    min_candidate_values = parse_int_list(args.min_candidates)
    checkpoint_specs = [parse_checkpoint_spec(value) for value in args.checkpoint]

    rows: list[dict[str, object]] = []
    print(f"device: {device}")
    print(f"records: {len(paths)}")
    print(f"decision_score: F{args.beta:g} + 0.25*p10_coverage - 0.50*zero_rate - 0.05*excess_distance")
    for label, checkpoint_path in checkpoint_specs:
        predictions = run_checkpoint(
            label=label,
            checkpoint_path=checkpoint_path,
            paths=paths,
            device=device,
            torch=torch,
            DGCNNEncoder=DGCNNEncoder,
            SkeletonProposalHead=SkeletonProposalHead,
        )
        for threshold, top_proposals, min_candidates in product(thresholds, top_proposals_values, min_candidate_values):
            if min_candidates > 0 and min_candidates < top_proposals:
                continue
            rows.append(
                evaluate_config(
                    checkpoint_label=label,
                    predictions=predictions,
                    threshold=threshold,
                    top_proposals=top_proposals,
                    min_candidates=min_candidates,
                    candidate_score_floor=args.candidate_score_floor,
                    nms_mode=args.nms_mode,
                    nms_radius=args.nms_radius,
                    iou_threshold=args.iou_threshold,
                    match_distance=args.match_distance,
                    coverage_distance=args.coverage_distance,
                    beta=args.beta,
                    torch=torch,
                )
            )

    rows.sort(key=lambda row: float(row["decision_score"]), reverse=True)
    print_ranked_rows(rows[: args.top_k])
    if args.csv:
        output_path = Path(args.csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv: {output_path}")
    return 0


def parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(value)
    return path.stem, path


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run_checkpoint(label, checkpoint_path, paths, device, torch, DGCNNEncoder, SkeletonProposalHead) -> list[CachedPrediction]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    k = int(checkpoint_args.get("k", 20))
    proposal_coordinate_mode = checkpoint_args.get("proposal_coordinate_mode", "raw")
    encoder = DGCNNEncoder(k=k).to(device)
    proposal_in_channels = int(checkpoint["proposal"]["mlp.0.weight"].shape[1])
    proposal = SkeletonProposalHead(
        in_channels=proposal_in_channels,
        include_xyz=(proposal_in_channels == encoder.output_dim + 3),
        coordinate_mode=proposal_coordinate_mode,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    proposal.load_state_dict(checkpoint["proposal"])
    encoder.eval()
    proposal.eval()

    predictions = []
    with torch.no_grad():
        for record_index, path in enumerate(paths, start=1):
            dataset = TrainingCacheDataset([path])
            batch = collate_training_records([dataset[0]])
            points = batch["points"].to(device)
            skeleton_nodes = batch["skeleton_nodes"].to(device)
            skeleton_mask = batch["skeleton_mask"].to(device)
            features = encoder(points)
            output = proposal(points, features)
            predictions.append(
                CachedPrediction(
                    scores=output.objectness_logits.softmax(dim=-1)[0, :, 1].detach().float().cpu(),
                    centers=output.center_proposals[0].detach().float().cpu(),
                    radii=output.radius[0, :, 0].detach().float().cpu(),
                    skeleton=skeleton_nodes[0, skeleton_mask[0]][:, 1:4].detach().float().cpu(),
                    sample_id=str(batch["metadata"][0].get("sample_id", path)),
                )
            )
            print(f"cached {label}: {record_index}/{len(paths)}", flush=True)
    return predictions


def evaluate_config(
    checkpoint_label: str,
    predictions: list[CachedPrediction],
    threshold: float,
    top_proposals: int,
    min_candidates: int,
    candidate_score_floor: float,
    nms_mode: str,
    nms_radius: float,
    iou_threshold: float,
    match_distance: float,
    coverage_distance: float,
    beta: float,
    torch,
) -> dict[str, object]:
    total_selected = 0
    total_hits = 0
    total_nodes = 0
    total_covered = 0
    coverage_rates = []
    precision_rates = []
    selected_counts = []
    mean_distances = []
    zero_selected = 0

    for prediction in predictions:
        selected_indices = select_proposals(
            centers=prediction.centers,
            radii=prediction.radii,
            scores=prediction.scores,
            score_threshold=threshold,
            min_candidates=min_candidates,
            candidate_score_floor=candidate_score_floor,
            top_proposals=top_proposals,
            nms_mode=nms_mode,
            nms_radius=nms_radius,
            iou_threshold=iou_threshold,
        )
        selected_count = len(selected_indices)
        node_count = int(prediction.skeleton.shape[0])
        if selected_count:
            selected_centers = prediction.centers[selected_indices]
            selected_to_swc = torch.cdist(selected_centers, prediction.skeleton).min(dim=1).values
            swc_to_selected = torch.cdist(prediction.skeleton, selected_centers).min(dim=1).values
            hits = int((selected_to_swc <= match_distance).sum().item())
            covered = int((swc_to_selected <= coverage_distance).sum().item())
            mean_distances.append(float(selected_to_swc.mean().item()))
        else:
            hits = 0
            covered = 0
            zero_selected += 1

        precision = hits / selected_count if selected_count else 0.0
        coverage = covered / node_count if node_count else 0.0
        precision_rates.append(precision)
        coverage_rates.append(coverage)
        selected_counts.append(selected_count)
        total_selected += selected_count
        total_hits += hits
        total_nodes += node_count
        total_covered += covered

    precision = total_hits / total_selected if total_selected else 0.0
    coverage = total_covered / total_nodes if total_nodes else 0.0
    fbeta = f_beta(precision, coverage, beta)
    p10_coverage = quantile(coverage_rates, 0.10)
    min_coverage = min(coverage_rates) if coverage_rates else 0.0
    zero_rate = zero_selected / len(predictions) if predictions else 0.0
    mean_distance = statistics.fmean(mean_distances) if mean_distances else float("inf")
    excess_distance = max(0.0, mean_distance - coverage_distance) / coverage_distance if mean_distances else 1.0
    decision_score = fbeta + 0.25 * p10_coverage - 0.50 * zero_rate - 0.05 * excess_distance

    return {
        "decision_score": round(decision_score, 6),
        "checkpoint": checkpoint_label,
        "threshold": threshold,
        "top_proposals": top_proposals,
        "min_candidates": min_candidates,
        "precision": round(precision, 6),
        "coverage": round(coverage, 6),
        f"f{beta:g}": round(fbeta, 6),
        "mean_sample_coverage": round(statistics.fmean(coverage_rates), 6),
        "p10_coverage": round(p10_coverage, 6),
        "min_coverage": round(min_coverage, 6),
        "zero_selected": zero_selected,
        "mean_selected_distance": round(mean_distance, 6) if mean_distances else "inf",
        "mean_selected_count": round(statistics.fmean(selected_counts), 3),
    }


def f_beta(precision: float, coverage: float, beta: float) -> float:
    if precision <= 0.0 or coverage <= 0.0:
        return 0.0
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * coverage / (beta_squared * precision + coverage)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def print_ranked_rows(rows: list[dict[str, object]]) -> None:
    print("ranked_configs:")
    for rank, row in enumerate(rows, start=1):
        print(
            f"  {rank:02d} score={row['decision_score']} checkpoint={row['checkpoint']} "
            f"thr={row['threshold']} top={row['top_proposals']} mincand={row['min_candidates']} "
            f"precision={row['precision']} coverage={row['coverage']} "
            f"p10={row['p10_coverage']} zero={row['zero_selected']} "
            f"mean_dist={row['mean_selected_distance']} mean_selected={row['mean_selected_count']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

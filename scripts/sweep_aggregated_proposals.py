from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
import statistics
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.swc import parse_swc
from scripts.aggregate_proposals import skeleton_role_masks
from scripts.visualize_proposals import select_proposals


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep final selection/NMS over saved full-sample proposal aggregations.")
    parser.add_argument("--input-dir", default="tmp/connectivity_dataset/aggregations", help="Directory of *_proposals.npz files.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--thresholds", default="0.0,0.1,0.2,0.3,0.5", help="Comma-separated final score thresholds.")
    parser.add_argument("--top-proposals", default="256,512,768,1024,1536,2048,4096", help="Comma-separated final proposal caps.")
    parser.add_argument("--nms-radii", default="4,6,8,10,12,14,18", help="Comma-separated final distance-NMS radii.")
    parser.add_argument("--nms-mode", default="distance", choices=["distance", "sphere"], help="Final NMS mode.")
    parser.add_argument("--match-distance", type=float, default=6.0, help="Proposal-to-SWC distance counted as precision hit.")
    parser.add_argument("--coverage-distance", type=float, default=8.0, help="SWC node distance counted as covered.")
    parser.add_argument("--beta", type=float, default=2.0, help="F-beta beta; values >1 prioritize coverage.")
    parser.add_argument("--top-k", type=int, default=12, help="Rows to print.")
    parser.add_argument("--csv-output", default="tmp/aggregated_proposal_sweep.csv", help="CSV path for all configs. Use empty string to skip.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required for distance calculations.")
        return 2

    samples = scan_gold166(args.root)
    records = load_records(Path(args.input_dir), samples, torch=torch)
    if not records:
        print(f"No aggregation files with all_centers found in {args.input_dir}")
        return 2

    rows = []
    for threshold, top_proposals, nms_radius in product(
        parse_float_list(args.thresholds),
        parse_int_list(args.top_proposals),
        parse_float_list(args.nms_radii),
    ):
        rows.append(
            evaluate_config(
                records=records,
                threshold=threshold,
                top_proposals=top_proposals,
                nms_radius=nms_radius,
                nms_mode=args.nms_mode,
                match_distance=args.match_distance,
                coverage_distance=args.coverage_distance,
                beta=args.beta,
                torch=torch,
            )
        )

    rows.sort(key=lambda row: float(row["decision_score"]), reverse=True)
    print(f"samples: {len(records)}")
    print(f"decision_score: F{args.beta:g} + 0.25*p10_terminal_coverage - 0.05*excess_distance")
    print_ranked(rows[: args.top_k])
    if args.csv_output:
        write_csv(Path(args.csv_output), rows)
        print(f"csv_output: {args.csv_output}")
    return 0


def load_records(input_dir: Path, samples, torch) -> list[dict]:
    records = []
    for path in sorted(input_dir.glob("*_proposals.npz")):
        payload = np.load(path, allow_pickle=False)
        if "all_centers" not in payload or "metadata" not in payload:
            continue
        metadata = json.loads(str(payload["metadata"]))
        sample_index = int(metadata["sample_index"])
        swc = parse_swc(samples[sample_index].swc_path)
        skeleton = np.array(
            [[node.node_id, node.x, node.y, node.z, node.radius, node.parent_id] for node in swc_to_skeleton_records(swc)],
            dtype=np.float32,
        )
        endpoint_mask, branch_mask = skeleton_role_masks(skeleton)
        records.append(
            {
                "path": path,
                "sample_index": sample_index,
                "sample_id": metadata.get("sample_id", str(path)),
                "centers": torch.from_numpy(payload["all_centers"].astype(np.float32, copy=False)),
                "radii": torch.from_numpy(payload["all_radii"].astype(np.float32, copy=False)),
                "scores": torch.from_numpy(payload["all_scores"].astype(np.float32, copy=False)),
                "skeleton": torch.from_numpy(skeleton[:, 1:4].astype(np.float32, copy=False)),
                "endpoint_mask": torch.from_numpy(endpoint_mask),
                "branch_mask": torch.from_numpy(branch_mask),
            }
        )
    return records


def evaluate_config(
    records: list[dict],
    threshold: float,
    top_proposals: int,
    nms_radius: float,
    nms_mode: str,
    match_distance: float,
    coverage_distance: float,
    beta: float,
    torch,
) -> dict:
    total_selected = 0
    total_hits = 0
    total_nodes = 0
    total_covered = 0
    total_terminals = 0
    total_terminal_covered = 0
    coverage_rates = []
    terminal_rates = []
    precision_rates = []
    selected_counts = []
    mean_distances = []

    for record in records:
        selected = select_proposals(
            centers=record["centers"],
            radii=record["radii"],
            scores=record["scores"],
            score_threshold=threshold,
            top_proposals=top_proposals,
            nms_mode=nms_mode,
            nms_radius=nms_radius,
            iou_threshold=0.1,
        )
        selected_count = len(selected)
        selected_counts.append(selected_count)
        node_count = int(record["skeleton"].shape[0])
        terminal_count = int(record["endpoint_mask"].sum().item())
        if selected_count:
            selected_centers = record["centers"][selected]
            selected_to_swc = torch.cdist(selected_centers, record["skeleton"]).min(dim=1).values
            swc_to_selected = torch.cdist(record["skeleton"], selected_centers).min(dim=1).values
            hits = int((selected_to_swc <= match_distance).sum().item())
            covered_mask = swc_to_selected <= coverage_distance
            covered = int(covered_mask.sum().item())
            terminal_covered = int(covered_mask[record["endpoint_mask"]].sum().item()) if terminal_count else 0
            mean_distances.append(float(selected_to_swc.mean().item()))
        else:
            hits = 0
            covered = 0
            terminal_covered = 0

        total_selected += selected_count
        total_hits += hits
        total_nodes += node_count
        total_covered += covered
        total_terminals += terminal_count
        total_terminal_covered += terminal_covered
        precision_rates.append(hits / selected_count if selected_count else 0.0)
        coverage_rates.append(covered / node_count if node_count else 0.0)
        terminal_rates.append(terminal_covered / terminal_count if terminal_count else 0.0)

    precision = total_hits / total_selected if total_selected else 0.0
    coverage = total_covered / total_nodes if total_nodes else 0.0
    terminal_coverage = total_terminal_covered / total_terminals if total_terminals else 0.0
    fbeta = f_beta(precision, coverage, beta)
    mean_distance = statistics.fmean(mean_distances) if mean_distances else float("inf")
    excess_distance = max(0.0, mean_distance - coverage_distance) / coverage_distance if mean_distances else 1.0
    p10_terminal = quantile(terminal_rates, 0.10)
    decision_score = fbeta + 0.25 * p10_terminal - 0.05 * excess_distance
    return {
        "decision_score": round(decision_score, 6),
        "threshold": threshold,
        "top_proposals": top_proposals,
        "nms_radius": nms_radius,
        "precision": round(precision, 6),
        "coverage": round(coverage, 6),
        "terminal_coverage": round(terminal_coverage, 6),
        f"f{beta:g}": round(fbeta, 6),
        "mean_sample_coverage": round(statistics.fmean(coverage_rates), 6),
        "mean_terminal_coverage": round(statistics.fmean(terminal_rates), 6),
        "p10_terminal_coverage": round(p10_terminal, 6),
        "min_terminal_coverage": round(min(terminal_rates), 6),
        "mean_precision": round(statistics.fmean(precision_rates), 6),
        "mean_selected_count": round(statistics.fmean(selected_counts), 3),
        "mean_selected_distance": round(mean_distance, 6) if mean_distances else "inf",
    }


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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


def print_ranked(rows: list[dict]) -> None:
    print("ranked_configs:")
    for rank, row in enumerate(rows, start=1):
        print(
            f"  {rank:02d} score={row['decision_score']} "
            f"thr={row['threshold']} top={row['top_proposals']} nms={row['nms_radius']} "
            f"precision={row['precision']} coverage={row['coverage']} "
            f"terminal={row['terminal_coverage']} mean_terminal={row['mean_terminal_coverage']} "
            f"p10_terminal={row['p10_terminal_coverage']} mean_selected={row['mean_selected_count']} "
            f"mean_dist={row['mean_selected_distance']}"
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

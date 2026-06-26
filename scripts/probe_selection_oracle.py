from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import SwcTree, parse_swc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "GT-aware diagnostic oracle for proposal-to-skeleton selection. "
            "This is not an inference method; it estimates whether the proposal cloud contains "
            "enough points for a better selector to preserve topology."
        )
    )
    parser.add_argument("--summary", action="append", required=True, help="E2E summary JSON. Can be repeated.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--sample-index", action="append", type=int, help="Limit to specific sample index. Can be repeated.")
    parser.add_argument("--hit-distance", type=float, default=8.0)
    parser.add_argument("--segment-step", type=float, default=4.0)
    parser.add_argument("--candidate-limit", type=int, default=4096, help="Keep this many top-scoring proposal candidates.")
    parser.add_argument("--budget", type=int, action="append", help="Oracle node budget. Can be repeated.")
    parser.add_argument(
        "--same-budget",
        action="store_true",
        help="Also run oracle with the current selected-node count for each sample.",
    )
    parser.add_argument("--endpoint-weight", type=float, default=8.0)
    parser.add_argument("--branch-weight", type=float, default=6.0)
    parser.add_argument("--segment-weight", type=float, default=1.0)
    parser.add_argument("--csv-output", default="tmp/selection_oracle_probe.csv")
    parser.add_argument("--json-output", default="tmp/selection_oracle_probe.json")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    requested_samples = set(args.sample_index or [])
    summary_rows = []
    for summary_path in [Path(path) for path in args.summary]:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary.get("samples", []):
            sample_index = int(row["sample_index"])
            if requested_samples and sample_index not in requested_samples:
                continue
            summary_rows.append(row)

    if not summary_rows:
        raise ValueError("No samples matched the requested summaries/sample indices")

    budgets = sorted(set(args.budget or []))
    rows = []
    for summary_row in summary_rows:
        rows.extend(
            probe_sample(
                summary_row=summary_row,
                samples=samples,
                static_budgets=budgets,
                same_budget=bool(args.same_budget),
                hit_distance=float(args.hit_distance),
                segment_step=float(args.segment_step),
                candidate_limit=int(args.candidate_limit),
                endpoint_weight=float(args.endpoint_weight),
                branch_weight=float(args.branch_weight),
                segment_weight=float(args.segment_weight),
            )
        )

    write_csv(Path(args.csv_output), rows)
    report = {"samples": rows, "summary": summarize(rows)}
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"rows: {len(rows)}")
    for key, value in report["summary"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("largest_oracle_lifts:")
    for row in sorted(rows, key=lambda item: item["oracle_selected_segment_lift"], reverse=True)[:10]:
        print(
            f"  sample {row['sample_index']} budget={row['oracle_budget']} "
            f"selected_seg={row['selected_segment_coverage']:.4f} "
            f"oracle_seg={row['oracle_segment_coverage']:.4f} "
            f"lift={row['oracle_selected_segment_lift']:.4f} "
            f"selected_endpoint={row['selected_endpoint_coverage']:.4f} "
            f"oracle_endpoint={row['oracle_endpoint_coverage']:.4f}"
        )
    print(f"csv_output: {args.csv_output}")
    print(f"json_output: {args.json_output}")
    return 0


def probe_sample(
    summary_row: dict,
    samples: list,
    static_budgets: list[int],
    same_budget: bool,
    hit_distance: float,
    segment_step: float,
    candidate_limit: int,
    endpoint_weight: float,
    branch_weight: float,
    segment_weight: float,
) -> list[dict]:
    sample_index = int(summary_row["sample_index"])
    sample = samples[sample_index]
    proposals = np.load(Path(summary_row["proposal_path"]), allow_pickle=False)
    graph = np.load(Path(summary_row["graph_path"]), allow_pickle=False)
    graph_metadata = json.loads(str(graph["metadata"])) if "metadata" in graph else {}

    swc = parse_swc(sample.swc_path)
    swc_info = swc_geometry(swc, segment_step=segment_step)

    centers = proposals["centers"].astype(np.float32, copy=False)
    scores = proposals["scores"].astype(np.float32, copy=False) if "scores" in proposals else np.ones((centers.shape[0],), dtype=np.float32)
    threshold = float(summary_row.get("graph_min_proposal_score", graph_metadata.get("min_proposal_score", 0.0)))
    candidate_indices = np.flatnonzero(scores >= threshold)
    if candidate_indices.size > candidate_limit > 0:
        order = np.argsort(scores[candidate_indices])[::-1][:candidate_limit]
        candidate_indices = candidate_indices[order]
    candidates = centers[candidate_indices]
    candidate_scores = scores[candidate_indices]

    selected_centers = graph["centers"].astype(np.float32, copy=False)
    selected_metrics = all_coverage_metrics(selected_centers, swc_info, hit_distance)
    proposal_metrics = all_coverage_metrics(candidates, swc_info, hit_distance)

    target_points, target_weights = weighted_targets(
        swc_info,
        segment_weight=segment_weight,
        endpoint_weight=endpoint_weight,
        branch_weight=branch_weight,
    )
    budgets = list(static_budgets)
    if same_budget:
        budgets.append(int(selected_centers.shape[0]))
    budgets = sorted({budget for budget in budgets if budget > 0})
    if not budgets:
        budgets = [int(selected_centers.shape[0])]

    rows = []
    for budget in budgets:
        oracle_indices = greedy_weighted_cover(
            candidates=candidates,
            scores=candidate_scores,
            targets=target_points,
            weights=target_weights,
            budget=min(int(budget), int(candidates.shape[0])),
            hit_distance=hit_distance,
        )
        oracle_centers = candidates[oracle_indices] if oracle_indices.size else candidates[:0]
        oracle_metrics = all_coverage_metrics(oracle_centers, swc_info, hit_distance)
        rows.append(
            {
                "sample_index": sample_index,
                "sample_id": sample.sample_id,
                "proposal_path": str(summary_row["proposal_path"]),
                "graph_path": str(summary_row["graph_path"]),
                "proposal_candidates": int(candidates.shape[0]),
                "selected_nodes": int(selected_centers.shape[0]),
                "oracle_budget": int(budget),
                "proposal_segment_coverage": proposal_metrics["segment_coverage"],
                "selected_segment_coverage": selected_metrics["segment_coverage"],
                "oracle_segment_coverage": oracle_metrics["segment_coverage"],
                "oracle_selected_segment_lift": oracle_metrics["segment_coverage"] - selected_metrics["segment_coverage"],
                "proposal_endpoint_coverage": proposal_metrics["endpoint_coverage"],
                "selected_endpoint_coverage": selected_metrics["endpoint_coverage"],
                "oracle_endpoint_coverage": oracle_metrics["endpoint_coverage"],
                "oracle_selected_endpoint_lift": oracle_metrics["endpoint_coverage"] - selected_metrics["endpoint_coverage"],
                "proposal_branch_coverage": proposal_metrics["branch_coverage"],
                "selected_branch_coverage": selected_metrics["branch_coverage"],
                "oracle_branch_coverage": oracle_metrics["branch_coverage"],
                "oracle_selected_branch_lift": oracle_metrics["branch_coverage"] - selected_metrics["branch_coverage"],
                "proposal_score_threshold": threshold,
                "hit_distance": float(hit_distance),
            }
        )
    return rows


def weighted_targets(
    swc_info: dict[str, np.ndarray],
    segment_weight: float,
    endpoint_weight: float,
    branch_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [
        swc_info["segments"],
        swc_info["endpoints"],
        swc_info["branches"],
    ]
    weights = [
        np.full((arrays[0].shape[0],), segment_weight, dtype=np.float32),
        np.full((arrays[1].shape[0],), endpoint_weight, dtype=np.float32),
        np.full((arrays[2].shape[0],), branch_weight, dtype=np.float32),
    ]
    points = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    target_weights = np.concatenate(weights, axis=0).astype(np.float32, copy=False)
    return points, target_weights


def greedy_weighted_cover(
    candidates: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    budget: int,
    hit_distance: float,
) -> np.ndarray:
    if candidates.size == 0 or targets.size == 0 or budget <= 0:
        return np.zeros((0,), dtype=np.int64)
    target_tree = cKDTree(targets)
    coverage_sets = target_tree.query_ball_point(candidates, r=float(hit_distance))
    selected: list[int] = []
    selected_mask = np.zeros((candidates.shape[0],), dtype=bool)
    covered = np.zeros((targets.shape[0],), dtype=bool)

    for _ in range(min(int(budget), int(candidates.shape[0]))):
        best_index = -1
        best_gain = 0.0
        best_score = -np.inf
        for index, covered_targets in enumerate(coverage_sets):
            if selected_mask[index] or not covered_targets:
                continue
            target_indices = np.array(covered_targets, dtype=np.int64)
            uncovered = target_indices[~covered[target_indices]]
            if uncovered.size == 0:
                continue
            gain = float(weights[uncovered].sum())
            score = float(scores[index])
            if gain > best_gain or (gain == best_gain and score > best_score):
                best_gain = gain
                best_score = score
                best_index = int(index)
        if best_index < 0:
            break
        selected.append(best_index)
        selected_mask[best_index] = True
        covered[np.array(coverage_sets[best_index], dtype=np.int64)] = True

    return np.array(selected, dtype=np.int64)


def all_coverage_metrics(candidate_points: np.ndarray, swc_info: dict[str, np.ndarray], hit_distance: float) -> dict[str, float]:
    segment = coverage_metrics(candidate_points, swc_info["segments"], hit_distance)
    endpoint = coverage_metrics(candidate_points, swc_info["endpoints"], hit_distance)
    branch = coverage_metrics(candidate_points, swc_info["branches"], hit_distance)
    node = coverage_metrics(candidate_points, swc_info["nodes"], hit_distance)
    return {
        "segment_coverage": segment["coverage"],
        "endpoint_coverage": endpoint["coverage"],
        "branch_coverage": branch["coverage"],
        "node_coverage": node["coverage"],
    }


def swc_geometry(swc: SwcTree, segment_step: float) -> dict[str, np.ndarray]:
    nodes_by_id = {node.node_id: node for node in swc.nodes}
    child_counts = {node.node_id: 0 for node in swc.nodes}
    for node in swc.nodes:
        if node.parent_id in child_counts:
            child_counts[node.parent_id] += 1

    node_coords = np.array([[node.x, node.y, node.z] for node in swc.nodes], dtype=np.float32)
    endpoint_coords = np.array(
        [[node.x, node.y, node.z] for node in swc.nodes if child_counts.get(node.node_id, 0) == 0],
        dtype=np.float32,
    ).reshape(-1, 3)
    branch_coords = np.array(
        [[node.x, node.y, node.z] for node in swc.nodes if child_counts.get(node.node_id, 0) >= 2],
        dtype=np.float32,
    ).reshape(-1, 3)
    segment_points: list[np.ndarray] = []
    for node in swc.nodes:
        if node.parent_id == -1 or node.parent_id not in nodes_by_id:
            continue
        parent = nodes_by_id[node.parent_id]
        start = np.array([node.x, node.y, node.z], dtype=np.float32)
        end = np.array([parent.x, parent.y, parent.z], dtype=np.float32)
        segment_points.append(sample_segment(start, end, step=segment_step))
    segment_coords = np.concatenate(segment_points, axis=0) if segment_points else node_coords
    return {
        "nodes": node_coords,
        "endpoints": endpoint_coords,
        "branches": branch_coords,
        "segments": segment_coords.astype(np.float32, copy=False),
    }


def sample_segment(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / max(step, 1.0e-3))) + 1)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32).reshape(-1, 1)
    return start.reshape(1, 3) * (1.0 - t) + end.reshape(1, 3) * t


def coverage_metrics(candidate_points: np.ndarray, target_points: np.ndarray, hit_distance: float) -> dict[str, float]:
    if target_points.size == 0:
        return {"coverage": 1.0}
    if candidate_points.size == 0:
        return {"coverage": 0.0}
    tree = cKDTree(candidate_points)
    distances, _ = tree.query(target_points, k=1)
    return {"coverage": float(np.mean(distances <= hit_distance))}


def summarize(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "mean_selected_segment_coverage": mean(rows, "selected_segment_coverage"),
        "mean_oracle_segment_coverage": mean(rows, "oracle_segment_coverage"),
        "mean_oracle_selected_segment_lift": mean(rows, "oracle_selected_segment_lift"),
        "mean_selected_endpoint_coverage": mean(rows, "selected_endpoint_coverage"),
        "mean_oracle_endpoint_coverage": mean(rows, "oracle_endpoint_coverage"),
        "mean_selected_branch_coverage": mean(rows, "selected_branch_coverage"),
        "mean_oracle_branch_coverage": mean(rows, "oracle_branch_coverage"),
    }


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return statistics.fmean(values) if values else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

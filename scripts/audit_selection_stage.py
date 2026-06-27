from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import SwcTree, parse_swc


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit proposal-to-graph skeletal point selection coverage.")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="End-to-end summary JSON. Can be repeated.",
    )
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--hit-distance", type=float, default=8.0, help="Voxel distance counted as covered.")
    parser.add_argument(
        "--proposal-score-threshold",
        type=float,
        default=None,
        help="Proposal score cutoff. Defaults to graph_min_proposal_score from summary row, or 0.0.",
    )
    parser.add_argument(
        "--segment-step",
        type=float,
        default=4.0,
        help="Sampling step along GT SWC edges for segment coverage.",
    )
    parser.add_argument("--csv-output", default="tmp/selection_stage_audit.csv")
    parser.add_argument("--json-output", default="tmp/selection_stage_audit.json")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    summary_paths = [Path(path) for path in args.summary]
    topology_by_graph = load_topology_reports(summary_paths)
    rows = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary.get("samples", []):
            rows.append(
                audit_sample(
                    summary_row=row,
                    samples=samples,
                    topology_by_graph=topology_by_graph,
                    proposal_score_threshold=args.proposal_score_threshold,
                    hit_distance=float(args.hit_distance),
                    segment_step=float(args.segment_step),
                )
            )

    write_csv(Path(args.csv_output), rows)
    report = {"samples": rows, "summary": summarize(rows)}
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"samples: {len(rows)}")
    for key, value in report["summary"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("worst_selected_segment_coverage:")
    for row in sorted(rows, key=lambda item: item["selected_segment_coverage"])[:10]:
        print(
            f"  sample {row['sample_index']}: bottleneck={row['bottleneck']} "
            f"f1={row['edge_f1']:.4f} proposal_seg={row['proposal_segment_coverage']:.4f} "
            f"selected_seg={row['selected_segment_coverage']:.4f} "
            f"proposal_nodes={row['proposal_nodes']} selected_nodes={row['selected_nodes']}"
        )
    print(f"csv_output: {args.csv_output}")
    print(f"json_output: {args.json_output}")
    return 0


def audit_sample(
    summary_row: dict,
    samples: list,
    topology_by_graph: dict[str, dict],
    proposal_score_threshold: float | None,
    hit_distance: float,
    segment_step: float,
) -> dict:
    sample_index = int(summary_row["sample_index"])
    sample = samples[sample_index]
    proposal_path = Path(summary_row["proposal_path"])
    graph_path = Path(summary_row["graph_path"])
    proposals = np.load(proposal_path, allow_pickle=False)
    graph = np.load(graph_path, allow_pickle=False)
    graph_metadata = json.loads(str(graph["metadata"])) if "metadata" in graph else {}
    swc = parse_swc(sample.swc_path)
    swc_info = swc_geometry(swc, segment_step=segment_step)

    threshold = proposal_score_threshold
    if threshold is None:
        threshold = float(summary_row.get("graph_min_proposal_score", graph_metadata.get("min_proposal_score", 0.0)))

    proposal_centers = proposals["centers"].astype(np.float32, copy=False)
    if "scores" in proposals:
        proposal_scores = proposals["scores"].astype(np.float32, copy=False)
        proposal_centers = proposal_centers[proposal_scores >= float(threshold)]
    selected_centers = graph["centers"].astype(np.float32, copy=False)

    proposal_node = coverage_metrics(proposal_centers, swc_info["nodes"], hit_distance)
    selected_node = coverage_metrics(selected_centers, swc_info["nodes"], hit_distance)
    proposal_endpoint = coverage_metrics(proposal_centers, swc_info["endpoints"], hit_distance)
    selected_endpoint = coverage_metrics(selected_centers, swc_info["endpoints"], hit_distance)
    proposal_branch = coverage_metrics(proposal_centers, swc_info["branches"], hit_distance)
    selected_branch = coverage_metrics(selected_centers, swc_info["branches"], hit_distance)
    proposal_segment = coverage_metrics(proposal_centers, swc_info["segments"], hit_distance)
    selected_segment = coverage_metrics(selected_centers, swc_info["segments"], hit_distance)
    proposal_skeleton = candidate_to_target_metrics(proposal_centers, swc_info["segments"], hit_distance)
    selected_skeleton = candidate_to_target_metrics(selected_centers, swc_info["segments"], hit_distance)
    selected_spacing = nearest_neighbor_metrics(selected_centers)

    topology = topology_by_graph.get(path_key(graph_path), {})
    edge_f1 = float(topology.get("edge_f1", float("nan")))
    bottleneck = classify_bottleneck(
        proposal_segment_coverage=proposal_segment["coverage"],
        selected_segment_coverage=selected_segment["coverage"],
        proposal_endpoint_coverage=proposal_endpoint["coverage"],
        selected_endpoint_coverage=selected_endpoint["coverage"],
        proposal_branch_coverage=proposal_branch["coverage"],
        selected_branch_coverage=selected_branch["coverage"],
        edge_f1=edge_f1,
    )

    return {
        "sample_index": sample_index,
        "sample_id": sample.sample_id,
        "proposal_path": str(proposal_path),
        "graph_path": str(graph_path),
        "proposal_score_threshold": float(threshold),
        "proposal_nodes": int(proposal_centers.shape[0]),
        "selected_nodes": int(selected_centers.shape[0]),
        "gt_nodes": int(swc_info["nodes"].shape[0]),
        "gt_endpoints": int(swc_info["endpoints"].shape[0]),
        "gt_branchpoints": int(swc_info["branches"].shape[0]),
        "gt_segment_samples": int(swc_info["segments"].shape[0]),
        "gt_length": float(swc_info["length"]),
        "proposal_node_coverage": proposal_node["coverage"],
        "selected_node_coverage": selected_node["coverage"],
        "node_coverage_drop": proposal_node["coverage"] - selected_node["coverage"],
        "proposal_endpoint_coverage": proposal_endpoint["coverage"],
        "selected_endpoint_coverage": selected_endpoint["coverage"],
        "endpoint_coverage_drop": proposal_endpoint["coverage"] - selected_endpoint["coverage"],
        "proposal_branch_coverage": proposal_branch["coverage"],
        "selected_branch_coverage": selected_branch["coverage"],
        "branch_coverage_drop": proposal_branch["coverage"] - selected_branch["coverage"],
        "proposal_segment_coverage": proposal_segment["coverage"],
        "selected_segment_coverage": selected_segment["coverage"],
        "segment_coverage_drop": proposal_segment["coverage"] - selected_segment["coverage"],
        "selected_segment_mean_distance": selected_segment["mean_distance"],
        "selected_segment_p90_distance": selected_segment["p90_distance"],
        "selected_segment_max_distance": selected_segment["max_distance"],
        "proposal_skeleton_precision": proposal_skeleton["hit_fraction"],
        "selected_skeleton_precision": selected_skeleton["hit_fraction"],
        "skeleton_precision_drop": proposal_skeleton["hit_fraction"] - selected_skeleton["hit_fraction"],
        "selected_skeleton_mean_distance": selected_skeleton["mean_distance"],
        "selected_skeleton_p90_distance": selected_skeleton["p90_distance"],
        "selected_spacing_p10": selected_spacing["p10_distance"],
        "selected_spacing_median": selected_spacing["median_distance"],
        "selected_spacing_mean": selected_spacing["mean_distance"],
        "selected_redundant_fraction": selected_spacing["redundant_fraction"],
        "selected_nodes_per_100_voxels": (
            float(selected_centers.shape[0]) / max(float(swc_info["length"]), 1.0e-6) * 100.0
        ),
        "edge_f1": edge_f1,
        "bridge_edges": int(topology.get("bridge_edges", summary_row.get("bridge_edges", -1))),
        "reachable_edge_fraction": float(topology.get("reachable_edge_fraction", summary_row.get("reachable_edge_fraction", float("nan")))),
        "bottleneck": bottleneck,
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
    total_length = 0.0
    for node in swc.nodes:
        if node.parent_id == -1 or node.parent_id not in nodes_by_id:
            continue
        parent = nodes_by_id[node.parent_id]
        start = np.array([node.x, node.y, node.z], dtype=np.float32)
        end = np.array([parent.x, parent.y, parent.z], dtype=np.float32)
        total_length += float(np.linalg.norm(end - start))
        segment_points.append(sample_segment(start, end, step=segment_step))
    segment_coords = np.concatenate(segment_points, axis=0) if segment_points else node_coords
    return {
        "nodes": node_coords,
        "endpoints": endpoint_coords,
        "branches": branch_coords,
        "segments": segment_coords.astype(np.float32, copy=False),
        "length": float(total_length),
    }


def sample_segment(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / max(step, 1.0e-3))) + 1)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32).reshape(-1, 1)
    return start.reshape(1, 3) * (1.0 - t) + end.reshape(1, 3) * t


def coverage_metrics(candidate_points: np.ndarray, target_points: np.ndarray, hit_distance: float) -> dict[str, float]:
    if target_points.size == 0:
        return {"coverage": 1.0, "mean_distance": 0.0, "p90_distance": 0.0, "max_distance": 0.0}
    if candidate_points.size == 0:
        return {
            "coverage": 0.0,
            "mean_distance": float("inf"),
            "p90_distance": float("inf"),
            "max_distance": float("inf"),
        }
    distances = nearest_distances(target_points, candidate_points)
    return {
        "coverage": float(np.mean(distances <= hit_distance)),
        "mean_distance": float(np.mean(distances)),
        "p90_distance": float(np.quantile(distances, 0.90)),
        "max_distance": float(np.max(distances)),
    }


def candidate_to_target_metrics(candidate_points: np.ndarray, target_points: np.ndarray, hit_distance: float) -> dict[str, float]:
    if candidate_points.size == 0:
        return {
            "hit_fraction": 0.0,
            "mean_distance": float("inf"),
            "p90_distance": float("inf"),
            "max_distance": float("inf"),
        }
    if target_points.size == 0:
        return {"hit_fraction": 1.0, "mean_distance": 0.0, "p90_distance": 0.0, "max_distance": 0.0}
    distances = nearest_distances(candidate_points, target_points)
    return {
        "hit_fraction": float(np.mean(distances <= hit_distance)),
        "mean_distance": float(np.mean(distances)),
        "p90_distance": float(np.quantile(distances, 0.90)),
        "max_distance": float(np.max(distances)),
    }


def nearest_neighbor_metrics(points: np.ndarray) -> dict[str, float]:
    if points.shape[0] <= 1:
        return {
            "p10_distance": float("inf"),
            "median_distance": float("inf"),
            "mean_distance": float("inf"),
            "redundant_fraction": 0.0,
        }
    nearest = point_nearest_neighbor_distances(points)
    return {
        "p10_distance": float(np.quantile(nearest, 0.10)),
        "median_distance": float(np.median(nearest)),
        "mean_distance": float(np.mean(nearest)),
        "redundant_fraction": float(np.mean(nearest <= 4.0)),
    }


def nearest_distances(query_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    query_points = query_points.astype(np.float32, copy=False)
    target_points = target_points.astype(np.float32, copy=False)
    if cKDTree is not None:
        tree = cKDTree(target_points)
        distances, _ = tree.query(query_points, k=1)
        return np.asarray(distances, dtype=np.float32)
    return brute_force_nearest_distances(query_points, target_points)


def point_nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    if cKDTree is not None:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        return np.asarray(distances[:, 1], dtype=np.float32)
    distance2 = squared_distances(points, points)
    np.fill_diagonal(distance2, np.inf)
    return np.sqrt(np.min(distance2, axis=1)).astype(np.float32, copy=False)


def brute_force_nearest_distances(query_points: np.ndarray, target_points: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    distances = np.empty((query_points.shape[0],), dtype=np.float32)
    for start in range(0, query_points.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), query_points.shape[0])
        distance2 = squared_distances(query_points[start:end], target_points)
        distances[start:end] = np.sqrt(np.min(distance2, axis=1)).astype(np.float32, copy=False)
    return distances


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, dtype=np.float32).reshape(-1, 1)
    right_norm = np.sum(right * right, axis=1, dtype=np.float32).reshape(1, -1)
    distance2 = left_norm + right_norm - 2.0 * left @ right.T
    return np.maximum(distance2, 0.0).astype(np.float32, copy=False)


def classify_bottleneck(
    proposal_segment_coverage: float,
    selected_segment_coverage: float,
    proposal_endpoint_coverage: float,
    selected_endpoint_coverage: float,
    proposal_branch_coverage: float,
    selected_branch_coverage: float,
    edge_f1: float,
) -> str:
    if np.isfinite(edge_f1) and edge_f1 >= 0.80:
        return "ok"
    if proposal_segment_coverage < 0.70 or proposal_endpoint_coverage < 0.65:
        return "proposal"
    critical_drop = max(
        proposal_endpoint_coverage - selected_endpoint_coverage,
        proposal_branch_coverage - selected_branch_coverage,
    )
    if critical_drop >= 0.25 or proposal_segment_coverage - selected_segment_coverage >= 0.35:
        return "selection"
    if np.isfinite(edge_f1) and edge_f1 < 0.70:
        return "connectivity"
    return "ok"


def load_topology_reports(summary_paths: list[Path]) -> dict[str, dict]:
    by_graph: dict[str, dict] = {}
    for summary_path in summary_paths:
        report_path = summary_path.with_name("topology_report.json")
        if not report_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_by_sample = {int(row["sample_index"]): row for row in summary.get("samples", [])}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report.get("samples", []):
            summary_row = summary_by_sample.get(int(row["sample_index"]))
            if summary_row is None:
                continue
            by_graph[path_key(Path(summary_row["graph_path"]))] = row
    return by_graph


def path_key(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def summarize(rows: list[dict]) -> dict:
    bottlenecks: dict[str, int] = {}
    for row in rows:
        bottlenecks[row["bottleneck"]] = bottlenecks.get(row["bottleneck"], 0) + 1
    return {
        "sample_count": len(rows),
        "mean_edge_f1": mean(rows, "edge_f1"),
        "mean_proposal_segment_coverage": mean(rows, "proposal_segment_coverage"),
        "mean_selected_segment_coverage": mean(rows, "selected_segment_coverage"),
        "mean_segment_coverage_drop": mean(rows, "segment_coverage_drop"),
        "mean_proposal_endpoint_coverage": mean(rows, "proposal_endpoint_coverage"),
        "mean_selected_endpoint_coverage": mean(rows, "selected_endpoint_coverage"),
        "mean_proposal_branch_coverage": mean(rows, "proposal_branch_coverage"),
        "mean_selected_branch_coverage": mean(rows, "selected_branch_coverage"),
        "mean_proposal_skeleton_precision": mean(rows, "proposal_skeleton_precision"),
        "mean_selected_skeleton_precision": mean(rows, "selected_skeleton_precision"),
        "mean_skeleton_precision_drop": mean(rows, "skeleton_precision_drop"),
        "mean_selected_spacing_median": mean(rows, "selected_spacing_median"),
        "mean_selected_redundant_fraction": mean(rows, "selected_redundant_fraction"),
        "mean_selected_nodes_per_100_voxels": mean(rows, "selected_nodes_per_100_voxels"),
        "bottlenecks": bottlenecks,
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

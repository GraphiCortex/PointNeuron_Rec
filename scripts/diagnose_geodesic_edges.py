from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.graph.initialization import initialize_proposal_graph


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose foreground-geodesic proposal edges against GT-induced proposal connectivity."
    )
    parser.add_argument("--summary", default="tmp/e2e_baseline_janelia/summary.json", help="End-to-end summary JSON.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--csv-output", default="tmp/e2e_baseline_janelia/edge_diagnostics.csv")
    parser.add_argument("--json-output", default="tmp/e2e_baseline_janelia/edge_diagnostics.json")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    samples = scan_gold166(args.root)
    edge_rows = []
    for sample_row in summary["samples"]:
        edge_rows.extend(diagnose_sample(sample_row, samples))

    report = {
        "edge_count": len(edge_rows),
        "summary": summarize(edge_rows),
        "edges": edge_rows,
    }
    write_csv(Path(args.csv_output), edge_rows)
    Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"edges: {len(edge_rows)}")
    for key, value in report["summary"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"csv_output: {args.csv_output}")
    print(f"json_output: {args.json_output}")
    return 0


def diagnose_sample(summary_row: dict, samples) -> list[dict]:
    graph_path = Path(summary_row["graph_path"])
    graph = np.load(graph_path, allow_pickle=False)
    centers = graph["centers"].astype(np.float32, copy=False)
    edges = graph["edges"].astype(np.int64, copy=False)
    scores = graph["scores"].astype(np.float32, copy=False) if "scores" in graph else np.ones((centers.shape[0],), dtype=np.float32)
    radii = graph["radii"].astype(np.float32, copy=False) if "radii" in graph else np.ones((centers.shape[0],), dtype=np.float32)
    bridge_edges = normalize_edges(graph["bridge_edges"].astype(np.int64, copy=False)) if "bridge_edges" in graph else set()
    reachable = (
        graph["edge_geodesic_reachable"].astype(bool, copy=False)
        if "edge_geodesic_reachable" in graph
        else np.ones((edges.shape[0],), dtype=bool)
    )
    geodesic = (
        graph["edge_tree_distance"].astype(np.float32, copy=False)
        if "edge_tree_distance" in graph
        else np.zeros((edges.shape[0],), dtype=np.float32)
    )
    euclidean = (
        graph["edge_euclidean_distance"].astype(np.float32, copy=False)
        if "edge_euclidean_distance" in graph
        else np.linalg.norm(centers[edges[:, 0]] - centers[edges[:, 1]], axis=1).astype(np.float32, copy=False)
    )
    path_points = graph["edge_path_points"].astype(np.float32, copy=False) if "edge_path_points" in graph else None
    path_offsets = graph["edge_path_offsets"].astype(np.int64, copy=False) if "edge_path_offsets" in graph else None

    sample = samples[int(summary_row["sample_index"])]
    gt = initialize_proposal_graph(centers, parse_swc(sample.swc_path), mode="mst")
    target_edges = normalize_edges(gt.edges)
    degrees = node_degrees(centers.shape[0], edges)
    endpoint_support = continuation_support(edges, centers, path_points, path_offsets)

    rows = []
    for edge_index, (left, right) in enumerate(edges.tolist()):
        left = int(left)
        right = int(right)
        edge = tuple(sorted((left, right)))
        edge_euclidean = float(euclidean[edge_index])
        edge_geodesic = float(geodesic[edge_index])
        rows.append(
            {
                "sample_index": int(summary_row["sample_index"]),
                "edge_index": int(edge_index),
                "left": left,
                "right": right,
                "is_target": edge in target_edges,
                "is_bridge": edge in bridge_edges,
                "is_reachable": bool(reachable[edge_index]),
                "euclidean_distance": edge_euclidean,
                "geodesic_distance": edge_geodesic,
                "geodesic_ratio": edge_geodesic / max(edge_euclidean, 1.0e-3),
                "left_degree": int(degrees[left]),
                "right_degree": int(degrees[right]),
                "min_degree": int(min(degrees[left], degrees[right])),
                "max_degree": int(max(degrees[left], degrees[right])),
                "continuation_mean": float(endpoint_support[edge_index, 0]),
                "continuation_max": float(endpoint_support[edge_index, 1]),
                "score_mean": float((scores[left] + scores[right]) * 0.5),
                "score_min": float(min(scores[left], scores[right])),
                "radius_mean": float((radii[left] + radii[right]) * 0.5),
            }
        )
    return rows


def continuation_support(
    edges: np.ndarray,
    centers: np.ndarray,
    path_points: np.ndarray | None,
    path_offsets: np.ndarray | None,
) -> np.ndarray:
    directions: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(centers.shape[0])]
    edge_dirs: list[tuple[np.ndarray, np.ndarray]] = []
    for edge_index, (left, right) in enumerate(edges.tolist()):
        left = int(left)
        right = int(right)
        left_dir, right_dir = edge_endpoint_directions(edge_index, left, right, centers, path_points, path_offsets)
        edge_dirs.append((left_dir, right_dir))
        directions[left].append((edge_index, left_dir))
        directions[right].append((edge_index, right_dir))

    support = np.zeros((edges.shape[0], 2), dtype=np.float32)
    for edge_index, (left, right) in enumerate(edges.tolist()):
        left = int(left)
        right = int(right)
        left_support = endpoint_alignment(edge_index, edge_dirs[edge_index][0], directions[left])
        right_support = endpoint_alignment(edge_index, edge_dirs[edge_index][1], directions[right])
        support[edge_index, 0] = (left_support + right_support) * 0.5
        support[edge_index, 1] = max(left_support, right_support)
    return support


def edge_endpoint_directions(
    edge_index: int,
    left: int,
    right: int,
    centers: np.ndarray,
    path_points: np.ndarray | None,
    path_offsets: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if path_points is not None and path_offsets is not None and edge_index + 1 < path_offsets.shape[0]:
        start = int(path_offsets[edge_index])
        stop = int(path_offsets[edge_index + 1])
        points = path_points[start:stop]
        if points.shape[0] >= 2:
            left_vec = points[min(3, points.shape[0] - 1)] - points[0]
            right_vec = points[max(0, points.shape[0] - 4)] - points[-1]
            return normalize(left_vec), normalize(right_vec)
    vector = centers[right] - centers[left]
    return normalize(vector), normalize(-vector)


def endpoint_alignment(edge_index: int, direction: np.ndarray, incident: list[tuple[int, np.ndarray]]) -> float:
    values = [abs(float(np.dot(direction, other))) for other_index, other in incident if other_index != edge_index]
    return max(values) if values else 0.0


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-6:
        return np.zeros((3,), dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


def node_degrees(node_count: int, edges: np.ndarray) -> np.ndarray:
    degrees = np.zeros((node_count,), dtype=np.int64)
    for left, right in edges.tolist():
        degrees[int(left)] += 1
        degrees[int(right)] += 1
    return degrees


def normalize_edges(edges: np.ndarray) -> set[tuple[int, int]]:
    result = set()
    for left, right in edges.tolist():
        left = int(left)
        right = int(right)
        if left != right:
            result.add(tuple(sorted((left, right))))
    return result


def summarize(rows: list[dict]) -> dict:
    true_edges = [row for row in rows if row["is_target"]]
    false_edges = [row for row in rows if not row["is_target"]]
    bridge_edges = [row for row in rows if row["is_bridge"]]
    traced_edges = [row for row in rows if not row["is_bridge"]]
    return {
        "true_edges": len(true_edges),
        "false_edges": len(false_edges),
        "bridge_edges": len(bridge_edges),
        "traced_edges": len(traced_edges),
        "true_edge_mean_ratio": mean(row["geodesic_ratio"] for row in true_edges),
        "false_edge_mean_ratio": mean(row["geodesic_ratio"] for row in false_edges),
        "true_edge_mean_continuation": mean(row["continuation_mean"] for row in true_edges),
        "false_edge_mean_continuation": mean(row["continuation_mean"] for row in false_edges),
        "true_edge_mean_score": mean(row["score_mean"] for row in true_edges),
        "false_edge_mean_score": mean(row["score_mean"] for row in false_edges),
        "bridge_hit_rate": mean(1.0 if row["is_target"] else 0.0 for row in bridge_edges),
        "traced_hit_rate": mean(1.0 if row["is_target"] else 0.0 for row in traced_edges),
        "false_bridge_mean_ratio": mean(row["geodesic_ratio"] for row in bridge_edges if not row["is_target"]),
        "false_traced_mean_ratio": mean(row["geodesic_ratio"] for row in traced_edges if not row["is_target"]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

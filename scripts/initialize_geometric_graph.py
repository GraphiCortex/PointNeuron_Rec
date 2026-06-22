from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.graph.initialization import initialize_geometric_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize proposal-node adjacency from proposal geometry only.")
    parser.add_argument("--proposals", required=True, help="Aggregated proposal .npz from scripts/aggregate_proposals.py.")
    parser.add_argument("--mode", default="mst_knn", choices=["mst", "knn", "mst_knn"], help="Graph initialization strategy.")
    parser.add_argument("--knn", type=int, default=2, help="Euclidean nearest neighbors per proposal for knn modes.")
    parser.add_argument("--max-distance", type=float, default=0.0, help="Optional max Euclidean distance for knn edges. 0 disables.")
    parser.add_argument("--min-proposal-score", type=float, default=0.0, help="Drop proposal nodes below this objectness score. 0 disables.")
    parser.add_argument("--nms-distance", type=float, default=12.0, help="Greedy Euclidean NMS spacing between kept proposals. 0 disables.")
    parser.add_argument("--max-nodes", type=int, default=256, help="Maximum proposal nodes to keep after filtering. 0 disables.")
    parser.add_argument("--output", default="tmp/graphs/geometric_graph.npz", help="Output graph .npz.")
    args = parser.parse_args()

    proposals_path = Path(args.proposals)
    payload = np.load(proposals_path, allow_pickle=False)
    centers = payload["centers"].astype(np.float32, copy=False)
    radii = payload["radii"].astype(np.float32, copy=False) if "radii" in payload else np.zeros((centers.shape[0],), dtype=np.float32)
    scores = payload["scores"].astype(np.float32, copy=False) if "scores" in payload else np.ones((centers.shape[0],), dtype=np.float32)
    features = payload["features"].astype(np.float32, copy=False) if "features" in payload else np.zeros((centers.shape[0], 0), dtype=np.float32)
    proposal_metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}

    centers, radii, scores, features, original_indices = filter_proposals(
        centers=centers,
        radii=radii,
        scores=scores,
        features=features,
        min_score=args.min_proposal_score,
        nms_distance=args.nms_distance,
        max_nodes=args.max_nodes,
    )
    result = initialize_geometric_graph(centers=centers, mode=args.mode, knn=args.knn, max_distance=args.max_distance)

    metadata = {
        "proposals": str(proposals_path),
        "proposal_metadata": proposal_metadata,
        "init_swc": None,
        "initializer": "geometry",
        "mode": args.mode,
        "knn": args.knn,
        "max_distance": args.max_distance,
        "min_proposal_score": args.min_proposal_score,
        "nms_distance": args.nms_distance,
        "max_nodes": args.max_nodes,
        "original_nodes": int(payload["centers"].shape[0]),
        "kept_nodes": int(centers.shape[0]),
        "dropped_nodes": int(payload["centers"].shape[0] - centers.shape[0]),
        "nodes": int(centers.shape[0]),
        "edges": int(result.edges.shape[0]),
        "components": connected_component_count(centers.shape[0], result.edges),
        "mean_edge_euclidean_distance": finite_mean(result.edge_euclidean_distance),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        centers=centers,
        radii=radii,
        scores=scores,
        features=features,
        adjacency=result.adjacency,
        edges=result.edges,
        assigned_swc_indices=result.assigned_swc_indices,
        assigned_swc_ids=result.assigned_swc_ids,
        nearest_swc_distance=result.nearest_swc_distance,
        prefilter_nearest_swc_distance=np.zeros((centers.shape[0],), dtype=np.float32),
        original_indices=original_indices,
        edge_tree_distance=result.edge_tree_distance,
        edge_euclidean_distance=result.edge_euclidean_distance,
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    print(f"proposal_nodes: {centers.shape[0]}")
    print(f"original_nodes: {metadata['original_nodes']}")
    print(f"dropped_nodes: {metadata['dropped_nodes']}")
    print(f"initializer: geometry")
    print(f"mode: {args.mode}")
    print(f"edges: {result.edges.shape[0]}")
    print(f"components: {metadata['components']}")
    print(f"mean_edge_euclidean_distance: {metadata['mean_edge_euclidean_distance']:.4f}")
    print(f"output: {output_path}")
    return 0


def filter_proposals(
    centers: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    features: np.ndarray,
    min_score: float,
    nms_distance: float,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones((centers.shape[0],), dtype=bool)
    if min_score > 0.0:
        keep &= scores >= float(min_score)
    candidate_indices = np.flatnonzero(keep)
    if candidate_indices.size == 0:
        empty_indices = np.zeros((0,), dtype=np.int64)
        return centers[:0], radii[:0], scores[:0], features[:0], empty_indices

    if nms_distance > 0.0:
        candidate_indices = euclidean_nms(centers[candidate_indices], scores[candidate_indices], float(nms_distance), candidate_indices)
    else:
        order = np.argsort(-scores[candidate_indices])
        candidate_indices = candidate_indices[order]

    if max_nodes > 0 and candidate_indices.size > max_nodes:
        candidate_indices = candidate_indices[:max_nodes]
    candidate_indices = np.sort(candidate_indices).astype(np.int64, copy=False)
    return centers[candidate_indices], radii[candidate_indices], scores[candidate_indices], features[candidate_indices], candidate_indices


def euclidean_nms(centers: np.ndarray, scores: np.ndarray, min_distance: float, original_indices: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    kept: list[int] = []
    kept_centers: list[np.ndarray] = []
    for candidate in order.tolist():
        candidate_center = centers[int(candidate)]
        if kept_centers:
            distances = np.linalg.norm(np.stack(kept_centers) - candidate_center.reshape(1, 3), axis=1)
            if np.any(distances < min_distance):
                continue
        kept.append(int(original_indices[int(candidate)]))
        kept_centers.append(candidate_center)
    return np.array(kept, dtype=np.int64)


def connected_component_count(node_count: int, edges: np.ndarray) -> int:
    if node_count == 0:
        return 0
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges.tolist():
        union(int(left), int(right))
    return len({find(index) for index in range(node_count)})


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("inf")


if __name__ == "__main__":
    raise SystemExit(main())

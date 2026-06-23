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

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.vaa3d_raw import read_volume
from pointneuron.graph.initialization import (
    euclidean_distance_matrix,
    initialize_weighted_geometric_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize proposal adjacency using geometry plus foreground support.")
    parser.add_argument("--proposals", required=True, help="Aggregated proposal .npz.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--sample-index", type=int, help="Gold166 sample index. Defaults to proposal metadata.")
    parser.add_argument("--mode", default="mst", choices=["mst", "knn", "mst_knn"], help="Graph initialization strategy.")
    parser.add_argument("--knn", type=int, default=2, help="Nearest neighbors per proposal for knn modes.")
    parser.add_argument("--max-distance", type=float, default=0.0, help="Optional max weighted distance for knn edges.")
    parser.add_argument("--min-proposal-score", type=float, default=0.0, help="Drop proposal nodes below this score.")
    parser.add_argument("--nms-distance", type=float, default=12.0, help="Greedy Euclidean NMS spacing.")
    parser.add_argument("--max-nodes", type=int, default=256, help="Maximum proposal nodes after filtering. 0 disables.")
    parser.add_argument("--foreground-threshold", type=int, default=0, help="Voxel threshold used for foreground support.")
    parser.add_argument("--sample-step", type=float, default=2.0, help="Approximate voxel spacing for line samples.")
    parser.add_argument("--empty-penalty", type=float, default=8.0, help="Multiplier added as foreground support decreases.")
    parser.add_argument("--min-support", type=float, default=0.0, help="Set edge cost to infinity below this support. 0 disables.")
    parser.add_argument("--output", default="tmp/graphs/image_supported_graph.npz", help="Output graph .npz.")
    args = parser.parse_args()

    proposals_path = Path(args.proposals)
    payload = np.load(proposals_path, allow_pickle=False)
    centers = payload["centers"].astype(np.float32, copy=False)
    radii = payload["radii"].astype(np.float32, copy=False) if "radii" in payload else np.zeros((centers.shape[0],), dtype=np.float32)
    scores = payload["scores"].astype(np.float32, copy=False) if "scores" in payload else np.ones((centers.shape[0],), dtype=np.float32)
    features = payload["features"].astype(np.float32, copy=False) if "features" in payload else np.zeros((centers.shape[0], 0), dtype=np.float32)
    proposal_metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}

    sample_index = args.sample_index
    if sample_index is None:
        sample_index = int(proposal_metadata.get("sample_index", 0))
    sample = scan_gold166(args.root)[sample_index]
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    centers, radii, scores, features, original_indices = filter_proposals(
        centers=centers,
        radii=radii,
        scores=scores,
        features=features,
        min_score=args.min_proposal_score,
        nms_distance=args.nms_distance,
        max_nodes=args.max_nodes,
    )

    volume = read_volume(sample.volume_path)
    data = np.frombuffer(volume.data, dtype=volume_dtype(volume)).reshape(-1)
    foreground = data > int(args.foreground_threshold)
    support = line_support_matrix(
        centers=centers,
        foreground=foreground,
        dimensions=volume.dimensions,
        sample_step=float(args.sample_step),
    )
    distances = euclidean_distance_matrix(centers)
    weights = distances * (1.0 + float(args.empty_penalty) * (1.0 - support))
    if args.min_support > 0.0:
        weights = np.where(support >= float(args.min_support), weights, np.inf)

    result = initialize_weighted_geometric_graph(
        centers=centers,
        weights=weights,
        mode=args.mode,
        knn=args.knn,
        max_distance=args.max_distance,
    )

    edge_support = edge_values(support, result.edges)
    edge_reachable = np.ones((result.edges.shape[0],), dtype=np.uint8)
    bridge_edges = np.zeros((0, 2), dtype=np.int64)
    metadata = {
        "proposals": str(proposals_path),
        "proposal_metadata": proposal_metadata,
        "sample_index": int(sample_index),
        "sample_id": sample.sample_id,
        "init_swc": None,
        "initializer": "image_supported_geometry",
        "mode": args.mode,
        "knn": args.knn,
        "max_distance": args.max_distance,
        "min_proposal_score": args.min_proposal_score,
        "nms_distance": args.nms_distance,
        "max_nodes": args.max_nodes,
        "foreground_threshold": args.foreground_threshold,
        "sample_step": args.sample_step,
        "empty_penalty": args.empty_penalty,
        "min_support": args.min_support,
        "original_nodes": int(payload["centers"].shape[0]),
        "kept_nodes": int(centers.shape[0]),
        "dropped_nodes": int(payload["centers"].shape[0] - centers.shape[0]),
        "nodes": int(centers.shape[0]),
        "edges": int(result.edges.shape[0]),
        "components": connected_component_count(centers.shape[0], result.edges),
        "bridge_edges": 0,
        "reachable_edge_fraction": 1.0,
        "mean_edge_euclidean_distance": finite_mean(result.edge_euclidean_distance),
        "mean_edge_weight": finite_mean(result.edge_tree_distance),
        "mean_edge_support": finite_mean(edge_support),
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
        edge_geodesic_reachable=edge_reachable,
        bridge_edges=bridge_edges,
        edge_foreground_support=edge_support,
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    print(f"sample_id: {sample.sample_id}")
    print(f"proposal_nodes: {centers.shape[0]}")
    print(f"original_nodes: {metadata['original_nodes']}")
    print(f"dropped_nodes: {metadata['dropped_nodes']}")
    print("initializer: image_supported_geometry")
    print(f"mode: {args.mode}")
    print(f"edges: {result.edges.shape[0]}")
    print(f"components: {metadata['components']}")
    print(f"mean_edge_euclidean_distance: {metadata['mean_edge_euclidean_distance']:.4f}")
    print(f"mean_edge_support: {metadata['mean_edge_support']:.4f}")
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


def line_support_matrix(
    centers: np.ndarray,
    foreground: np.ndarray,
    dimensions: tuple[int, int, int, int],
    sample_step: float,
) -> np.ndarray:
    count = centers.shape[0]
    support = np.ones((count, count), dtype=np.float32)
    for left in range(count):
        for right in range(left + 1, count):
            value = line_foreground_support(centers[left], centers[right], foreground, dimensions, sample_step)
            support[left, right] = value
            support[right, left] = value
    return support


def line_foreground_support(
    start: np.ndarray,
    end: np.ndarray,
    foreground: np.ndarray,
    dimensions: tuple[int, int, int, int],
    sample_step: float,
) -> float:
    width, height, depth, _channels = dimensions
    distance = float(np.linalg.norm(end - start))
    steps = max(2, int(np.ceil(distance / max(sample_step, 1.0e-3))) + 1)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    points = start.reshape(1, 3) * (1.0 - t.reshape(-1, 1)) + end.reshape(1, 3) * t.reshape(-1, 1)
    xyz = np.rint(points).astype(np.int64)
    in_bounds = (
        (xyz[:, 0] >= 0)
        & (xyz[:, 0] < width)
        & (xyz[:, 1] >= 0)
        & (xyz[:, 1] < height)
        & (xyz[:, 2] >= 0)
        & (xyz[:, 2] < depth)
    )
    if not np.any(in_bounds):
        return 0.0
    xyz = xyz[in_bounds]
    flat = xyz[:, 2] * width * height + xyz[:, 1] * width + xyz[:, 0]
    return float(np.count_nonzero(foreground[flat]) / flat.shape[0])


def volume_dtype(volume) -> np.dtype:
    if volume.header.datatype == 1:
        return np.dtype(np.uint8)
    if volume.header.datatype == 2:
        if volume.header.endian == "L":
            return np.dtype("<u2")
        if volume.header.endian == "B":
            return np.dtype(">u2")
    raise NotImplementedError(f"Vaa3D datatype {volume.header.datatype} is not supported yet")


def edge_values(matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix[edges[:, 0], edges[:, 1]].astype(np.float32, copy=False)


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

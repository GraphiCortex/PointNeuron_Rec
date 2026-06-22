from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

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
    parser = argparse.ArgumentParser(description="Initialize proposal adjacency from foreground geodesic paths.")
    parser.add_argument("--proposals", required=True, help="Aggregated proposal .npz.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--sample-index", type=int, help="Gold166 sample index. Defaults to proposal metadata.")
    parser.add_argument("--mode", default="mst", choices=["mst", "knn", "mst_knn"], help="Graph initialization strategy.")
    parser.add_argument("--knn", type=int, default=2, help="Nearest neighbors per proposal for knn modes.")
    parser.add_argument("--max-distance", type=float, default=0.0, help="Optional max geodesic distance for knn edges.")
    parser.add_argument("--min-proposal-score", type=float, default=0.0, help="Drop proposal nodes below this score.")
    parser.add_argument("--nms-distance", type=float, default=12.0, help="Greedy Euclidean NMS spacing.")
    parser.add_argument("--max-nodes", type=int, default=256, help="Maximum proposal nodes after filtering. 0 disables.")
    parser.add_argument("--foreground-threshold", type=int, help="Voxel threshold. Defaults to proposal metadata threshold.")
    parser.add_argument("--intensity-penalty", type=float, default=2.0, help="Extra cost for dim foreground voxels.")
    parser.add_argument("--disconnected-multiplier", type=float, default=100.0, help="Euclidean fallback multiplier for disconnected pairs.")
    parser.add_argument("--output", default="tmp/graphs/geodesic_graph.npz", help="Output graph .npz.")
    args = parser.parse_args()

    proposals_path = Path(args.proposals)
    payload = np.load(proposals_path, allow_pickle=False)
    centers = payload["centers"].astype(np.float32, copy=False)
    radii = payload["radii"].astype(np.float32, copy=False) if "radii" in payload else np.ones((centers.shape[0],), dtype=np.float32)
    scores = payload["scores"].astype(np.float32, copy=False) if "scores" in payload else np.ones((centers.shape[0],), dtype=np.float32)
    features = payload["features"].astype(np.float32, copy=False) if "features" in payload else np.zeros((centers.shape[0], 0), dtype=np.float32)
    proposal_metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}

    sample_index = args.sample_index
    if sample_index is None:
        sample_index = int(proposal_metadata.get("sample_index", 0))
    sample = scan_gold166(args.root)[sample_index]
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    threshold = args.foreground_threshold
    if threshold is None:
        threshold = int(proposal_metadata.get("threshold", 0))

    centers, radii, scores, features, original_indices = filter_proposals(
        centers=centers,
        radii=radii,
        scores=scores,
        features=features,
        min_score=args.min_proposal_score,
        nms_distance=args.nms_distance,
        max_nodes=args.max_nodes,
    )
    if centers.shape[0] == 0:
        raise ValueError("No proposal nodes remain after filtering")

    volume = read_volume(sample.volume_path)
    dtype = volume_dtype(volume)
    data = np.frombuffer(volume.data, dtype=dtype).reshape(-1)
    width, height, depth, channels = volume.dimensions
    if channels != 1:
        raise NotImplementedError(f"Expected single-channel volume, got {channels}")

    foreground_flats = np.flatnonzero(data > int(threshold)).astype(np.int64, copy=False)
    if foreground_flats.size == 0:
        raise ValueError(f"No foreground voxels above threshold {threshold}")
    foreground_coords = flats_to_xyz(foreground_flats, width=width, height=height).astype(np.float32, copy=False)
    snapped_local, snap_distances = snap_centers_to_foreground(centers, foreground_coords)

    sparse_graph = build_foreground_graph(
        foreground_flats=foreground_flats,
        data=data,
        dimensions=volume.dimensions,
        threshold=int(threshold),
        intensity_penalty=float(args.intensity_penalty),
    )
    distances, predecessors = dijkstra(
        sparse_graph,
        directed=False,
        indices=snapped_local,
        return_predecessors=True,
    )
    proposal_geodesic = distances[:, snapped_local].astype(np.float32, copy=False)
    reachable = np.isfinite(proposal_geodesic)
    euclidean = euclidean_distance_matrix(centers)
    weights = proposal_geodesic.copy()
    weights[~reachable] = euclidean[~reachable] * float(args.disconnected_multiplier)
    np.fill_diagonal(weights, np.inf)

    result = initialize_weighted_geometric_graph(
        centers=centers,
        weights=weights,
        mode=args.mode,
        knn=args.knn,
        max_distance=args.max_distance,
    )

    path_points, path_offsets, edge_reachable = reconstruct_edge_paths(
        edges=result.edges,
        centers=centers,
        foreground_coords=foreground_coords,
        snapped_local=snapped_local,
        predecessors=predecessors,
    )
    edge_geodesic = edge_values(weights, result.edges)

    metadata = {
        "proposals": str(proposals_path),
        "proposal_metadata": proposal_metadata,
        "sample_index": int(sample_index),
        "sample_id": sample.sample_id,
        "init_swc": None,
        "initializer": "foreground_geodesic",
        "mode": args.mode,
        "knn": args.knn,
        "max_distance": args.max_distance,
        "min_proposal_score": args.min_proposal_score,
        "nms_distance": args.nms_distance,
        "max_nodes": args.max_nodes,
        "foreground_threshold": int(threshold),
        "foreground_voxels": int(foreground_flats.size),
        "intensity_penalty": args.intensity_penalty,
        "disconnected_multiplier": args.disconnected_multiplier,
        "original_nodes": int(payload["centers"].shape[0]),
        "kept_nodes": int(centers.shape[0]),
        "dropped_nodes": int(payload["centers"].shape[0] - centers.shape[0]),
        "nodes": int(centers.shape[0]),
        "edges": int(result.edges.shape[0]),
        "components": connected_component_count(centers.shape[0], result.edges),
        "mean_snap_distance": finite_mean(snap_distances),
        "mean_edge_euclidean_distance": finite_mean(result.edge_euclidean_distance),
        "mean_edge_geodesic_distance": finite_mean(edge_geodesic),
        "reachable_edge_fraction": float(np.count_nonzero(edge_reachable) / max(edge_reachable.size, 1)),
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
        nearest_swc_distance=snap_distances.astype(np.float32, copy=False),
        prefilter_nearest_swc_distance=np.zeros((centers.shape[0],), dtype=np.float32),
        original_indices=original_indices,
        edge_tree_distance=edge_geodesic.astype(np.float32, copy=False),
        edge_euclidean_distance=result.edge_euclidean_distance,
        edge_geodesic_reachable=edge_reachable.astype(np.uint8, copy=False),
        edge_path_points=path_points.astype(np.float32, copy=False),
        edge_path_offsets=path_offsets.astype(np.int64, copy=False),
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    print(f"sample_id: {sample.sample_id}")
    print(f"proposal_nodes: {centers.shape[0]}")
    print(f"foreground_threshold: {threshold}")
    print(f"foreground_voxels: {foreground_flats.size}")
    print("initializer: foreground_geodesic")
    print(f"mode: {args.mode}")
    print(f"edges: {result.edges.shape[0]}")
    print(f"components: {metadata['components']}")
    print(f"mean_snap_distance: {metadata['mean_snap_distance']:.4f}")
    print(f"mean_edge_euclidean_distance: {metadata['mean_edge_euclidean_distance']:.4f}")
    print(f"mean_edge_geodesic_distance: {metadata['mean_edge_geodesic_distance']:.4f}")
    print(f"reachable_edge_fraction: {metadata['reachable_edge_fraction']:.4f}")
    print(f"output: {output_path}")
    return 0


def build_foreground_graph(
    foreground_flats: np.ndarray,
    data: np.ndarray,
    dimensions: tuple[int, int, int, int],
    threshold: int,
    intensity_penalty: float,
):
    width, height, depth, _channels = dimensions
    coords = flats_to_xyz(foreground_flats, width=width, height=height)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    max_value = float(np.iinfo(data.dtype).max)
    denom = max(max_value - float(threshold), 1.0)
    intensities = np.clip((data[foreground_flats].astype(np.float32) - float(threshold)) / denom, 0.0, 1.0)

    offsets = [
        (dx, dy, dz)
        for dz in (0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
        and (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0))
    ]
    local_indices = np.arange(foreground_flats.size, dtype=np.int64)
    plane = width * height
    for dx, dy, dz in offsets:
        mask = (
            (x + dx >= 0)
            & (x + dx < width)
            & (y + dy >= 0)
            & (y + dy < height)
            & (z + dz >= 0)
            & (z + dz < depth)
        )
        if not np.any(mask):
            continue
        source_local = local_indices[mask]
        neighbor_flat = foreground_flats[mask] + dx + dy * width + dz * plane
        positions = np.searchsorted(foreground_flats, neighbor_flat)
        in_range = positions < foreground_flats.size
        valid = np.zeros_like(in_range, dtype=bool)
        valid[in_range] = foreground_flats[positions[in_range]] == neighbor_flat[in_range]
        if not np.any(valid):
            continue
        source_local = source_local[valid]
        target_local = positions[valid].astype(np.int64, copy=False)
        step = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        mean_intensity = (intensities[source_local] + intensities[target_local]) * 0.5
        edge_weight = step * (1.0 + float(intensity_penalty) * (1.0 - mean_intensity))
        rows.extend([source_local, target_local])
        cols.extend([target_local, source_local])
        weights.extend([edge_weight, edge_weight])

    if not rows:
        raise ValueError("Foreground graph has no edges")
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    weight = np.concatenate(weights).astype(np.float32, copy=False)
    return coo_matrix((weight, (row, col)), shape=(foreground_flats.size, foreground_flats.size)).tocsr()


def reconstruct_edge_paths(
    edges: np.ndarray,
    centers: np.ndarray,
    foreground_coords: np.ndarray,
    snapped_local: np.ndarray,
    predecessors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_arrays: list[np.ndarray] = []
    offsets = [0]
    reachable = []
    for left, right in edges.tolist():
        source_row = int(left)
        target = int(snapped_local[int(right)])
        local_path = predecessor_path(predecessors[source_row], int(snapped_local[int(left)]), target)
        if local_path is None:
            points = straight_line_points(centers[int(left)], centers[int(right)])
            reachable.append(False)
        else:
            points = foreground_coords[np.array(local_path, dtype=np.int64)]
            reachable.append(True)
        path_arrays.append(points.astype(np.float32, copy=False))
        offsets.append(offsets[-1] + points.shape[0])
    if path_arrays:
        path_points = np.concatenate(path_arrays, axis=0)
    else:
        path_points = np.zeros((0, 3), dtype=np.float32)
    return path_points, np.array(offsets, dtype=np.int64), np.array(reachable, dtype=bool)


def predecessor_path(predecessors: np.ndarray, source: int, target: int) -> list[int] | None:
    if source == target:
        return [source]
    path = [target]
    current = target
    limit = predecessors.shape[0] + 1
    for _ in range(limit):
        parent = int(predecessors[current])
        if parent < 0:
            return None
        path.append(parent)
        if parent == source:
            path.reverse()
            return path
        current = parent
    return None


def straight_line_points(start: np.ndarray, end: np.ndarray, step: float = 1.0) -> np.ndarray:
    distance = float(np.linalg.norm(end - start))
    steps = max(2, int(np.ceil(distance / max(step, 1.0e-3))) + 1)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return start.reshape(1, 3) * (1.0 - t.reshape(-1, 1)) + end.reshape(1, 3) * t.reshape(-1, 1)


def snap_centers_to_foreground(centers: np.ndarray, foreground_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(foreground_coords)
    distances, indices = tree.query(centers, k=1)
    return indices.astype(np.int64, copy=False), distances.astype(np.float32, copy=False)


def flats_to_xyz(flats: np.ndarray, width: int, height: int) -> np.ndarray:
    plane = width * height
    z = flats // plane
    offset = flats - z * plane
    y = offset // width
    x = offset - y * width
    return np.stack([x, y, z], axis=1).astype(np.int64, copy=False)


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


def volume_dtype(volume) -> np.dtype:
    if volume.header.datatype == 1:
        return np.dtype(np.uint8)
    if volume.header.datatype == 2:
        if volume.header.endian == "L":
            return np.dtype("<u2")
        if volume.header.endian == "B":
            return np.dtype(">u2")
    raise NotImplementedError(f"Vaa3D datatype {volume.header.datatype} is not supported yet")


if __name__ == "__main__":
    raise SystemExit(main())

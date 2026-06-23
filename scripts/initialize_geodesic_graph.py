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
    parser.add_argument(
        "--max-foreground-voxels",
        type=int,
        default=0,
        help="If >0, raise the foreground threshold until the geodesic graph has at most this many voxels.",
    )
    parser.add_argument("--intensity-penalty", type=float, default=2.0, help="Extra cost for dim foreground voxels.")
    parser.add_argument("--disconnected-multiplier", type=float, default=100.0, help="Euclidean fallback multiplier for disconnected pairs.")
    parser.add_argument("--candidate-k", type=int, default=0, help="Only allow each node's K nearest Euclidean proposal candidates. 0 disables.")
    parser.add_argument("--max-euclidean-distance", type=float, default=0.0, help="Drop proposal candidate pairs farther than this Euclidean distance. 0 disables.")
    parser.add_argument("--max-geodesic-ratio", type=float, default=0.0, help="Drop reachable pairs whose geodesic/euclidean ratio exceeds this value. 0 disables.")
    parser.add_argument("--allow-unreachable-fallback", action="store_true", help="Allow disconnected pairs with fallback Euclidean cost.")
    parser.add_argument("--bridge-components", action="store_true", help="After constrained MST, add minimum-cost reachable edges to connect components.")
    parser.add_argument("--bridge-max-geodesic-ratio", type=float, default=0.0, help="Max geodesic/euclidean ratio for bridge edges. 0 disables.")
    parser.add_argument("--bridge-allow-unreachable-fallback", action="store_true", help="Allow fallback straight bridge edges only during component bridging.")
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

    requested_threshold = args.foreground_threshold
    if requested_threshold is None:
        requested_threshold = int(proposal_metadata.get("threshold", 0))

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

    threshold, threshold_was_adapted = choose_foreground_threshold(
        data=data,
        requested_threshold=int(requested_threshold),
        max_foreground_voxels=int(args.max_foreground_voxels),
    )
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
    base_weights = weights.copy()
    candidate_mask = proposal_candidate_mask(
        euclidean=euclidean,
        weights=weights,
        reachable=reachable,
        candidate_k=args.candidate_k,
        max_euclidean_distance=args.max_euclidean_distance,
        max_geodesic_ratio=args.max_geodesic_ratio,
        allow_unreachable_fallback=args.allow_unreachable_fallback,
    )
    weights = np.where(candidate_mask, weights, np.inf)

    result = initialize_weighted_geometric_graph(
        centers=centers,
        weights=weights,
        mode=args.mode,
        knn=args.knn,
        max_distance=args.max_distance,
    )
    final_edges = result.edges
    bridge_edges = np.zeros((0, 2), dtype=np.int64)
    if args.bridge_components:
        bridge_edges = component_bridge_edges(
            node_count=centers.shape[0],
            existing_edges=final_edges,
            weights=base_weights,
            euclidean=euclidean,
            reachable=reachable,
            max_geodesic_ratio=args.bridge_max_geodesic_ratio,
            allow_unreachable_fallback=args.bridge_allow_unreachable_fallback,
        )
        final_edges = merge_edges(final_edges, bridge_edges)
    final_adjacency = edges_to_adjacency(centers.shape[0], final_edges)
    final_edge_euclidean = edge_euclidean_distances(centers, final_edges)

    path_points, path_offsets, edge_reachable = reconstruct_edge_paths(
        edges=final_edges,
        centers=centers,
        foreground_coords=foreground_coords,
        snapped_local=snapped_local,
        predecessors=predecessors,
    )
    edge_geodesic = edge_values(base_weights, final_edges)

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
        "requested_foreground_threshold": int(requested_threshold),
        "foreground_threshold_was_adapted": bool(threshold_was_adapted),
        "max_foreground_voxels": int(args.max_foreground_voxels),
        "foreground_voxels": int(foreground_flats.size),
        "intensity_penalty": args.intensity_penalty,
        "disconnected_multiplier": args.disconnected_multiplier,
        "candidate_k": args.candidate_k,
        "max_euclidean_distance": args.max_euclidean_distance,
        "max_geodesic_ratio": args.max_geodesic_ratio,
        "allow_unreachable_fallback": args.allow_unreachable_fallback,
        "bridge_components": args.bridge_components,
        "bridge_max_geodesic_ratio": args.bridge_max_geodesic_ratio,
        "bridge_allow_unreachable_fallback": args.bridge_allow_unreachable_fallback,
        "bridge_edges": int(bridge_edges.shape[0]),
        "eligible_candidate_pairs": int(np.count_nonzero(np.triu(candidate_mask, k=1))),
        "original_nodes": int(payload["centers"].shape[0]),
        "kept_nodes": int(centers.shape[0]),
        "dropped_nodes": int(payload["centers"].shape[0] - centers.shape[0]),
        "nodes": int(centers.shape[0]),
        "edges": int(final_edges.shape[0]),
        "components": connected_component_count(centers.shape[0], final_edges),
        "mean_snap_distance": finite_mean(snap_distances),
        "mean_edge_euclidean_distance": finite_mean(final_edge_euclidean),
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
        adjacency=final_adjacency,
        edges=final_edges,
        assigned_swc_indices=result.assigned_swc_indices,
        assigned_swc_ids=result.assigned_swc_ids,
        nearest_swc_distance=snap_distances.astype(np.float32, copy=False),
        prefilter_nearest_swc_distance=np.zeros((centers.shape[0],), dtype=np.float32),
        original_indices=original_indices,
        edge_tree_distance=edge_geodesic.astype(np.float32, copy=False),
        edge_euclidean_distance=final_edge_euclidean.astype(np.float32, copy=False),
        edge_geodesic_reachable=edge_reachable.astype(np.uint8, copy=False),
        bridge_edges=bridge_edges.astype(np.int64, copy=False),
        edge_path_points=path_points.astype(np.float32, copy=False),
        edge_path_offsets=path_offsets.astype(np.int64, copy=False),
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    print(f"sample_id: {sample.sample_id}")
    print(f"proposal_nodes: {centers.shape[0]}")
    print(f"foreground_threshold: {threshold}")
    print(f"requested_foreground_threshold: {requested_threshold}")
    print(f"foreground_threshold_was_adapted: {threshold_was_adapted}")
    print(f"max_foreground_voxels: {args.max_foreground_voxels}")
    print(f"foreground_voxels: {foreground_flats.size}")
    print("initializer: foreground_geodesic")
    print(f"mode: {args.mode}")
    print(f"edges: {final_edges.shape[0]}")
    print(f"bridge_edges: {bridge_edges.shape[0]}")
    print(f"components: {metadata['components']}")
    print(f"mean_snap_distance: {metadata['mean_snap_distance']:.4f}")
    print(f"mean_edge_euclidean_distance: {metadata['mean_edge_euclidean_distance']:.4f}")
    print(f"mean_edge_geodesic_distance: {metadata['mean_edge_geodesic_distance']:.4f}")
    print(f"reachable_edge_fraction: {metadata['reachable_edge_fraction']:.4f}")
    print(f"eligible_candidate_pairs: {metadata['eligible_candidate_pairs']}")
    print(f"output: {output_path}")
    return 0


def choose_foreground_threshold(data: np.ndarray, requested_threshold: int, max_foreground_voxels: int) -> tuple[int, bool]:
    requested_threshold = int(requested_threshold)
    if max_foreground_voxels <= 0:
        return requested_threshold, False

    requested_count = int(np.count_nonzero(data > requested_threshold))
    if requested_count <= int(max_foreground_voxels):
        return requested_threshold, False

    max_value = int(data.max()) if data.size else requested_threshold
    if max_value <= requested_threshold:
        return requested_threshold, False

    histogram = np.bincount(data, minlength=max_value + 1)
    greater_than = np.cumsum(histogram[::-1])[::-1] - histogram
    for threshold in range(requested_threshold + 1, max_value + 1):
        count = int(greater_than[threshold])
        if 0 < count <= int(max_foreground_voxels):
            return threshold, True

    # Keep at least the brightest non-empty foreground instead of returning an empty graph.
    nonzero_values = np.flatnonzero(histogram)
    brighter = nonzero_values[nonzero_values > requested_threshold]
    if brighter.size:
        return max(int(brighter[-1]) - 1, requested_threshold), True
    return requested_threshold, False


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


def proposal_candidate_mask(
    euclidean: np.ndarray,
    weights: np.ndarray,
    reachable: np.ndarray,
    candidate_k: int,
    max_euclidean_distance: float,
    max_geodesic_ratio: float,
    allow_unreachable_fallback: bool,
) -> np.ndarray:
    count = euclidean.shape[0]
    mask = np.ones((count, count), dtype=bool)
    np.fill_diagonal(mask, False)

    if candidate_k > 0:
        local = np.zeros((count, count), dtype=bool)
        for index in range(count):
            distances = euclidean[index].copy()
            distances[index] = np.inf
            order = np.argsort(distances)[:candidate_k]
            local[index, order] = True
        mask &= local | local.T

    if max_euclidean_distance > 0.0:
        mask &= euclidean <= float(max_euclidean_distance)

    if max_geodesic_ratio > 0.0:
        denominator = np.maximum(euclidean, 1.0e-3)
        ratio = weights / denominator
        mask &= (~reachable) | (ratio <= float(max_geodesic_ratio))

    if not allow_unreachable_fallback:
        mask &= reachable

    return mask | mask.T


def component_bridge_edges(
    node_count: int,
    existing_edges: np.ndarray,
    weights: np.ndarray,
    euclidean: np.ndarray,
    reachable: np.ndarray,
    max_geodesic_ratio: float,
    allow_unreachable_fallback: bool,
) -> np.ndarray:
    parent = list(range(node_count))
    rank = [0] * node_count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1
        return True

    for left, right in existing_edges.tolist():
        union(int(left), int(right))

    candidates: list[tuple[float, int, int]] = []
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if find(left) == find(right):
                continue
            if not allow_unreachable_fallback and not bool(reachable[left, right]):
                continue
            weight = float(weights[left, right])
            if not np.isfinite(weight):
                continue
            if max_geodesic_ratio > 0.0 and bool(reachable[left, right]):
                ratio = weight / max(float(euclidean[left, right]), 1.0e-3)
                if ratio > float(max_geodesic_ratio):
                    continue
            candidates.append((weight, left, right))
    candidates.sort(key=lambda item: item[0])

    bridges: list[tuple[int, int]] = []
    for _weight, left, right in candidates:
        if union(left, right):
            bridges.append((left, right))
            if len({find(index) for index in range(node_count)}) == 1:
                break
    return np.array(bridges, dtype=np.int64).reshape(-1, 2) if bridges else np.zeros((0, 2), dtype=np.int64)


def merge_edges(left_edges: np.ndarray, right_edges: np.ndarray) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for left, right in left_edges.tolist() + right_edges.tolist():
        if int(left) == int(right):
            continue
        edge_set.add(tuple(sorted((int(left), int(right)))))
    return np.array(sorted(edge_set), dtype=np.int64).reshape(-1, 2) if edge_set else np.zeros((0, 2), dtype=np.int64)


def edges_to_adjacency(node_count: int, edges: np.ndarray) -> np.ndarray:
    adjacency = np.zeros((node_count, node_count), dtype=np.uint8)
    if edges.size:
        adjacency[edges[:, 0], edges[:, 1]] = 1
        adjacency[edges[:, 1], edges[:, 0]] = 1
    return adjacency


def edge_euclidean_distances(centers: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(centers[edges[:, 0]] - centers[edges[:, 1]], axis=1).astype(np.float32, copy=False)


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

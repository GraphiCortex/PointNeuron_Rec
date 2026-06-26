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

HYBRID_COVERAGE_FILL_FLOOR = 40
CONNECTED_COVERAGE_POOL_LIMIT = 512


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
    parser.add_argument("--nms-mode", default="distance", choices=["distance", "sphere"], help="Proposal-node NMS mode.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Sphere IoU threshold for --nms-mode sphere.")
    parser.add_argument("--max-nodes", type=int, default=256, help="Maximum proposal nodes after filtering. 0 disables.")
    parser.add_argument(
        "--selection-mode",
        default="score_nms",
        choices=["score_nms", "coverage_nms", "hybrid_nms", "structural_nms", "connected_coverage_nms"],
        help="How to truncate NMS survivors when --max-nodes is reached.",
    )
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
        nms_mode=args.nms_mode,
        iou_threshold=args.iou_threshold,
        max_nodes=args.max_nodes,
        selection_mode=args.selection_mode,
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
    if args.selection_mode == "connected_coverage_nms" and args.max_nodes > 0 and centers.shape[0] > args.max_nodes:
        local_keep = connected_coverage_selection(
            centers=centers,
            scores=scores,
            geodesic=proposal_geodesic,
            euclidean=euclidean,
            max_nodes=int(args.max_nodes),
            max_geodesic_ratio=float(args.max_geodesic_ratio),
        )
        centers = centers[local_keep]
        radii = radii[local_keep]
        scores = scores[local_keep]
        features = features[local_keep]
        original_indices = original_indices[local_keep]
        snapped_local = snapped_local[local_keep]
        snap_distances = snap_distances[local_keep]
        distances = distances[local_keep]
        predecessors = predecessors[local_keep]
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
        "nms_mode": args.nms_mode,
        "iou_threshold": args.iou_threshold,
        "max_nodes": args.max_nodes,
        "selection_mode": args.selection_mode,
        "foreground_threshold": int(threshold),
        "requested_foreground_threshold": int(requested_threshold),
        "foreground_threshold_was_adapted": bool(threshold_was_adapted),
        "max_foreground_voxels": int(args.max_foreground_voxels),
        "foreground_voxels": int(foreground_flats.size),
        "foreground_cap_satisfied": bool(args.max_foreground_voxels <= 0 or foreground_flats.size <= int(args.max_foreground_voxels)),
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
    print(f"foreground_cap_satisfied: {metadata['foreground_cap_satisfied']}")
    print("initializer: foreground_geodesic")
    print(f"mode: {args.mode}")
    print(f"selection_mode: {args.selection_mode}")
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
    nms_mode: str,
    iou_threshold: float,
    max_nodes: int,
    selection_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones((centers.shape[0],), dtype=bool)
    if min_score > 0.0:
        keep &= scores >= float(min_score)
    candidate_indices = np.flatnonzero(keep)
    if candidate_indices.size == 0:
        empty_indices = np.zeros((0,), dtype=np.int64)
        return centers[:0], radii[:0], scores[:0], features[:0], empty_indices

    if selection_mode == "coverage_nms":
        order = np.argsort(-scores[candidate_indices])
        candidate_indices = candidate_indices[order]
    elif selection_mode == "structural_nms":
        structural_scores = structural_selection_scores(centers[candidate_indices], scores[candidate_indices])
        if nms_distance > 0.0 or nms_mode == "sphere":
            candidate_indices = nms_proposals(
                centers=centers[candidate_indices],
                radii=radii[candidate_indices],
                scores=structural_scores,
                nms_mode=nms_mode,
                nms_distance=float(nms_distance),
                iou_threshold=float(iou_threshold),
                original_indices=candidate_indices,
            )
        else:
            order = np.argsort(-structural_scores)
            candidate_indices = candidate_indices[order]
    elif selection_mode == "hybrid_nms":
        if nms_distance > 0.0 or nms_mode == "sphere":
            score_indices = nms_proposals(
                centers=centers[candidate_indices],
                radii=radii[candidate_indices],
                scores=scores[candidate_indices],
                nms_mode=nms_mode,
                nms_distance=float(nms_distance),
                iou_threshold=float(iou_threshold),
                original_indices=candidate_indices,
            )
        else:
            order = np.argsort(-scores[candidate_indices])
            score_indices = candidate_indices[order]

        fill_floor = min(int(HYBRID_COVERAGE_FILL_FLOOR), int(max_nodes)) if max_nodes > 0 else 0
        if max_nodes > 0 and score_indices.size > max_nodes:
            candidate_indices = score_indices[:max_nodes]
        elif max_nodes > 0 and score_indices.size < fill_floor and candidate_indices.size > score_indices.size:
            candidate_indices = coverage_truncate(
                centers=centers,
                scores=scores,
                candidate_indices=candidate_indices,
                max_nodes=max_nodes,
                seed_indices=score_indices,
            )
        else:
            candidate_indices = score_indices
    elif selection_mode == "connected_coverage_nms":
        order = np.argsort(-scores[candidate_indices])
        candidate_indices = candidate_indices[order]
    elif nms_distance > 0.0 or nms_mode == "sphere":
        candidate_indices = nms_proposals(
            centers=centers[candidate_indices],
            radii=radii[candidate_indices],
            scores=scores[candidate_indices],
            nms_mode=nms_mode,
            nms_distance=float(nms_distance),
            iou_threshold=float(iou_threshold),
            original_indices=candidate_indices,
        )
    else:
        order = np.argsort(-scores[candidate_indices])
        candidate_indices = candidate_indices[order]

    if max_nodes > 0 and candidate_indices.size > max_nodes and selection_mode == "connected_coverage_nms":
        pool_limit = min(candidate_indices.size, max(int(max_nodes), min(CONNECTED_COVERAGE_POOL_LIMIT, int(max_nodes) * 4)))
        candidate_indices = coverage_truncate(
            centers=centers,
            scores=scores,
            candidate_indices=candidate_indices,
            max_nodes=pool_limit,
        )
    elif max_nodes > 0 and candidate_indices.size > max_nodes:
        if selection_mode in {"coverage_nms", "structural_nms"}:
            candidate_indices = coverage_truncate(
                centers=centers,
                scores=scores,
                candidate_indices=candidate_indices,
                max_nodes=max_nodes,
            )
        else:
            candidate_indices = candidate_indices[:max_nodes]
    candidate_indices = np.sort(candidate_indices).astype(np.int64, copy=False)
    return centers[candidate_indices], radii[candidate_indices], scores[candidate_indices], features[candidate_indices], candidate_indices


def connected_coverage_selection(
    centers: np.ndarray,
    scores: np.ndarray,
    geodesic: np.ndarray,
    euclidean: np.ndarray,
    max_nodes: int,
    max_geodesic_ratio: float,
) -> np.ndarray:
    if centers.shape[0] <= max_nodes:
        return np.arange(centers.shape[0], dtype=np.int64)

    connectivity = np.isfinite(geodesic)
    np.fill_diagonal(connectivity, False)
    if max_geodesic_ratio > 0.0:
        ratio = geodesic / np.maximum(euclidean, 1.0e-3)
        connectivity &= ratio <= float(max_geodesic_ratio)
    connectivity |= connectivity.T

    components = connected_components_from_mask(connectivity)
    components.sort(key=lambda component: (-component.size, -float(scores[component].mean()) if component.size else 0.0))
    eligible = [component for component in components if component.size >= 3]
    if not eligible:
        eligible = components

    selected: list[int] = []
    selected_set: set[int] = set()
    remaining = int(max_nodes)
    remaining_source = int(sum(component.size for component in eligible))
    for component in eligible:
        if remaining <= 0:
            break
        if remaining_source <= 0:
            budget = remaining
        else:
            budget = int(round(float(max_nodes) * float(component.size) / float(remaining_source)))
        budget = max(1, min(int(component.size), budget, remaining))
        picked = coverage_truncate(
            centers=centers,
            scores=scores,
            candidate_indices=component.astype(np.int64, copy=False),
            max_nodes=budget,
        )
        for index in picked.tolist():
            if int(index) not in selected_set:
                selected.append(int(index))
                selected_set.add(int(index))
        remaining = int(max_nodes) - len(selected)
        remaining_source -= int(component.size)

    if len(selected) < int(max_nodes):
        seed_indices = np.array(selected, dtype=np.int64) if selected else None
        filled = coverage_truncate(
            centers=centers,
            scores=scores,
            candidate_indices=np.arange(centers.shape[0], dtype=np.int64),
            max_nodes=int(max_nodes),
            seed_indices=seed_indices,
        )
        selected = [int(index) for index in filled.tolist()]

    return np.array(selected[: int(max_nodes)], dtype=np.int64)


def connected_components_from_mask(connectivity: np.ndarray) -> list[np.ndarray]:
    node_count = int(connectivity.shape[0])
    seen = np.zeros((node_count,), dtype=bool)
    components: list[np.ndarray] = []
    for start in range(node_count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = np.flatnonzero(connectivity[node])
            for neighbor in neighbors.tolist():
                if not seen[int(neighbor)]:
                    seen[int(neighbor)] = True
                    stack.append(int(neighbor))
        components.append(np.array(component, dtype=np.int64))
    return components


def structural_selection_scores(centers: np.ndarray, scores: np.ndarray, neighbor_count: int = 12) -> np.ndarray:
    if centers.shape[0] == 0:
        return scores.astype(np.float32, copy=True)
    if centers.shape[0] <= 2:
        return scores.astype(np.float32, copy=True)

    k = min(int(neighbor_count) + 1, int(centers.shape[0]))
    tree = cKDTree(centers)
    distances, indices = tree.query(centers, k=k)
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float32)
    neighbor_indices = np.asarray(indices[:, 1:], dtype=np.int64)

    mean_neighbor_distance = neighbor_distances.mean(axis=1)
    isolation = robust_normalize(mean_neighbor_distance)

    anisotropy = np.zeros((centers.shape[0],), dtype=np.float32)
    branchiness = np.zeros((centers.shape[0],), dtype=np.float32)
    for row, neighbors in enumerate(neighbor_indices):
        vectors = centers[neighbors] - centers[row].reshape(1, 3)
        if vectors.shape[0] < 3:
            continue
        covariance = np.cov(vectors.T)
        eigenvalues = np.linalg.eigvalsh(covariance).astype(np.float32, copy=False)
        total = float(np.maximum(eigenvalues.sum(), 1.0e-6))
        eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
        anisotropy[row] = float(eigenvalues[0] / total)
        branchiness[row] = float((eigenvalues[1] + eigenvalues[2]) / total)

    score_term = robust_normalize(scores.astype(np.float32, copy=False))
    line_term = robust_normalize(anisotropy)
    branch_term = robust_normalize(branchiness)
    # Confidence keeps false positives down; geometry promotes tips, sparse paths, and junction-like neighborhoods.
    structural = 0.50 * score_term + 0.25 * isolation + 0.15 * line_term + 0.10 * branch_term
    return structural.astype(np.float32, copy=False)


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    if values.size == 0:
        return values.copy()
    lo = float(np.quantile(values, 0.05))
    hi = float(np.quantile(values, 0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def coverage_truncate(
    centers: np.ndarray,
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    max_nodes: int,
    seed_indices: np.ndarray | None = None,
) -> np.ndarray:
    if max_nodes <= 0:
        return candidate_indices
    if seed_indices is None and candidate_indices.size <= max_nodes:
        return candidate_indices
    if candidate_indices.size == 0:
        return candidate_indices

    local_centers = centers[candidate_indices].astype(np.float32, copy=False)
    local_scores = scores[candidate_indices].astype(np.float32, copy=False)
    selected: list[int] = []
    selected_mask = np.zeros((candidate_indices.size,), dtype=bool)
    if seed_indices is not None and seed_indices.size > 0:
        local_by_global = {int(global_index): local_index for local_index, global_index in enumerate(candidate_indices.tolist())}
        for global_index in seed_indices.tolist():
            local_index = local_by_global.get(int(global_index))
            if local_index is None or selected_mask[local_index]:
                continue
            selected.append(local_index)
            selected_mask[local_index] = True
            if len(selected) >= int(max_nodes):
                return candidate_indices[np.array(selected[: int(max_nodes)], dtype=np.int64)]

    if selected:
        selected_centers = local_centers[np.array(selected, dtype=np.int64)]
        diff = local_centers[:, None, :] - selected_centers[None, :, :]
        min_distance2 = np.sum(diff * diff, axis=2).min(axis=1)
        min_distance2[selected_mask] = -np.inf
    else:
        first = int(np.argmax(local_scores))
        selected.append(first)
        selected_mask[first] = True
        min_distance2 = np.sum((local_centers - local_centers[first].reshape(1, 3)) ** 2, axis=1)
        min_distance2[first] = -np.inf

    while len(selected) < int(max_nodes):
        next_index = int(np.argmax(min_distance2))
        if not np.isfinite(min_distance2[next_index]) or min_distance2[next_index] < 0.0:
            break
        selected.append(next_index)
        selected_mask[next_index] = True
        distance2 = np.sum((local_centers - local_centers[next_index].reshape(1, 3)) ** 2, axis=1)
        min_distance2 = np.minimum(min_distance2, distance2)
        min_distance2[selected_mask] = -np.inf

    return candidate_indices[np.array(selected, dtype=np.int64)]


def nms_proposals(
    centers: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    nms_mode: str,
    nms_distance: float,
    iou_threshold: float,
    original_indices: np.ndarray,
) -> np.ndarray:
    if nms_mode == "sphere":
        return sphere_iou_nms(
            centers=centers,
            radii=radii,
            scores=scores,
            iou_threshold=float(iou_threshold),
            original_indices=original_indices,
        )
    return euclidean_nms(
        centers=centers,
        scores=scores,
        min_distance=float(nms_distance),
        original_indices=original_indices,
    )


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


def sphere_iou_nms(
    centers: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    original_indices: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-scores)
    kept: list[int] = []
    kept_centers: list[np.ndarray] = []
    kept_radii: list[float] = []
    for candidate in order.tolist():
        candidate_index = int(candidate)
        candidate_center = centers[candidate_index]
        candidate_radius = max(float(radii[candidate_index]), 0.0)
        if kept_centers:
            ious = sphere_iou_many_np(
                center_a=candidate_center,
                radius_a=candidate_radius,
                centers_b=np.stack(kept_centers).astype(np.float32, copy=False),
                radii_b=np.asarray(kept_radii, dtype=np.float32),
            )
            if np.any(ious > float(iou_threshold)):
                continue
        kept.append(int(original_indices[candidate_index]))
        kept_centers.append(candidate_center)
        kept_radii.append(candidate_radius)
    return np.array(kept, dtype=np.int64)


def sphere_iou_many_np(center_a: np.ndarray, radius_a: float, centers_b: np.ndarray, radii_b: np.ndarray) -> np.ndarray:
    radius_a = max(float(radius_a), 0.0)
    radii_b = np.maximum(radii_b.astype(np.float32, copy=False), 0.0)
    distances = np.linalg.norm(centers_b - center_a.reshape(1, 3), axis=1).astype(np.float32, copy=False)
    volume_a = 4.0 * np.pi * radius_a**3 / 3.0
    volumes_b = 4.0 * np.pi * radii_b**3 / 3.0
    valid = (radius_a > 0.0) & (radii_b > 0.0)

    intersections = np.zeros_like(distances, dtype=np.float32)
    contained = valid & (distances <= np.abs(radius_a - radii_b))
    separated = valid & (distances >= radius_a + radii_b)
    partial = valid & ~contained & ~separated
    intersections[contained] = np.minimum(volume_a, volumes_b[contained])
    if np.any(partial):
        partial_distances = np.maximum(distances[partial], 1.0e-6)
        partial_radii = radii_b[partial]
        term = radius_a + partial_radii - partial_distances
        intersections[partial] = (
            np.pi
            * term**2
            * (
                partial_distances**2
                + 2.0 * partial_distances * (radius_a + partial_radii)
                - 3.0 * (radius_a - partial_radii) ** 2
            )
            / (12.0 * partial_distances)
        )
    unions = volume_a + volumes_b - intersections
    return np.where(unions > 0.0, intersections / unions, np.zeros_like(unions))


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

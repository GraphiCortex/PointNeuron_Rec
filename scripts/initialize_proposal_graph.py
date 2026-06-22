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
from pointneuron.graph.initialization import dijkstra_many, initialize_proposal_graph, nearest_swc_nodes, swc_arrays, weighted_swc_adjacency


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize proposal-node adjacency from an SWC trace.")
    parser.add_argument("--proposals", required=True, help="Aggregated proposal .npz from scripts/aggregate_proposals.py.")
    parser.add_argument("--init-swc", help="SWC trace used to initialize adjacency, e.g. APP2 output.")
    parser.add_argument("--use-ground-truth", action="store_true", help="Use the Gold166 selected SWC for the proposal sample.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root, used with --use-ground-truth.")
    parser.add_argument("--sample-index", type=int, help="Gold166 sample index override. Defaults to proposal metadata.")
    parser.add_argument("--mode", default="mst", choices=["mst", "knn", "mst_knn"], help="Graph initialization strategy.")
    parser.add_argument("--knn", type=int, default=2, help="Tree-distance nearest neighbors per proposal for knn modes.")
    parser.add_argument("--max-tree-distance", type=float, default=0.0, help="Optional max tree distance for knn edges. 0 disables.")
    parser.add_argument("--max-proposal-init-distance", type=float, default=0.0, help="Drop proposal nodes farther than this distance from the initialization SWC. 0 disables.")
    parser.add_argument("--min-proposal-score", type=float, default=0.0, help="Drop proposal nodes below this objectness score. 0 disables.")
    parser.add_argument("--tree-nms-distance", type=float, default=0.0, help="Greedily keep proposals at least this far apart along the initialization SWC tree. 0 disables.")
    parser.add_argument("--adaptive-tree-nms", action="store_true", help="Use density-adaptive tree-distance NMS instead of a fixed tree spacing.")
    parser.add_argument("--adaptive-tree-base-distance", type=float, default=16.0, help="Base tree spacing for --adaptive-tree-nms.")
    parser.add_argument("--adaptive-tree-min-distance", type=float, default=8.0, help="Minimum local tree spacing for --adaptive-tree-nms.")
    parser.add_argument("--adaptive-tree-max-distance", type=float, default=28.0, help="Maximum local tree spacing for --adaptive-tree-nms.")
    parser.add_argument("--adaptive-tree-density-radius", type=float, default=32.0, help="Tree radius used to estimate local proposal density.")
    parser.add_argument("--output", default="tmp/graphs/initialized_graph.npz", help="Output graph .npz.")
    parser.add_argument("--edge-csv", help="Optional CSV edge report.")
    args = parser.parse_args()

    if bool(args.init_swc) == bool(args.use_ground_truth):
        print("Provide exactly one of --init-swc or --use-ground-truth.")
        return 2

    proposals_path = Path(args.proposals)
    proposal_payload = np.load(proposals_path, allow_pickle=False)
    centers = proposal_payload["centers"].astype(np.float32, copy=False)
    radii = proposal_payload["radii"].astype(np.float32, copy=False) if "radii" in proposal_payload else np.zeros((centers.shape[0],), dtype=np.float32)
    scores = proposal_payload["scores"].astype(np.float32, copy=False) if "scores" in proposal_payload else np.zeros((centers.shape[0],), dtype=np.float32)
    features = proposal_payload["features"].astype(np.float32, copy=False) if "features" in proposal_payload else np.zeros((centers.shape[0], 0), dtype=np.float32)
    proposal_metadata = json.loads(str(proposal_payload["metadata"])) if "metadata" in proposal_payload else {}

    if args.use_ground_truth:
        sample_index = args.sample_index
        if sample_index is None:
            sample_index = int(proposal_metadata.get("sample_index", 0))
        samples = scan_gold166(args.root)
        init_swc_path = samples[sample_index].swc_path
    else:
        init_swc_path = Path(args.init_swc)

    swc = parse_swc(init_swc_path)
    errors = swc.validate()
    if errors:
        print(f"Invalid initialization SWC: {init_swc_path}")
        for error in errors:
            print(f"  {error}")
        return 2

    original_indices = np.arange(centers.shape[0], dtype=np.int64)
    prefilter_distance = np.zeros((centers.shape[0],), dtype=np.float32)
    if args.max_proposal_init_distance > 0.0 or args.min_proposal_score > 0.0 or args.tree_nms_distance > 0.0 or args.adaptive_tree_nms:
        centers, radii, scores, features, original_indices, prefilter_distance = filter_proposals(
            centers=centers,
            radii=radii,
            scores=scores,
            features=features,
            swc=swc,
            max_init_distance=args.max_proposal_init_distance,
            min_score=args.min_proposal_score,
            tree_nms_distance=args.tree_nms_distance,
            adaptive_tree_nms=args.adaptive_tree_nms,
            adaptive_tree_base_distance=args.adaptive_tree_base_distance,
            adaptive_tree_min_distance=args.adaptive_tree_min_distance,
            adaptive_tree_max_distance=args.adaptive_tree_max_distance,
            adaptive_tree_density_radius=args.adaptive_tree_density_radius,
        )

    result = initialize_proposal_graph(
        centers=centers,
        swc=swc,
        mode=args.mode,
        knn=args.knn,
        max_tree_distance=args.max_tree_distance,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "proposals": str(proposals_path),
        "proposal_metadata": proposal_metadata,
        "init_swc": str(init_swc_path),
        "mode": args.mode,
        "knn": args.knn,
        "max_tree_distance": args.max_tree_distance,
        "max_proposal_init_distance": args.max_proposal_init_distance,
        "min_proposal_score": args.min_proposal_score,
        "tree_nms_distance": args.tree_nms_distance,
        "adaptive_tree_nms": args.adaptive_tree_nms,
        "adaptive_tree_base_distance": args.adaptive_tree_base_distance,
        "adaptive_tree_min_distance": args.adaptive_tree_min_distance,
        "adaptive_tree_max_distance": args.adaptive_tree_max_distance,
        "adaptive_tree_density_radius": args.adaptive_tree_density_radius,
        "original_nodes": int(proposal_payload["centers"].shape[0]),
        "kept_nodes": int(centers.shape[0]),
        "dropped_nodes": int(proposal_payload["centers"].shape[0] - centers.shape[0]),
        "nodes": int(centers.shape[0]),
        "edges": int(result.edges.shape[0]),
        "components": connected_component_count(centers.shape[0], result.edges),
        "mean_nearest_swc_distance": finite_mean(result.nearest_swc_distance),
        "mean_edge_tree_distance": finite_mean(result.edge_tree_distance),
        "mean_edge_euclidean_distance": finite_mean(result.edge_euclidean_distance),
    }
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
        prefilter_nearest_swc_distance=prefilter_distance,
        original_indices=original_indices,
        edge_tree_distance=result.edge_tree_distance,
        edge_euclidean_distance=result.edge_euclidean_distance,
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    if args.edge_csv:
        write_edge_csv(Path(args.edge_csv), result)

    print(f"proposal_nodes: {centers.shape[0]}")
    print(f"original_nodes: {metadata['original_nodes']}")
    print(f"dropped_nodes: {metadata['dropped_nodes']}")
    print(f"init_swc: {init_swc_path}")
    print(f"mode: {args.mode}")
    print(f"edges: {result.edges.shape[0]}")
    print(f"components: {metadata['components']}")
    print(f"mean_nearest_swc_distance: {metadata['mean_nearest_swc_distance']:.4f}")
    print(f"mean_edge_tree_distance: {metadata['mean_edge_tree_distance']:.4f}")
    print(f"mean_edge_euclidean_distance: {metadata['mean_edge_euclidean_distance']:.4f}")
    print(f"output: {output_path}")
    if args.edge_csv:
        print(f"edge_csv: {args.edge_csv}")
    return 0


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("inf")


def filter_proposals(
    centers: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    features: np.ndarray,
    swc,
    max_init_distance: float,
    min_score: float,
    tree_nms_distance: float,
    adaptive_tree_nms: bool,
    adaptive_tree_base_distance: float,
    adaptive_tree_min_distance: float,
    adaptive_tree_max_distance: float,
    adaptive_tree_density_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    swc_xyz, _swc_ids, swc_edges = swc_arrays(swc)
    nearest_indices, nearest_distances = nearest_swc_nodes(centers, swc_xyz)
    keep = np.ones((centers.shape[0],), dtype=bool)
    if max_init_distance > 0.0:
        keep &= nearest_distances <= float(max_init_distance)
    if min_score > 0.0:
        keep &= scores >= float(min_score)
    if adaptive_tree_nms:
        tree_keep = adaptive_tree_distance_nms(
            assigned_swc_indices=nearest_indices[keep],
            scores=scores[keep],
            swc_xyz=swc_xyz,
            swc_edges=swc_edges,
            base_distance=adaptive_tree_base_distance,
            min_distance=adaptive_tree_min_distance,
            max_distance=adaptive_tree_max_distance,
            density_radius=adaptive_tree_density_radius,
        )
        filtered_indices = np.flatnonzero(keep)
        next_keep = np.zeros_like(keep)
        next_keep[filtered_indices[tree_keep]] = True
        keep = next_keep
    elif tree_nms_distance > 0.0:
        tree_keep = tree_distance_nms(
            assigned_swc_indices=nearest_indices[keep],
            scores=scores[keep],
            swc_xyz=swc_xyz,
            swc_edges=swc_edges,
            min_tree_distance=tree_nms_distance,
        )
        filtered_indices = np.flatnonzero(keep)
        next_keep = np.zeros_like(keep)
        next_keep[filtered_indices[tree_keep]] = True
        keep = next_keep
    original_indices = np.flatnonzero(keep).astype(np.int64, copy=False)
    return (
        centers[keep],
        radii[keep],
        scores[keep],
        features[keep],
        original_indices,
        nearest_distances[keep].astype(np.float32, copy=False),
    )


def tree_distance_nms(
    assigned_swc_indices: np.ndarray,
    scores: np.ndarray,
    swc_xyz: np.ndarray,
    swc_edges: np.ndarray,
    min_tree_distance: float,
) -> np.ndarray:
    if assigned_swc_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    tree_graph = weighted_swc_adjacency(swc_xyz, swc_edges)
    unique_sources, inverse = np.unique(assigned_swc_indices, return_inverse=True)
    source_distances = dijkstra_many(tree_graph, unique_sources)
    tree_distances = source_distances[inverse][:, assigned_swc_indices]
    tree_distances = np.maximum(tree_distances, tree_distances.T)
    order = np.argsort(-scores)
    kept: list[int] = []
    for candidate in order.tolist():
        if not kept:
            kept.append(int(candidate))
            continue
        distances_to_kept = tree_distances[int(candidate), np.array(kept, dtype=np.int64)]
        if np.all(distances_to_kept >= float(min_tree_distance)):
            kept.append(int(candidate))
    return np.array(sorted(kept), dtype=np.int64)


def adaptive_tree_distance_nms(
    assigned_swc_indices: np.ndarray,
    scores: np.ndarray,
    swc_xyz: np.ndarray,
    swc_edges: np.ndarray,
    base_distance: float,
    min_distance: float,
    max_distance: float,
    density_radius: float,
) -> np.ndarray:
    if assigned_swc_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    tree_distances = proposal_tree_distances(
        assigned_swc_indices=assigned_swc_indices,
        swc_xyz=swc_xyz,
        swc_edges=swc_edges,
    )
    local_density = (tree_distances <= float(density_radius)).sum(axis=1).astype(np.float32)
    median_density = float(np.median(local_density))
    if median_density <= 0.0:
        median_density = 1.0
    local_spacing = float(base_distance) * np.sqrt(local_density / median_density)
    local_spacing = np.clip(local_spacing, float(min_distance), float(max_distance))

    order = np.argsort(-scores)
    kept: list[int] = []
    for candidate in order.tolist():
        candidate = int(candidate)
        if not kept:
            kept.append(candidate)
            continue
        kept_array = np.array(kept, dtype=np.int64)
        required_spacing = np.minimum(local_spacing[candidate], local_spacing[kept_array])
        distances_to_kept = tree_distances[candidate, kept_array]
        if np.all(distances_to_kept >= required_spacing):
            kept.append(candidate)
    return np.array(sorted(kept), dtype=np.int64)


def proposal_tree_distances(
    assigned_swc_indices: np.ndarray,
    swc_xyz: np.ndarray,
    swc_edges: np.ndarray,
) -> np.ndarray:
    tree_graph = weighted_swc_adjacency(swc_xyz, swc_edges)
    unique_sources, inverse = np.unique(assigned_swc_indices, return_inverse=True)
    source_distances = dijkstra_many(tree_graph, unique_sources)
    tree_distances = source_distances[inverse][:, assigned_swc_indices]
    return np.maximum(tree_distances, tree_distances.T)


def write_edge_csv(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "target",
                "source_swc_id",
                "target_swc_id",
                "tree_distance",
                "euclidean_distance",
            ],
        )
        writer.writeheader()
        for edge_index, (source, target) in enumerate(result.edges.tolist()):
            writer.writerow(
                {
                    "source": source,
                    "target": target,
                    "source_swc_id": int(result.assigned_swc_ids[source]),
                    "target_swc_id": int(result.assigned_swc_ids[target]),
                    "tree_distance": float(result.edge_tree_distance[edge_index]),
                    "euclidean_distance": float(result.edge_euclidean_distance[edge_index]),
                }
            )


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


if __name__ == "__main__":
    raise SystemExit(main())

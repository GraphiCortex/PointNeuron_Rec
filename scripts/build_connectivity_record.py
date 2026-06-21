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
from pointneuron.data.swc import parse_swc
from pointneuron.graph.initialization import initialize_proposal_graph


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PointNeuron connectivity training record from proposal nodes, "
            "an initialized adjacency, and a GT-induced target adjacency."
        )
    )
    parser.add_argument("--init-graph", required=True, help="Initialized proposal graph .npz.")
    parser.add_argument("--target-swc", help="Ground-truth SWC used to build the target adjacency.")
    parser.add_argument("--use-ground-truth", action="store_true", help="Use the Gold166 selected SWC for this sample.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root, used with --use-ground-truth.")
    parser.add_argument("--sample-index", type=int, help="Gold166 sample index override. Defaults to graph metadata.")
    parser.add_argument("--target-mode", default="mst", choices=["mst", "knn", "mst_knn"], help="Target adjacency strategy.")
    parser.add_argument("--target-knn", type=int, default=2, help="Tree-distance nearest neighbors for target knn modes.")
    parser.add_argument("--target-max-tree-distance", type=float, default=0.0, help="Optional max tree distance for target knn edges. 0 disables.")
    parser.add_argument("--include-score", action="store_true", help="Append proposal objectness score to node features.")
    parser.add_argument("--output", default="tmp/connectivity/connectivity_record.npz", help="Output connectivity .npz.")
    args = parser.parse_args()

    if bool(args.target_swc) == bool(args.use_ground_truth):
        print("Provide exactly one of --target-swc or --use-ground-truth.")
        return 2

    graph_path = Path(args.init_graph)
    graph_payload = np.load(graph_path, allow_pickle=False)
    graph_metadata = parse_metadata(graph_payload)

    centers = graph_payload["centers"].astype(np.float32, copy=False)
    radii = graph_payload["radii"].astype(np.float32, copy=False)
    scores = graph_payload["scores"].astype(np.float32, copy=False)
    proposal_features = graph_payload["features"].astype(np.float32, copy=False)
    init_adjacency = graph_payload["adjacency"].astype(np.uint8, copy=False)
    init_edges = graph_payload["edges"].astype(np.int64, copy=False)

    if centers.shape[0] != init_adjacency.shape[0] or init_adjacency.shape[0] != init_adjacency.shape[1]:
        raise ValueError("Initialized graph has inconsistent center and adjacency shapes")

    if args.use_ground_truth:
        sample_index = args.sample_index
        if sample_index is None:
            proposal_metadata = graph_metadata.get("proposal_metadata", {})
            sample_index = int(proposal_metadata.get("sample_index", 0))
        samples = scan_gold166(args.root)
        target_swc_path = samples[sample_index].swc_path
    else:
        target_swc_path = Path(args.target_swc)

    target_swc = parse_swc(target_swc_path)
    errors = target_swc.validate()
    if errors:
        print(f"Invalid target SWC: {target_swc_path}")
        for error in errors:
            print(f"  {error}")
        return 2

    target = initialize_proposal_graph(
        centers=centers,
        swc=target_swc,
        mode=args.target_mode,
        knn=args.target_knn,
        max_tree_distance=args.target_max_tree_distance,
    )

    node_feature_parts = [proposal_features, centers, radii.reshape(-1, 1)]
    if args.include_score:
        node_feature_parts.append(scores.reshape(-1, 1))
    node_features = np.concatenate(node_feature_parts, axis=1).astype(np.float32, copy=False)

    metrics = compare_edges(init_edges, target.edges)
    metadata = {
        "init_graph": str(graph_path),
        "init_graph_metadata": graph_metadata,
        "target_swc": str(target_swc_path),
        "target_mode": args.target_mode,
        "target_knn": args.target_knn,
        "target_max_tree_distance": args.target_max_tree_distance,
        "include_score": bool(args.include_score),
        "nodes": int(centers.shape[0]),
        "proposal_feature_dim": int(proposal_features.shape[1]),
        "node_feature_dim": int(node_features.shape[1]),
        "init_edges": int(init_edges.shape[0]),
        "target_edges": int(target.edges.shape[0]),
        "shared_edges": int(metrics["shared_edges"]),
        "edge_precision": metrics["edge_precision"],
        "edge_recall": metrics["edge_recall"],
        "edge_f1": metrics["edge_f1"],
        "adjacency_hamming_fraction": metrics["adjacency_hamming_fraction"],
        "mean_target_nearest_swc_distance": finite_mean(target.nearest_swc_distance),
        "mean_target_edge_tree_distance": finite_mean(target.edge_tree_distance),
        "mean_target_edge_euclidean_distance": finite_mean(target.edge_euclidean_distance),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        node_features=node_features,
        proposal_features=proposal_features,
        centers=centers,
        radii=radii,
        scores=scores,
        init_adjacency=init_adjacency,
        init_edges=init_edges,
        target_adjacency=target.adjacency,
        target_edges=target.edges,
        target_assigned_swc_indices=target.assigned_swc_indices,
        target_assigned_swc_ids=target.assigned_swc_ids,
        target_nearest_swc_distance=target.nearest_swc_distance,
        target_edge_tree_distance=target.edge_tree_distance,
        target_edge_euclidean_distance=target.edge_euclidean_distance,
        original_indices=graph_payload["original_indices"] if "original_indices" in graph_payload else np.arange(centers.shape[0], dtype=np.int64),
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    print(f"nodes: {metadata['nodes']}")
    print(f"proposal_feature_dim: {metadata['proposal_feature_dim']}")
    print(f"node_feature_dim: {metadata['node_feature_dim']}")
    print(f"init_edges: {metadata['init_edges']}")
    print(f"target_edges: {metadata['target_edges']}")
    print(f"shared_edges: {metadata['shared_edges']}")
    print(f"edge_precision: {metadata['edge_precision']:.4f}")
    print(f"edge_recall: {metadata['edge_recall']:.4f}")
    print(f"edge_f1: {metadata['edge_f1']:.4f}")
    print(f"adjacency_hamming_fraction: {metadata['adjacency_hamming_fraction']:.4f}")
    print(f"mean_target_nearest_swc_distance: {metadata['mean_target_nearest_swc_distance']:.4f}")
    print(f"output: {output_path}")
    return 0


def parse_metadata(payload) -> dict:
    if "metadata" not in payload:
        return {}
    value = str(payload["metadata"])
    return json.loads(value)


def compare_edges(init_edges: np.ndarray, target_edges: np.ndarray) -> dict[str, float]:
    init_set = edge_set(init_edges)
    target_set = edge_set(target_edges)
    shared = init_set & target_set
    precision = len(shared) / len(init_set) if init_set else 0.0
    recall = len(shared) / len(target_set) if target_set else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    union = init_set | target_set
    return {
        "shared_edges": float(len(shared)),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "adjacency_hamming_fraction": float((len(union) - len(shared)) / len(union)) if union else 0.0,
    }


def edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for left, right in edges.tolist():
        left = int(left)
        right = int(right)
        if left == right:
            continue
        result.add((left, right) if left < right else (right, left))
    return result


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("inf")


if __name__ == "__main__":
    raise SystemExit(main())

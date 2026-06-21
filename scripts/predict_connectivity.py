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


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a refined proposal adjacency with a trained connectivity GAE.")
    parser.add_argument("--record", required=True, help="Connectivity record .npz.")
    parser.add_argument("--checkpoint", required=True, help="Connectivity checkpoint from scripts/train_connectivity.py.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum edge probability.")
    parser.add_argument(
        "--selection",
        default="top",
        choices=["top", "maxst", "threshold_mst", "init_top", "threshold_init_mst"],
        help="How to select predicted edges.",
    )
    parser.add_argument("--max-edges", type=int, default=0, help="Maximum edges to keep. 0 defaults to N - 1.")
    parser.add_argument("--min-edges", type=int, default=0, help="Keep this many top edges even below threshold. 0 disables.")
    parser.add_argument("--output", default="tmp/graphs/connectivity_predicted_graph.npz", help="Output graph .npz.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before prediction.")
        return 2

    from pointneuron.models.connectivity import ConnectivityGAE

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is false.")
        return 2

    record_path = Path(args.record)
    record = np.load(record_path, allow_pickle=False)
    record_metadata = json.loads(str(record["metadata"])) if "metadata" in record else {}
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})

    node_features = record["node_features"].astype(np.float32, copy=True)
    if train_args.get("normalize_node_features", False):
        mean = node_features.mean(axis=0, keepdims=True)
        std = node_features.std(axis=0, keepdims=True)
        node_features = (node_features - mean) / np.maximum(std, 1e-6)

    model = ConnectivityGAE(in_channels=int(checkpoint.get("input_dim", node_features.shape[1]))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        output = model(
            torch.from_numpy(node_features).to(device),
            torch.from_numpy(record["init_adjacency"].astype(np.float32, copy=False)).to(device),
        )
        probabilities = torch.sigmoid(output.adjacency_logits).detach().cpu().numpy()

    node_count = probabilities.shape[0]
    init_edges = record["init_edges"].astype(np.int64, copy=False) if "init_edges" in record else np.zeros((0, 2), dtype=np.int64)
    if args.max_edges > 0:
        max_edges = args.max_edges
    elif args.selection in {"threshold_mst", "threshold_init_mst"}:
        max_edges = 0
    else:
        max_edges = max(0, node_count - 1)
    if args.selection == "maxst":
        edges, edge_probabilities = maximum_spanning_tree(probabilities)
    elif args.selection == "init_top":
        edges, edge_probabilities = select_from_candidates(
            probabilities=probabilities,
            candidate_edges=init_edges,
            threshold=args.threshold,
            max_edges=max_edges,
            min_edges=args.min_edges,
        )
    elif args.selection == "threshold_mst":
        edges, edge_probabilities = threshold_then_connect(
            probabilities=probabilities,
            threshold=args.threshold,
            max_edges=max_edges,
            min_edges=args.min_edges,
            bridge_edges=None,
        )
    elif args.selection == "threshold_init_mst":
        edges, edge_probabilities = threshold_then_connect(
            probabilities=probabilities,
            threshold=args.threshold,
            max_edges=max_edges,
            min_edges=args.min_edges,
            bridge_edges=init_edges,
        )
    else:
        edges, edge_probabilities = select_edges(
            probabilities=probabilities,
            threshold=args.threshold,
            max_edges=max_edges,
            min_edges=args.min_edges,
        )
    adjacency = np.zeros((node_count, node_count), dtype=np.uint8)
    if edges.size:
        adjacency[edges[:, 0], edges[:, 1]] = 1
        adjacency[edges[:, 1], edges[:, 0]] = 1

    init_graph_metadata = record_metadata.get("init_graph_metadata", {})
    metadata = {
        "record": str(record_path),
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "selection": args.selection,
        "max_edges": max_edges,
        "min_edges": args.min_edges,
        "nodes": int(node_count),
        "edges": int(edges.shape[0]),
        "components": connected_component_count(node_count, edges),
        "init_swc": init_graph_metadata.get("init_swc"),
        "proposal_metadata": init_graph_metadata.get("proposal_metadata", {}),
        "source_record_metadata": record_metadata,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        centers=record["centers"],
        radii=record["radii"],
        scores=record["scores"],
        features=record["proposal_features"],
        adjacency=adjacency,
        edges=edges,
        edge_probabilities=edge_probabilities,
        original_indices=record["original_indices"] if "original_indices" in record else np.arange(node_count, dtype=np.int64),
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    print(f"nodes: {node_count}")
    print(f"edges: {edges.shape[0]}")
    print(f"components: {metadata['components']}")
    print(f"threshold: {args.threshold}")
    print(f"selection: {args.selection}")
    print(f"output: {output_path}")
    return 0


def select_edges(probabilities: np.ndarray, threshold: float, max_edges: int, min_edges: int) -> tuple[np.ndarray, np.ndarray]:
    candidates: list[tuple[float, int, int]] = []
    for left in range(probabilities.shape[0]):
        for right in range(left + 1, probabilities.shape[1]):
            candidates.append((float(probabilities[left, right]), left, right))
    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = [(probability, left, right) for probability, left, right in candidates if probability >= threshold]
    if min_edges > 0 and len(selected) < min_edges:
        selected = candidates[:min_edges]
    if max_edges > 0:
        selected = selected[:max_edges]
    if not selected:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    edges = np.array([[left, right] for _probability, left, right in selected], dtype=np.int64)
    edge_probabilities = np.array([probability for probability, _left, _right in selected], dtype=np.float32)
    return edges, edge_probabilities


def select_from_candidates(
    probabilities: np.ndarray,
    candidate_edges: np.ndarray,
    threshold: float,
    max_edges: int,
    min_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = candidate_edge_scores(probabilities, candidate_edges)
    selected = [(probability, left, right) for probability, left, right in candidates if probability >= threshold]
    if min_edges > 0 and len(selected) < min_edges:
        selected = candidates[:min_edges]
    if max_edges > 0:
        selected = selected[:max_edges]
    if not selected:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    edges = np.array([[left, right] for probability, left, right in selected], dtype=np.int64)
    edge_probabilities = np.array([probability for probability, _left, _right in selected], dtype=np.float32)
    return edges, edge_probabilities


def candidate_edge_scores(probabilities: np.ndarray, candidate_edges: np.ndarray) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for left, right in candidate_edges.tolist():
        left = int(left)
        right = int(right)
        if left == right:
            continue
        if left > right:
            left, right = right, left
        candidates.append((float(probabilities[left, right]), left, right))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates


def maximum_spanning_tree(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_count = probabilities.shape[0]
    candidates = [
        (float(probabilities[left, right]), left, right)
        for left in range(node_count)
        for right in range(left + 1, node_count)
    ]
    candidates.sort(reverse=True, key=lambda item: item[0])
    parent = list(range(node_count))
    rank = [0] * node_count
    selected: list[tuple[float, int, int]] = []

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

    for probability, left, right in candidates:
        if union(left, right):
            selected.append((probability, left, right))
            if len(selected) == max(0, node_count - 1):
                break

    if not selected:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    edges = np.array([[left, right] for _probability, left, right in selected], dtype=np.int64)
    edge_probabilities = np.array([probability for probability, _left, _right in selected], dtype=np.float32)
    return edges, edge_probabilities


def threshold_then_connect(
    probabilities: np.ndarray,
    threshold: float,
    max_edges: int,
    min_edges: int,
    bridge_edges: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = probabilities.shape[0]
    selected_edges, selected_probabilities = select_edges(
        probabilities=probabilities,
        threshold=threshold,
        max_edges=0,
        min_edges=min_edges,
    )
    selected = [
        (float(probability), int(edge[0]), int(edge[1]))
        for edge, probability in zip(selected_edges.tolist(), selected_probabilities.tolist())
    ]
    selected_pairs = {(min(left, right), max(left, right)) for _probability, left, right in selected}

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

    for _probability, left, right in selected:
        union(left, right)

    if bridge_edges is None:
        candidates = [
            (float(probabilities[left, right]), left, right)
            for left in range(node_count)
            for right in range(left + 1, node_count)
            if (left, right) not in selected_pairs
        ]
        candidates.sort(reverse=True, key=lambda item: item[0])
    else:
        candidates = [
            (probability, left, right)
            for probability, left, right in candidate_edge_scores(probabilities, bridge_edges)
            if (left, right) not in selected_pairs
        ]
    for probability, left, right in candidates:
        if union(left, right):
            selected.append((probability, left, right))
            selected_pairs.add((left, right))
            if connected_component_count(node_count, np.array([[edge_left, edge_right] for _p, edge_left, edge_right in selected], dtype=np.int64)) == 1:
                break

    selected.sort(reverse=True, key=lambda item: item[0])
    if max_edges > 0 and len(selected) > max_edges:
        selected = selected[:max_edges]
    if not selected:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    edges = np.array([[left, right] for _probability, left, right in selected], dtype=np.int64)
    edge_probabilities = np.array([probability for probability, _left, _right in selected], dtype=np.float32)
    return edges, edge_probabilities


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

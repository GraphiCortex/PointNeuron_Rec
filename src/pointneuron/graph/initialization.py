from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from pointneuron.data.swc import SwcTree


@dataclass(frozen=True)
class GraphInitializationResult:
    adjacency: np.ndarray
    edges: np.ndarray
    assigned_swc_indices: np.ndarray
    assigned_swc_ids: np.ndarray
    nearest_swc_distance: np.ndarray
    edge_tree_distance: np.ndarray
    edge_euclidean_distance: np.ndarray


def initialize_proposal_graph(
    centers: np.ndarray,
    swc: SwcTree,
    mode: str = "mst",
    knn: int = 2,
    max_tree_distance: float = 0.0,
) -> GraphInitializationResult:
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"Expected centers with shape [N, 3], got {centers.shape}")
    if mode not in {"mst", "knn", "mst_knn"}:
        raise ValueError("mode must be one of: mst, knn, mst_knn")

    swc_xyz, swc_ids, swc_edges = swc_arrays(swc)
    if centers.shape[0] == 0:
        return empty_result()
    if swc_xyz.shape[0] == 0:
        raise ValueError("Cannot initialize a graph from an empty SWC")

    assigned_indices, nearest_distances = nearest_swc_nodes(centers, swc_xyz)
    tree_graph = weighted_swc_adjacency(swc_xyz, swc_edges)
    unique_sources, inverse = np.unique(assigned_indices, return_inverse=True)
    source_distances = dijkstra_many(tree_graph, unique_sources)
    proposal_tree_distances = source_distances[inverse][:, assigned_indices]
    proposal_tree_distances = np.maximum(proposal_tree_distances, proposal_tree_distances.T)

    edge_set: set[tuple[int, int]] = set()
    if mode in {"mst", "mst_knn"}:
        edge_set.update(minimum_spanning_tree(proposal_tree_distances))
    if mode in {"knn", "mst_knn"}:
        edge_set.update(knn_tree_edges(proposal_tree_distances, knn=knn, max_tree_distance=max_tree_distance))

    edges = np.array(sorted(edge_set), dtype=np.int64).reshape(-1, 2) if edge_set else np.zeros((0, 2), dtype=np.int64)
    adjacency = np.zeros((centers.shape[0], centers.shape[0]), dtype=np.uint8)
    if edges.size:
        adjacency[edges[:, 0], edges[:, 1]] = 1
        adjacency[edges[:, 1], edges[:, 0]] = 1

    edge_tree_distance = edge_distances_from_matrix(proposal_tree_distances, edges)
    edge_euclidean_distance = edge_euclidean_distances(centers, edges)
    return GraphInitializationResult(
        adjacency=adjacency,
        edges=edges,
        assigned_swc_indices=assigned_indices.astype(np.int64, copy=False),
        assigned_swc_ids=swc_ids[assigned_indices].astype(np.int64, copy=False),
        nearest_swc_distance=nearest_distances.astype(np.float32, copy=False),
        edge_tree_distance=edge_tree_distance.astype(np.float32, copy=False),
        edge_euclidean_distance=edge_euclidean_distance.astype(np.float32, copy=False),
    )


def initialize_geometric_graph(
    centers: np.ndarray,
    mode: str = "mst",
    knn: int = 2,
    max_distance: float = 0.0,
) -> GraphInitializationResult:
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"Expected centers with shape [N, 3], got {centers.shape}")
    if mode not in {"mst", "knn", "mst_knn"}:
        raise ValueError("mode must be one of: mst, knn, mst_knn")
    if centers.shape[0] == 0:
        return empty_result()

    distances = euclidean_distance_matrix(centers)
    edge_set: set[tuple[int, int]] = set()
    if mode in {"mst", "mst_knn"}:
        edge_set.update(minimum_spanning_tree(distances))
    if mode in {"knn", "mst_knn"}:
        edge_set.update(knn_tree_edges(distances, knn=knn, max_tree_distance=max_distance))

    edges = np.array(sorted(edge_set), dtype=np.int64).reshape(-1, 2) if edge_set else np.zeros((0, 2), dtype=np.int64)
    adjacency = np.zeros((centers.shape[0], centers.shape[0]), dtype=np.uint8)
    if edges.size:
        adjacency[edges[:, 0], edges[:, 1]] = 1
        adjacency[edges[:, 1], edges[:, 0]] = 1

    edge_distances = edge_distances_from_matrix(distances, edges).astype(np.float32, copy=False)
    return GraphInitializationResult(
        adjacency=adjacency,
        edges=edges,
        assigned_swc_indices=np.full((centers.shape[0],), -1, dtype=np.int64),
        assigned_swc_ids=np.full((centers.shape[0],), -1, dtype=np.int64),
        nearest_swc_distance=np.zeros((centers.shape[0],), dtype=np.float32),
        edge_tree_distance=edge_distances,
        edge_euclidean_distance=edge_distances,
    )


def euclidean_distance_matrix(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    deltas = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(deltas, axis=2).astype(np.float32, copy=False)


def swc_arrays(swc: SwcTree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.array([node.node_id for node in swc.nodes], dtype=np.int64)
    xyz = np.array([[node.x, node.y, node.z] for node in swc.nodes], dtype=np.float32)
    id_to_index = {int(node_id): index for index, node_id in enumerate(ids.tolist())}
    edges = []
    for child_index, node in enumerate(swc.nodes):
        parent_index = id_to_index.get(int(node.parent_id))
        if parent_index is not None:
            edges.append((parent_index, child_index))
    edge_array = np.array(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64)
    return xyz, ids, edge_array


def nearest_swc_nodes(centers: np.ndarray, swc_xyz: np.ndarray, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    nearest_indices = np.zeros((centers.shape[0],), dtype=np.int64)
    nearest_distances = np.zeros((centers.shape[0],), dtype=np.float32)
    for start in range(0, centers.shape[0], chunk_size):
        end = min(start + chunk_size, centers.shape[0])
        distances = np.linalg.norm(centers[start:end, None, :] - swc_xyz[None, :, :], axis=2)
        nearest_indices[start:end] = distances.argmin(axis=1)
        nearest_distances[start:end] = distances.min(axis=1)
    return nearest_indices, nearest_distances


def weighted_swc_adjacency(swc_xyz: np.ndarray, swc_edges: np.ndarray) -> list[list[tuple[int, float]]]:
    graph: list[list[tuple[int, float]]] = [[] for _ in range(swc_xyz.shape[0])]
    for left, right in swc_edges.tolist():
        weight = float(np.linalg.norm(swc_xyz[left] - swc_xyz[right]))
        graph[left].append((right, weight))
        graph[right].append((left, weight))
    return graph


def dijkstra_many(graph: list[list[tuple[int, float]]], sources: np.ndarray) -> np.ndarray:
    distances = np.zeros((sources.shape[0], len(graph)), dtype=np.float64)
    for index, source in enumerate(sources.tolist()):
        distances[index] = dijkstra(graph, int(source))
    return distances


def dijkstra(graph: list[list[tuple[int, float]]], source: int) -> np.ndarray:
    distances = np.full((len(graph),), np.inf, dtype=np.float64)
    distances[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        distance, node = heapq.heappop(heap)
        node = int(node)
        distance = float(distance)
        if distance > float(distances[node]):
            continue
        for neighbor, weight in graph[node]:
            neighbor = int(neighbor)
            candidate = distance + float(weight)
            if candidate < float(distances[neighbor]):
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distances


def minimum_spanning_tree(distance_matrix: np.ndarray) -> list[tuple[int, int]]:
    count = distance_matrix.shape[0]
    if count <= 1:
        return []
    candidates = [
        (float(distance_matrix[left, right]), left, right)
        for left in range(count)
        for right in range(left + 1, count)
        if np.isfinite(distance_matrix[left, right])
    ]
    candidates.sort(key=lambda item: item[0])
    parent = list(range(count))
    rank = [0] * count
    edges: list[tuple[int, int]] = []

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

    for _distance, left, right in candidates:
        if union(left, right):
            edges.append((left, right))
            if len(edges) == count - 1:
                break
    return edges


def knn_tree_edges(distance_matrix: np.ndarray, knn: int, max_tree_distance: float = 0.0) -> list[tuple[int, int]]:
    if knn <= 0:
        return []
    edges: set[tuple[int, int]] = set()
    for index in range(distance_matrix.shape[0]):
        distances = distance_matrix[index].copy()
        distances[index] = np.inf
        order = np.argsort(distances)
        added = 0
        for neighbor in order.tolist():
            distance = float(distances[neighbor])
            if not np.isfinite(distance):
                continue
            if max_tree_distance > 0.0 and distance > max_tree_distance:
                continue
            left, right = sorted((index, int(neighbor)))
            edges.add((left, right))
            added += 1
            if added >= knn:
                break
    return sorted(edges)


def edge_distances_from_matrix(distance_matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return distance_matrix[edges[:, 0], edges[:, 1]]


def edge_euclidean_distances(centers: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(centers[edges[:, 0]] - centers[edges[:, 1]], axis=1)


def empty_result() -> GraphInitializationResult:
    return GraphInitializationResult(
        adjacency=np.zeros((0, 0), dtype=np.uint8),
        edges=np.zeros((0, 2), dtype=np.int64),
        assigned_swc_indices=np.zeros((0,), dtype=np.int64),
        assigned_swc_ids=np.zeros((0,), dtype=np.int64),
        nearest_swc_distance=np.zeros((0,), dtype=np.float32),
        edge_tree_distance=np.zeros((0,), dtype=np.float32),
        edge_euclidean_distance=np.zeros((0,), dtype=np.float32),
    )

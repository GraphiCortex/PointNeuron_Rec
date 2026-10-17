"""Reference-free score-thresholded spanning forest and diagnostic SWC export."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from pointneuron.data.swc import SwcNode, parse_swc, write_swc


@dataclass(frozen=True)
class ForestDecoding:
    status: str
    accepted_nodes: tuple[int, ...]
    selected_edges: tuple[tuple[int, int], ...]
    parent: tuple[int, ...]  # -2 rejected; -1 root; otherwise parent input index.
    traversal: tuple[int, ...]
    statistics: dict


def _logit_threshold(probability):
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("Probability cutoffs must be strictly between zero and one")
    return math.log(probability)-math.log1p(-probability)


def decode_forest(packet, candidates, edge_logits, *, node_cutoff=.5, edge_cutoff=.5):
    """Kruskal on eligible edges; no connectedness completion, pruning or cap.

Threshold/sort in logit space to avoid sigmoid saturation creating false ties.
Stable proposal IDs settle exact ties; root choice is a serialization convention.
"""
    packet.validate()
    candidates.validate(packet)
    if edge_logits.shape != (len(candidates.edges),):
        raise ValueError("Expected one undirected edge logit per candidate")
    logits = edge_logits.detach().cpu().double().numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite edge logits")
    node_threshold, edge_threshold = _logit_threshold(node_cutoff), _logit_threshold(edge_cutoff)
    node_logits = packet.objectness_logits.detach().cpu().double().numpy()
    confidence_logit = node_logits[:, 1]-node_logits[:, 0]
    accepted_mask = confidence_logit >= node_threshold
    accepted = np.flatnonzero(accepted_mask).tolist()
    pairs = candidates.edges.detach().cpu().numpy()
    ids = packet.node_ids.detach().cpu().numpy()
    eligible, rejected_node, below_cutoff = [], 0, 0
    for e, (u, v) in enumerate(pairs):
        if not accepted_mask[u] or not accepted_mask[v]:
            rejected_node += 1
        elif logits[e] < edge_threshold:
            below_cutoff += 1
        else:
            eligible.append(e)
    eligible.sort(key=lambda e: (-logits[e], min(ids[pairs[e, 0]], ids[pairs[e, 1]]),
                                 max(ids[pairs[e, 0]], ids[pairs[e, 1]])))
    union_parent = list(range(len(ids)))
    size = [1]*len(ids)

    def root(i):
        while union_parent[i] != i:
            union_parent[i] = union_parent[union_parent[i]]
            i = union_parent[i]
        return i

    selected, cycles = [], 0
    adjacency = [[] for _ in ids]
    for e in eligible:
        u, v = map(int, pairs[e])
        a, b = root(u), root(v)
        if a == b:
            cycles += 1
            continue
        if size[a] < size[b]:
            a, b = b, a
        union_parent[b] = a
        size[a] += size[b]
        selected.append((u, v))
        adjacency[u].append(v)
        adjacency[v].append(u)
    components = {}
    for u in accepted:
        components.setdefault(root(u), []).append(u)
    roots = [min(nodes, key=lambda i: (-confidence_logit[i], ids[i])) for nodes in components.values()]
    roots.sort(key=lambda i: ids[i])
    parent, traversal = [-2]*len(ids), []
    for r in roots:
        parent[r] = -1
        pending = deque([r])
        while pending:
            u = pending.popleft()
            traversal.append(u)
            for v in sorted(adjacency[u], key=lambda i: ids[i]):
                if parent[v] != -2:
                    continue
                parent[v] = u
                pending.append(v)
    if len(traversal) != len(accepted) or len(selected) != len(accepted)-len(roots):
        raise RuntimeError("Invalid decoded forest")
    statistics = {"input_nodes": len(ids), "accepted_nodes": len(accepted), "rejected_nodes": len(ids)-len(accepted),
                  "candidate_edges": len(pairs), "edges_rejected_by_node_cutoff": rejected_node,
                  "edges_below_edge_cutoff": below_cutoff, "eligible_edges": len(eligible),
                  "cycle_rejections": cycles, "selected_edges": len(selected), "components": len(roots),
                  "isolated_nodes": sum(not adjacency[u] for u in accepted),
                  "node_cutoff": node_cutoff, "edge_cutoff": edge_cutoff,
                  "root_policy": "highest_node_logit_difference_then_smallest_stable_id"}
    return ForestDecoding("decoded" if accepted else "empty_reconstruction", tuple(accepted), tuple(selected),
                          tuple(parent), tuple(traversal), statistics)


def export_decoded_swc(packet, decoded, path, *, description, radius_floor=1.):
    """Verify SWC serialization; do not write empty success or overwrite evidence."""
    packet.validate()
    if decoded.status != "decoded" or not decoded.accepted_nodes:
        raise ValueError("Empty accepted node set is a reconstruction failure")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing SWC")
    if not math.isfinite(radius_floor) or radius_floor <= 0:
        raise ValueError("Invalid export radius floor")
    if not description.strip():
        raise ValueError("Describe the provenance of diagnostic scores")
    order = decoded.traversal
    positions = packet.positions.detach().cpu().double().numpy()
    radii = packet.radii.detach().cpu().double().numpy()[:, 0]
    serial_ids = {u: i+1 for i, u in enumerate(order)}
    rows = [SwcNode(serial_ids[u], 3, *map(float, positions[u]), max(float(radii[u]), radius_floor),
                    -1 if decoded.parent[u] == -1 else serial_ids[decoded.parent[u]]) for u in order]
    write_swc(path, rows, comments=[description, "Root/type are serialization conventions, not biological predictions."])
    reloaded = parse_swc(path)
    errors = reloaded.validate()
    if errors:
        raise ValueError("Exported SWC invalid: " + "; ".join(errors))
    inverse = {serial_ids[u]: u for u in order}
    reloaded_edges = {tuple(sorted((inverse[node.node_id], inverse[node.parent_id])))
                      for node in reloaded.nodes if node.parent_id != -1}
    expected = {tuple(sorted(pair)) for pair in decoded.selected_edges}
    restored_xyz = np.array([[n.x, n.y, n.z] for n in reloaded.nodes])
    restored_radius = np.array([n.radius for n in reloaded.nodes])
    delta = float(np.abs(restored_xyz-positions[list(order)]).max())
    radius_delta = float(np.abs(restored_radius-np.maximum(radii[list(order)], radius_floor)).max())
    parent_before_child = all(node.parent_id == -1 or node.parent_id < node.node_id for node in reloaded.nodes)
    passed = (reloaded_edges == expected and len(reloaded.nodes) == len(order)
              and reloaded.root_count == decoded.statistics["components"]
              and delta <= 1e-6 and radius_delta <= 1e-6 and parent_before_child)
    if not passed:
        raise ValueError("SWC serialization changed decoded geometry/topology")
    return {"passed": passed, "maximum_coordinate_delta": delta, "maximum_radius_delta": radius_delta,
            "parent_before_child": parent_before_child, "nodes": len(order),
            "edges": len(expected), "roots": reloaded.root_count, "radius_floor": radius_floor}

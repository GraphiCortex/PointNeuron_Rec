"""Reference-only representation diagnostics, never an inference graph builder.

Reference identities/order intentionally provide perfect correspondence and edge
labels. Nothing here trains a model or changes the production reconstruction.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pointneuron.data.curve_targets import clip_parameters, curve_geometry, patch_targets
from pointneuron.data.swc import SwcNode, SwcTree, parse_swc, write_swc


@dataclass
class ReferenceGraph:
    identities: list[tuple]
    xyz: np.ndarray
    radii: np.ndarray
    edges: list[tuple[int, int]]  # Directed according to reference order.


def edge_identity(parent, child, fraction):
    if abs(fraction) < 1e-8:
        return ("node", parent)
    if abs(fraction - 1) < 1e-8:
        return ("node", child)
    return ("edge", parent, child, round(float(fraction), 8))


def merge_intervals(intervals, tolerance=1e-7):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1] + tolerance:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return result


def segment_intervals(a, b, boxes):
    if boxes is None:
        return [(0., 1.)]
    return merge_intervals([interval for center, radius in boxes
                            if (interval := clip_parameters(a, b, np.asarray(center) - radius,
                                                            np.asarray(center) + radius)) is not None])


def inside(xyz, boxes):
    return boxes is None or any(np.all(np.abs(xyz - np.asarray(center)) <= radius + 1e-8)
                                for center, radius in boxes)


def clipped_reference_graph(tree: SwcTree, boxes=None):
    """Clip original SWC edges independently, preserving original vertices.

The domain is the union of closed crop boxes; no edges bridge excluded portions.
Distinct identities at identical coordinates remain distinct.
"""
    curve_geometry(tree)  # Validate the forest, including cycles and finite data.
    nodes = {n.node_id: n for n in tree.nodes}
    identities, positions, radii, lookup, edges = [], [], [], {}, set()

    def add(identity, xyz, radius):
        if identity not in lookup:
            lookup[identity] = len(identities)
            identities.append(identity)
            positions.append(xyz)
            radii.append(radius)
        return lookup[identity]

    for n in tree.nodes:
        xyz = np.array([n.x, n.y, n.z])
        if inside(xyz, boxes):
            add(("node", n.node_id), xyz, n.radius)
        if n.parent_id == -1:
            continue
        p = nodes[n.parent_id]
        a, b = np.array([p.x, p.y, p.z]), xyz
        for lo, hi in segment_intervals(a, b, boxes):
            u, v = [add(edge_identity(p.node_id, n.node_id, t), a + t * (b-a),
                        p.radius + t * (n.radius-p.radius)) for t in (lo, hi)]
            if u != v:
                edges.add((u, v))
    return ReferenceGraph(identities, np.asarray(positions, dtype=float).reshape(-1, 3),
                          np.asarray(radii, dtype=float), sorted(edges))


def target_graph(tree: SwcTree, boxes=None, step=8.):
    """Existing target points plus oracle edges along visible reference chains.

For multiple crops, union identities and re-order by the reference. This is
oracle consolidation, not a deployable duplicate matcher. Chords replace the
intervening reference polylines, so geometry loss remains measurable.
"""
    geometry = curve_geometry(tree)
    if boxes is None:
        xyz = np.array([[n.x, n.y, n.z] for n in tree.nodes])
        center = (xyz.min(0) + xyz.max(0)) / 2
        target_boxes = [(center, float(np.ptp(xyz, axis=0).max()/2 + 1))]
    else:
        if not boxes:
            raise ValueError("At least one crop is required")
        target_boxes = boxes
    targets = {}
    for center, radius in target_boxes:
        for target in patch_targets(geometry, center, radius, step):
            key = tuple(target["reference_identity"])
            if key in targets and not np.allclose(targets[key]["xyz"], target["xyz"], atol=1e-5, rtol=0):
                raise ValueError("Inconsistent oracle identities between crops")
            targets[key] = target
    identities = list(targets)
    lookup = {key: i for i, key in enumerate(identities)}
    edges = set()
    for ids, xyz, radii, arc in geometry[3]:
        node_positions = {node: float(arc[k]) for k, node in enumerate(ids)}
        edge_positions = {(p, c): (float(arc[k]), float(arc[k+1]-arc[k]))
                          for k, (p, c) in enumerate(zip(ids[:-1], ids[1:]))}
        events = []
        for key, index in lookup.items():
            if key[0] == "node" and key[1] in node_positions:
                events.append((node_positions[key[1]], index))
            elif key[0] == "edge" and tuple(key[1:3]) in edge_positions:
                start, length = edge_positions[tuple(key[1:3])]
                events.append((start + key[3]*length, index))
        events.sort()
        intervals = []
        for k in range(len(ids)-1):
            for lo, hi in segment_intervals(xyz[k], xyz[k+1], boxes):
                intervals.append((float(arc[k] + lo*(arc[k+1]-arc[k])),
                                  float(arc[k] + hi*(arc[k+1]-arc[k]))))
        for lo, hi in merge_intervals(intervals):
            visible = [(s, i) for s, i in events if lo-1e-5 <= s <= hi+1e-5]
            if not visible or abs(visible[0][0]-lo) > 1e-5 or abs(visible[-1][0]-hi) > 1e-5:
                raise ValueError("Target representation lacks a visible-chain boundary")
            for (_, u), (_, v) in zip(visible[:-1], visible[1:]):
                if u != v:
                    edges.add((u, v))
    graph = ReferenceGraph(identities, np.array([targets[k]["xyz"] for k in identities]).reshape(-1, 3),
                           np.array([targets[k]["radius"] for k in identities]), sorted(edges))
    topology(graph)  # Fail rather than silently decode a malformed oracle.
    return graph


def topology(graph):
    n = len(graph.xyz)
    incoming = np.zeros(n, dtype=int)
    children = [[] for _ in range(n)]
    degree = np.zeros(n, dtype=int)
    if len(set(graph.edges)) != len(graph.edges):
        raise ValueError("Duplicate graph edges")
    for u, v in graph.edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("Invalid graph edge")
        incoming[v] += 1
        degree[u] += 1
        degree[v] += 1
        children[u].append(v)
    if (incoming > 1).any():
        raise ValueError("Oracle node has multiple parents")
    roots = np.flatnonzero(incoming == 0).tolist()
    pending, order = deque(roots), []
    component = np.full(n, -1, dtype=int)
    component[roots] = np.arange(len(roots))
    while pending:
        u = pending.popleft()
        order.append(u)
        for v in children[u]:
            component[v] = component[u]
            pending.append(v)
    if len(order) != n:
        raise ValueError("Oracle graph contains a cycle")
    return {"nodes": n, "edges": len(graph.edges), "components": len(roots),
            "junctions": int((degree >= 3).sum()), "terminals": int((degree == 1).sum()),
            "isolated": int((degree == 0).sum()), "degree": degree,
            "component": component, "order": order}


def topology_comparison(reference, predicted):
    rt, pt = topology(reference), topology(predicted)
    lookup = {key: i for i, key in enumerate(predicted.identities)}
    required = [i for i, degree in enumerate(rt["degree"]) if degree != 2]
    missing = [i for i in required if reference.identities[i] not in lookup]
    degree_errors = [i for i in required if reference.identities[i] in lookup
                     and rt["degree"][i] != pt["degree"][lookup[reference.identities[i]]]]
    reference_lookup = {key: i for i, key in enumerate(reference.identities)}
    portions = {}
    for u, v in reference.edges:
        a, b = reference.identities[u], reference.identities[v]
        edge = tuple(a[1:3]) if a[0] == "edge" else tuple(b[1:3]) if b[0] == "edge" else (a[1], b[1])
        lo = a[3] if a[0] == "edge" else 0.
        hi = b[3] if b[0] == "edge" else 1.
        portions.setdefault(edge, []).append((lo, hi, int(rt["component"][u])))
    components = []
    for key in predicted.identities:
        if key in reference_lookup:
            components.append(int(rt["component"][reference_lookup[key]]))
        elif key[0] == "edge":
            matches = {c for lo, hi, c in portions.get(tuple(key[1:3]), []) if lo-1e-7 <= key[3] <= hi+1e-7}
            components.append(next(iter(matches)) if len(matches) == 1 else None)
        else:
            components.append(None)
    cross_components = sum(components[u] != components[v] for u, v in predicted.edges)
    unresolved = sum(c is None for c in components)

    # Compare branch incidence after suppressing degree-two vertices; equal
    # degree/component counts alone would miss a rewiring between terminals.
    critical = {reference.identities[i] for i in required}
    def incidence(graph):
        adjacency = [[] for _ in graph.xyz]
        for u, v in graph.edges:
            adjacency[u].append(v)
            adjacency[v].append(u)
        result = set()
        for start, key in enumerate(graph.identities):
            if key not in critical:
                continue
            for neighbor in adjacency[start]:
                previous, current = start, neighbor
                while graph.identities[current] not in critical:
                    following = [v for v in adjacency[current] if v != previous]
                    if len(following) != 1:
                        raise ValueError("Unexpected branch/terminal between structural identities")
                    previous, current = current, following[0]
                result.add(frozenset((key, graph.identities[current])))
        return result
    incidence_errors = len(incidence(reference) ^ incidence(predicted)) if not missing and not degree_errors else None
    return {"reference": {k: rt[k] for k in ("nodes", "edges", "components", "junctions", "terminals", "isolated")},
            "target": {k: pt[k] for k in ("nodes", "edges", "components", "junctions", "terminals", "isolated")},
            "missing_structural_identities": len(missing), "structural_degree_errors": len(degree_errors),
            "detected_cross_component_edges": int(cross_components),
            "unresolved_component_identities": unresolved, "branch_incidence_errors": incidence_errors,
            "passed": not missing and not degree_errors and not cross_components and not unresolved and incidence_errors == 0
                      and rt["components"] == pt["components"]}


def candidate_edges(xyz, neighbors=16, radius=8.):
    """Image-independent geometric rule; references are not consulted here."""
    xyz = np.asarray(xyz, dtype=float)
    if neighbors < 1 or radius <= 0 or not np.isfinite(radius):
        raise ValueError("Invalid candidate graph settings")
    if len(xyz) <= 1:
        return set(), set()
    tree = cKDTree(xyz)
    k = min(len(xyz), neighbors+1)
    _, indices = tree.query(xyz, k=k)
    knn = {tuple(sorted((i, int(j)))) for i, rows in enumerate(indices)
           for j in [j for j in rows if i != j][:neighbors]}
    # The small tolerance admits floating-point coordinates at the exact boundary.
    radial = {tuple(map(int, pair)) for pair in tree.query_pairs(radius+1e-6)}
    return knn, knn | radial


def candidate_report(graph, neighbors=16, radius=8.):
    required = {tuple(sorted(edge)) for edge in graph.edges}
    knn, combined = candidate_edges(graph.xyz, neighbors, radius)
    def row(edges):
        missing = sorted(required-edges)
        return {"candidate_edges": len(edges), "required_edges": len(required),
                "missing_required_edges": len(missing), "required_edge_recall":
                1. if not required else (len(required)-len(missing))/len(required),
                "nonrequired_candidates": len(edges-required), "missing_edge_indices": missing}
    return {"knn": row(knn), "knn_union_radius": row(combined)}


def segments(graph):
    indices = np.array(graph.edges, dtype=int).reshape(-1, 2)
    return graph.xyz[indices] if len(indices) else np.empty((0, 2, 3))


def sample_arclength(graph, maximum_step=.5):
    """Midpoint quadrature weighted by segment length, not vertex count."""
    if not np.isfinite(maximum_step) or maximum_step <= 0:
        raise ValueError("Invalid quadrature step")
    points, weights = [], []
    for a, b in segments(graph):
        length = float(np.linalg.norm(b-a))
        if length == 0:
            continue
        count = max(1, int(np.ceil(length/maximum_step)))
        t = (np.arange(count)+.5)/count
        points.append(a + t[:, None]*(b-a))
        weights.append(np.full(count, length/count))
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.concatenate(weights) if weights else np.empty(0))


def distance_to_graph(points, graph):
    """Exact point-to-segment distance in bounded chunks; no vertex proxy."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    result = np.full(len(points), np.inf)
    if not len(graph.xyz):
        return result
    # Include isolated vertices too. Arc-length metrics do not weight them.
    result[:] = cKDTree(graph.xyz).query(points)[0]
    for start in range(0, len(points), 256):
        p = points[start:start+256]
        best = result[start:start+256]**2
        all_segments = segments(graph)
        for offset in range(0, len(all_segments), 256):
            batch = all_segments[offset:offset+256]
            a, delta = batch[:, 0], batch[:, 1]-batch[:, 0]
            length2 = np.sum(delta*delta, axis=1)
            vector = p[:, None, :]-a[None]
            fraction = np.clip(np.sum(vector*delta[None], axis=2)/np.maximum(length2, 1e-30), 0, 1)
            dist2 = np.sum((vector-fraction[..., None]*delta[None])**2, axis=2)
            best = np.minimum(best, dist2.min(axis=1))
        result[start:start+len(p)] = np.sqrt(best)
    return result


def geometry_report(reference, predicted, maximum_step=.5, tolerances=(2., 6.)):
    samples, weights = sample_arclength(reference, maximum_step)
    predicted_samples, predicted_weights = sample_arclength(predicted, maximum_step)
    recall_distance = distance_to_graph(samples, predicted)
    precision_distance = distance_to_graph(predicted_samples, reference)
    def fraction(distance, weight, tolerance):
        return None if not len(weight) else float(weight[distance <= tolerance].sum()/weight.sum())
    return {"quadrature_maximum_step": maximum_step,
            "reference_length": float(weights.sum()), "export_length": float(predicted_weights.sum()),
            "reference_quadrature_samples": len(weights), "export_quadrature_samples": len(predicted_weights),
            "max_reference_distance": None if not len(weights) else float(recall_distance.max()),
            "max_export_distance": None if not len(predicted_weights) else float(precision_distance.max()),
            "tolerances": {str(float(t)): {"recall": fraction(recall_distance, weights, t),
                                         "precision": fraction(precision_distance, predicted_weights, t)}
                           for t in tolerances}}


def export_and_verify(graph, path: Path, radius_floor=1.):
    """Write a parent-before-child SWC and verify serialized geometry/topology."""
    state = topology(graph)
    order = state["order"]
    ids = {u: k+1 for k, u in enumerate(order)}
    parents = {v: u for u, v in graph.edges}
    rows = [SwcNode(ids[u], 3, *map(float, graph.xyz[u]), max(float(graph.radii[u]), radius_floor),
                    -1 if u not in parents else ids[parents[u]]) for u in order]
    write_swc(path, rows, comments=["REFERENCE ORACLE ONLY; not a learned reconstruction."])
    reloaded = parse_swc(path)
    errors = reloaded.validate()
    if errors:
        raise ValueError("Invalid exported SWC: " + "; ".join(errors))
    xyz = np.array([[n.x, n.y, n.z] for n in reloaded.nodes]).reshape(-1, 3)
    delta = float(np.abs(xyz-graph.xyz[order]).max()) if len(xyz) else 0.
    expected_edges = {(ids[u], ids[v]) for u, v in graph.edges}
    actual_edges = {(n.parent_id, n.node_id) for n in reloaded.nodes if n.parent_id != -1}
    passed = (delta <= 1e-6 and expected_edges == actual_edges
              and reloaded.root_count == state["components"] and len(reloaded.nodes) == len(graph.xyz))
    if not passed:
        raise ValueError("SWC round-trip changed geometry or topology")
    return {"passed": passed, "maximum_coordinate_delta": delta, "roots": reloaded.root_count,
            "nodes": len(rows), "edges": len(actual_edges), "radius_floor": radius_floor}

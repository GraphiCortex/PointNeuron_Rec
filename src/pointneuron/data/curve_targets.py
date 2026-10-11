"""Version 1 reference-curve targets for the PN2 query comparison."""
from __future__ import annotations

import numpy as np
from pointneuron.data.swc import SwcTree

TARGET_SCHEMA = "pointneuron2-curve-targets-v1"
ROLE_BITS = {"root": 1, "leaf": 2, "junction": 4, "arc_sample": 8, "crop_boundary": 16}


def curve_geometry(tree: SwcTree):
    """Split a validated forest at roots, leaves and branch junctions."""
    errors = tree.validate()
    if errors:
        raise ValueError("; ".join(errors))
    nodes = {n.node_id: n for n in tree.nodes}
    children = {i: [] for i in nodes}
    for n in tree.nodes:
        if not np.isfinite([n.x, n.y, n.z, n.radius]).all() or n.radius < 0:
            raise ValueError("Nonfinite geometry or negative radius")
        if n.parent_id != -1:
            children[n.parent_id].append(n.node_id)
    reachable, pending = set(), [i for i, n in nodes.items() if n.parent_id == -1]
    while pending:
        i = pending.pop()
        if i in reachable:
            raise ValueError("Cycle in reference")
        reachable.add(i)
        pending.extend(children[i])
    if len(reachable) != len(nodes):
        raise ValueError("Reference contains an unrooted cycle")
    critical = {i for i, n in nodes.items()
                if n.parent_id == -1 or len(children[i]) != 1}
    chains, visited = [], set()
    for start in sorted(critical):
        for child in sorted(children[start]):
            ids = [start]
            current = child
            while True:
                edge = (ids[-1], current)
                if edge in visited:
                    raise ValueError("Repeated edge or cycle")
                visited.add(edge)
                ids.append(current)
                if current in critical:
                    break
                current = children[current][0]
            xyz = np.array([[nodes[i].x, nodes[i].y, nodes[i].z] for i in ids])
            radii = np.array([nodes[i].radius for i in ids])
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            chains.append((ids, xyz, radii, arc))
    if len(visited) != tree.edge_count:
        raise ValueError("Forest contains an unvisited cycle")
    return nodes, children, critical, chains


def clip_parameters(a, b, lower, upper):
    entry, leave = 0., 1.
    for axis, delta in enumerate(b - a):
        if abs(delta) < 1e-12:
            if not lower[axis] <= a[axis] <= upper[axis]:
                return None
        else:
            bounds = sorted(((lower[axis] - a[axis]) / delta,
                             (upper[axis] - a[axis]) / delta))
            entry, leave = max(entry, bounds[0]), min(leave, bounds[1])
    return (entry, leave) if entry <= leave else None


def patch_targets(geometry, center, radius, step):
    """Keep structural nodes, full-chain arc samples and crop intersections.

    Phase is anchored at the full-reference chain start, before cropping.
    Keys preserve distinct coincident reference nodes; no spatial NMS occurs.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Invalid patch center")
    if not np.isfinite([radius, step]).all() or min(radius, step) <= 0:
        raise ValueError("Radius and step must be positive and finite")
    lower, upper = center - radius, center + radius
    nodes, children, critical, chains = geometry
    targets = {}

    def add(key, xyz, r, role):
        if np.all((xyz >= lower - 1e-8) & (xyz <= upper + 1e-8)):
            if key not in targets:
                targets[key] = {"xyz": xyz.tolist(), "radius": float(r), "roles": set(),
                                "reference_identity": list(key)}
            targets[key]["roles"].add(role)

    def edge_key(parent, child, t):
        if abs(t) < 1e-8:
            return ("node", parent)
        if abs(t - 1) < 1e-8:
            return ("node", child)
        return ("edge", parent, child, round(float(t), 8))

    for i in sorted(critical):
        n = nodes[i]
        xyz = np.array([n.x, n.y, n.z])
        if n.parent_id == -1:
            add(("node", i), xyz, n.radius, "root")
        if not children[i]:
            add(("node", i), xyz, n.radius, "leaf")
        if len(children[i]) > 1:
            add(("node", i), xyz, n.radius, "junction")

    for ids, xyz, radii, arc in chains:
        if np.any(xyz.max(axis=0) < lower) or np.any(xyz.min(axis=0) > upper):
            continue
        distances = np.arange(step, arc[-1] - 1e-8, step)
        indices = np.minimum(np.searchsorted(arc, distances, side="right") - 1, len(ids) - 2)
        lengths = arc[indices + 1] - arc[indices]
        if np.any(lengths <= 0):
            raise ValueError("Invalid interpolation interval")
        fractions = (distances - arc[indices]) / lengths
        positions = xyz[indices] + fractions[:, None] * (xyz[indices + 1] - xyz[indices])
        inside = np.all((positions >= lower - 1e-8) & (positions <= upper + 1e-8), axis=1)
        for k in np.flatnonzero(inside):
            j, t = indices[k], fractions[k]
            add(edge_key(ids[j], ids[j + 1], t), positions[k],
                radii[j] + t * (radii[j + 1] - radii[j]), "arc_sample")
        possible = np.all((np.maximum(xyz[:-1], xyz[1:]) >= lower) &
                          (np.minimum(xyz[:-1], xyz[1:]) <= upper), axis=1)
        for j in np.flatnonzero(possible):
            interval = clip_parameters(xyz[j], xyz[j + 1], lower, upper)
            if interval is None or interval == (0., 1.):
                continue
            # Only a box intersection is forced; internal SWC vertices are not.
            for t in set(interval):
                p = xyz[j] + t * (xyz[j + 1] - xyz[j])
                if np.any(np.isclose(p, lower, atol=1e-8, rtol=0)) or np.any(np.isclose(p, upper, atol=1e-8, rtol=0)):
                    add(edge_key(ids[j], ids[j + 1], t), p,
                        radii[j] + t * (radii[j + 1] - radii[j]), "crop_boundary")
    return list(targets.values())



def target_arrays(targets, capacity):
    """Encode loss geometry and role flags without truncating reference nodes."""
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("Capacity must be a positive integer")
    if len(targets) > capacity:
        raise ValueError(f"Target overflow: {len(targets)} targets exceed {capacity} slots")
    nodes = np.array([[i + 1, *t["xyz"], t["radius"], -1] for i, t in enumerate(targets)], dtype=np.float32).reshape(-1, 6)
    roles = np.array([sum(ROLE_BITS[r] for r in t["roles"]) for t in targets], dtype=np.uint8)
    if not np.isfinite(nodes).all() or np.any(nodes[:, 4] < 0):
        raise ValueError("Invalid target geometry")
    return nodes, roles

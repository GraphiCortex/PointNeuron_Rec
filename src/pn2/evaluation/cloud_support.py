"""Reference-curve support by source points; no reconstruction/topology claim."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import _CurveIndex


def branch_paths(morphology):
    """Maximal edge paths in an already validated SWC forest; include zero edges."""
    adjacency = [[] for _ in morphology.xyz]
    for edge_id, (u, v) in enumerate(morphology.edges):
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
    visited, paths = set(), []
    for start, neighbors in enumerate(adjacency):
        if len(neighbors) == 2:
            continue
        for current, edge in neighbors:
            if edge in visited:
                continue
            path = [start, current]
            visited.add(edge)
            while len(adjacency[current]) == 2:
                nxt, next_edge = next(pair for pair in adjacency[current] if pair[1] != edge)
                if next_edge in visited:
                    raise ValueError("Expected an acyclic validated reference")
                visited.add(next_edge)
                path.append(nxt)
                current, edge = nxt, next_edge
            paths.append(np.array(path, dtype=np.int64))
    if len(visited) != len(morphology.edges):
        raise ValueError("Reference has unvisited edges; expected a validated forest")
    return paths


def polyline_samples(vertices, step):
    """Ordered midpoint quadrature; remove roundoff-collinear subdivisions only."""
    simplified = []
    for point in vertices:
        if simplified and np.array_equal(simplified[-1], point):
            continue
        while len(simplified) >= 2:
            a, b = simplified[-1] - simplified[-2], point - simplified[-1]
            bound = 64 * np.finfo(float).eps * np.linalg.norm(a) * np.linalg.norm(b)
            if np.dot(a, b) <= 0 or np.linalg.norm(np.cross(a, b)) > bound:
                break
            simplified.pop()
        simplified.append(point)
    points, weights = [], []
    for a, b in zip(simplified[:-1], simplified[1:]):
        length = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(length / step)))
        points.append(a + ((np.arange(n) + .5) / n)[:, None] * (b - a))
        weights.append(np.full(n, length / n))
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.concatenate(weights) if weights else np.empty(0))


def longest_run(mask, weights):
    current = longest = 0.
    for missing, length in zip(mask, weights):
        current = current + float(length) if missing else 0.
        longest = max(longest, current)
    return longest


class CloudSupport:
    def __init__(self, morphology, step, dimensions_xyz):
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Quadrature step must be finite and positive")
        self.morphology, self.step = morphology, step
        self.branches = [polyline_samples(morphology.xyz[path], step) for path in branch_paths(morphology)]
        self.branches = [(p, w) for p, w in self.branches if len(w)]
        self.samples = np.concatenate([p for p, _ in self.branches]) if self.branches else np.empty((0, 3))
        self.weights = np.concatenate([w for _, w in self.branches]) if self.branches else np.empty(0)
        self.outside = int(np.count_nonzero(np.any(
            (self.samples < -.5) | (self.samples > np.asarray(dimensions_xyz) - .5), axis=1)))
        segments = morphology.segments
        segments = segments[np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 0]
        self.index = _CurveIndex(segments, 4.) if len(segments) else None

    def measure(self, xyz, tolerances, include_point_agreement=True):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError("Expected nonempty finite XYZ point cloud")
        if any(not np.isfinite(t) or t <= 0 for t in tolerances):
            raise ValueError("Tolerances must be finite and positive")
        if self.index is None:
            return {"status": "unscorable_segmentless_reference"}
        tree = cKDTree(xyz)
        distance = tree.query(self.samples)[0]
        endpoints = self.morphology.xyz[self.morphology.degrees == 1]
        junctions = self.morphology.xyz[self.morphology.degrees >= 3]
        point_distance = self.index.distances(xyz) if include_point_agreement else None
        result = {"status": "scored_reference_agreement", "reference_length": float(self.weights.sum()),
                  "branch_count": len(self.branches), "reference_node_count": self.morphology.node_count,
                  "isolated_nodes": len(self.morphology.isolated_points),
                  "zero_length_edges": int(np.count_nonzero(np.linalg.norm(
                      self.morphology.segments[:, 1] - self.morphology.segments[:, 0], axis=1) == 0)),
                  "endpoint_count": len(endpoints), "junction_count": len(junctions),
                  "outside_image_quadrature_samples": self.outside,
                  "quadrature_step": self.step, "quadrature_samples": len(self.samples),
                  "mean_curve_to_point_distance": float(np.average(distance, weights=self.weights)),
                  "tolerances": {}}
        for tolerance in tolerances:
            branch_coverage, branch_gaps, offset = [], [], 0
            for _, weights in self.branches:
                d = distance[offset:offset + len(weights)]
                offset += len(weights)
                branch_coverage.append(float(np.average(d <= tolerance, weights=weights)))
                branch_gaps.append(longest_run(d > tolerance, weights))
            result["tolerances"][str(float(tolerance))] = {
                "curve_support": float(np.average(distance <= tolerance, weights=self.weights)),
                "worst_branch_support": min(branch_coverage),
                "branch_p10_support": float(np.quantile(branch_coverage, .1)),
                "longest_unsupported_run": max(branch_gaps),
                "endpoint_support": float(np.mean(tree.query(endpoints)[0] <= tolerance)) if len(endpoints) else None,
                "junction_support": float(np.mean(tree.query(junctions)[0] <= tolerance)) if len(junctions) else None,
                "selected_point_reference_agreement": float(np.mean(point_distance <= tolerance))
                    if point_distance is not None else None,
                "branch_support": branch_coverage, "branch_longest_gap": branch_gaps}
        return result

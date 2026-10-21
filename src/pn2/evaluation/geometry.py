"""Arclength reconstruction agreement, independent of SWC node sampling density.

The integration convention follows the old PN2 representation-oracle diagnostic:
length-weighted midpoint quadrature and point-to-continuous-segment distances.
This implementation is standalone. Exact segment projections use a spatial
index with a conservative search bound instead of comparing every sample with
every segment. Integration and its sampled maximum remain approximations, not
an exact Hausdorff metric. Geometry agreement does not establish correct wiring.
"""
from __future__ import annotations

from numbers import Integral
from typing import Protocol

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree


class Morphology(Protocol):
    """Minimal geometry interface; SWC validation belongs to the input reader."""

    xyz: np.ndarray

    @property
    def edges(self) -> np.ndarray: ...


def _coordinates_and_edges(morphology, spacing):
    xyz = np.asarray(morphology.xyz, dtype=np.float64)
    raw_edges = np.asarray(morphology.edges)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Morphology coordinates must be finite with shape (N, 3)")
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError("Morphology edges must have shape (E, 2)")
    if not np.issubdtype(raw_edges.dtype, np.integer):
        raise ValueError("Morphology edge indices must be integers")
    edges = raw_edges.astype(np.int64, copy=False)
    if len(edges) and ((edges < 0).any() or (edges >= len(xyz)).any()):
        raise ValueError("Morphology edge index is outside the coordinate array")
    if len(edges) and (edges[:, 0] == edges[:, 1]).any():
        raise ValueError("Morphology cannot contain self edges")
    scaled = xyz * spacing
    if not np.isfinite(scaled).all():
        raise ValueError("Spacing produces nonfinite coordinates")
    degree = np.bincount(edges.ravel(), minlength=len(xyz))
    segments = scaled[edges]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    if not np.isfinite(lengths).all() or not np.isfinite(lengths.sum()):
        raise ValueError("Coordinates produce nonfinite curve lengths")
    return scaled, edges, degree, lengths


def _integration_segments(xyz, edges, lengths):
    """Merge only roundoff-collinear degree-two subdivisions for quadrature.

    This changes neither the geometric support nor branch incidence. It makes
    arbitrary SWC subdivisions of a straight run use the same midpoint grid.
    Original degrees, including zero-length edges, are used in structural counts.
    """
    edges = edges[lengths > 0]
    adjacency = [[] for _ in xyz]
    for edge_id, (u, v) in enumerate(edges):
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
    visited = np.zeros(len(edges), dtype=bool)
    result = []

    def extend(previous, current, incoming):
        while len(adjacency[current]) == 2:
            following, outgoing = next(pair for pair in adjacency[current] if pair[1] != incoming)
            if visited[outgoing]:
                break
            a, b = xyz[current] - xyz[previous], xyz[following] - xyz[current]
            numerical_bound = 64 * np.finfo(float).eps * np.linalg.norm(a) * np.linalg.norm(b)
            if np.dot(a, b) <= 0 or np.linalg.norm(np.cross(a, b)) > numerical_bound:
                break
            visited[outgoing] = True
            previous, current, incoming = current, following, outgoing
        return current

    for edge_id, (u, v) in enumerate(edges):
        if visited[edge_id]:
            continue
        visited[edge_id] = True
        left = extend(v, u, edge_id)
        right = extend(u, v, edge_id)
        result.append((xyz[left], xyz[right]))
    return np.asarray(result, dtype=float).reshape(-1, 2, 3)


def _sample_arclength(segments, maximum_step):
    points, weights = [], []
    for a, b in segments:
        length = float(np.linalg.norm(b - a))
        count = max(1, int(np.ceil(length / maximum_step)))
        fractions = (np.arange(count) + .5) / count
        points.append(a + fractions[:, None] * (b - a))
        weights.append(np.full(count, length / count))
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.concatenate(weights) if weights else np.empty(0))


class _CurveIndex:
    """Exact distance to positive-length line segments, with an admissible index.

    Splitting segments changes only acceleration. For a fragment with midpoint
    m and half length h, distance(p, fragment) >= distance(p, m) - h. Therefore
    searching midpoints within best_distance + maximum_half_length cannot omit
    any fragment improving the current exact distance upper bound.
    """

    def __init__(self, segments, fragment_length):
        fragments = []
        for a, b in segments:
            count = max(1, int(np.ceil(np.linalg.norm(b - a) / fragment_length)))
            fractions = np.arange(count + 1) / count
            vertices = a + fractions[:, None] * (b - a)
            fragments.append(np.stack((vertices[:-1], vertices[1:]), axis=1))
        fragments = np.concatenate(fragments)
        self.starts = fragments[:, 0]
        self.delta = fragments[:, 1] - fragments[:, 0]
        self.length2 = np.einsum("ij,ij->i", self.delta, self.delta)
        self.half_length = float(np.sqrt(self.length2.max()) / 2)
        self.tree = cKDTree(self.starts + .5 * self.delta)

    def distances(self, points, chunk_size=256):
        result = np.empty(len(points))
        for offset in range(0, len(points), chunk_size):
            batch = points[offset:offset + chunk_size]
            nearest = self.tree.query(batch, k=1)[1]
            vectors = batch - self.starts[nearest]
            fractions = np.clip(np.einsum("ij,ij->i", vectors, self.delta[nearest])
                                / self.length2[nearest], 0, 1)
            residual = self._residual(vectors, fractions, self.delta[nearest])
            best = np.einsum("ij,ij->i", residual, residual)
            radii = np.sqrt(best) + self.half_length
            # Rounded fragment midpoints/lengths and KD-tree distances can differ
            # by more than one ulp at an endpoint. Expand only the candidate
            # search (never the measured distance or scoring tolerance).
            margin = 64 * np.finfo(float).eps * (
                np.linalg.norm(batch, axis=1) + np.linalg.norm(self.tree.data[nearest], axis=1) + radii)
            radii = np.nextafter(radii + margin, np.inf)
            candidates = self.tree.query_ball_point(batch, radii)
            for index, candidate_ids in enumerate(candidates):
                # Retain the fragment furnishing the known upper bound even
                # under an unexpectedly empty spatial-index boundary query.
                ids = np.union1d(candidate_ids, [nearest[index]]).astype(np.int64)
                vectors = batch[index] - self.starts[ids]
                fractions = np.clip(np.einsum("ij,ij->i", vectors, self.delta[ids])
                                    / self.length2[ids], 0, 1)
                residual = self._residual(vectors, fractions, self.delta[ids])
                distance2 = np.einsum("ij,ij->i", residual, residual)
                best[index] = min(best[index], float(distance2.min()))
            result[offset:offset + len(batch)] = np.sqrt(best)
        return result

    @staticmethod
    def _residual(vectors, fractions, delta):
        projected = fractions[:, None] * delta
        residual = vectors - projected
        # Remove componentwise floating cancellation only. A small genuine
        # perpendicular offset is retained, regardless of segment length.
        roundoff = 8 * np.finfo(float).eps * (np.abs(vectors) + np.abs(projected))
        return np.where(np.abs(residual) <= roundoff, 0., residual)


def _directed(samples, weights, target_segments, fragment_length, chunk_size):
    if not len(weights):
        return None, {"distance_status": "no_source_curve", "mean_distance": None,
                      "maximum_distance": None, "quadrature_samples": 0}
    if not len(target_segments):
        return np.full(len(samples), np.inf), {
            "distance_status": "no_target_curve", "mean_distance": None,
            "maximum_distance": None, "quadrature_samples": len(samples)}
    distance = _CurveIndex(target_segments, fragment_length).distances(samples, chunk_size)
    return distance, {"distance_status": "finite", "mean_distance": float(np.average(distance, weights=weights)),
                      "maximum_distance": float(distance.max()), "quadrature_samples": len(samples)}


def tolerance_key(tolerance):
    return str(int(tolerance)) if float(tolerance).is_integer() else str(float(tolerance))


def _structural_support(reference, predicted, tolerances):
    result = {"reference_count": len(reference), "predicted_count": len(predicted), "tolerances": {}}
    for tolerance in tolerances:
        matched = 0
        if len(reference) and len(predicted):
            neighbors = cKDTree(reference).query_ball_tree(cKDTree(predicted), tolerance)
            rows = np.repeat(np.arange(len(reference)), [len(row) for row in neighbors])
            columns = np.array([column for row in neighbors for column in row], dtype=int)
            graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)),
                               shape=(len(reference), len(predicted)))
            matching = maximum_bipartite_matching(graph, perm_type="column")
            matched = int((matching >= 0).sum())
        precision = matched / len(predicted) if len(predicted) else None
        recall = matched / len(reference) if len(reference) else None
        f1 = (2 * matched / (len(predicted) + len(reference))
              if len(predicted) + len(reference) else None)
        result["tolerances"][tolerance_key(tolerance)] = {
            "matched": matched, "precision": precision, "recall": recall, "f1": f1}
    return result


def geometry_metrics(reference: Morphology, predicted: Morphology, spacing=(1., 1., 1.),
                     tolerances=(2., 6.), maximum_step=.5, chunk_size=256):
    """Measure continuous-curve agreement in the common scaled XYZ frame.

    Precision integrates predicted arclength supported by reference segments;
    recall reverses this direction. Isolated nodes never supply curve support.
    A reference with no positive-length edges is unscorable, not background.
    An empty/segmentless prediction against a nonempty curve has P=R=F1=0.
    Radius, node type, roots, branch identity and correctness of wiring are not
    scored here. Tolerances and step use the units after applying ``spacing``.
    """
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Spacing must contain three finite positive XYZ scales")
    tolerances = tuple(float(t) for t in tolerances)
    if not tolerances or any(not np.isfinite(t) or t < 0 for t in tolerances):
        raise ValueError("Tolerances must be finite, nonnegative and nonempty")
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("Tolerances must not repeat")
    if not np.isfinite(maximum_step) or maximum_step <= 0:
        raise ValueError("Maximum quadrature step must be finite and positive")
    if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    rxyz, redges, rdegree, rlengths = _coordinates_and_edges(reference, spacing)
    pxyz, pedges, pdegree, plengths = _coordinates_and_edges(predicted, spacing)
    # A shared translation improves numerical conditioning without alignment.
    origin = rxyz[0] if len(rxyz) else pxyz[0] if len(pxyz) else np.zeros(3)
    rxyz, pxyz = rxyz - origin, pxyz - origin
    rsegments = _integration_segments(rxyz, redges, rlengths)
    psegments = _integration_segments(pxyz, pedges, plengths)
    rsamples, rweights = _sample_arclength(rsegments, maximum_step)
    psamples, pweights = _sample_arclength(psegments, maximum_step)
    fragment_length = 16 * float(maximum_step)
    rdistance, rdirected = _directed(rsamples, rweights, psegments, fragment_length, chunk_size)
    pdistance, pdirected = _directed(psamples, pweights, rsegments, fragment_length, chunk_size)
    reference_length, predicted_length = float(rlengths.sum()), float(plengths.sum())
    result = {
        "status": "scorable" if reference_length > 0 else "unscorable_reference",
        "prediction_has_curve": predicted_length > 0,
        "reference_length": reference_length, "predicted_length": predicted_length,
        "directed": {"reference_to_prediction": rdirected, "prediction_to_reference": pdirected},
        "tolerances": {},
        "isolated": {"reference": int((rdegree == 0).sum()), "predicted": int((pdegree == 0).sum())},
        "zero_length_edges": {"reference": int((rlengths == 0).sum()),
                              "predicted": int((plengths == 0).sum())},
        "structural_support": {
            "interpretation": "One-to-one geometric support; does not establish degree agreement or correct wiring",
            "endpoints": _structural_support(rxyz[rdegree == 1], pxyz[pdegree == 1], tolerances),
            "junctions": _structural_support(rxyz[rdegree >= 3], pxyz[pdegree >= 3], tolerances)},
        "numerical": {"method": "arclength_weighted_midpoint_quadrature",
                      "distance_method": "exact_point_to_segments_with_conservative_midpoint_index",
                      "maximum_step": float(maximum_step), "index_fragment_length": fragment_length,
                      "maximum_distance_is_sampled": True, "thresholded_lengths_are_quadrature_estimates": True,
                      "collinear_degree_two_subdivisions_merged": True,
                      "spacing_xyz": spacing.tolist()},
    }
    for tolerance in tolerances:
        if reference_length == 0:
            row = dict.fromkeys(("precision", "recall", "f1", "extra_length", "missed_length"))
        elif predicted_length == 0:
            row = {"precision": 0., "recall": 0., "f1": 0., "extra_length": 0.,
                   "missed_length": reference_length}
        else:
            precision = float(pweights[pdistance <= tolerance].sum() / pweights.sum())
            recall = float(rweights[rdistance <= tolerance].sum() / rweights.sum())
            row = {"precision": precision, "recall": recall,
                   "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                   "extra_length": float(pweights[pdistance > tolerance].sum()),
                   "missed_length": float(rweights[rdistance > tolerance].sum())}
        result["tolerances"][tolerance_key(tolerance)] = row
    return result

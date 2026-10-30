"""Audit candidate: full-chain arclength nodes clipped to actual voxel boxes.

Retains the historical eight-voxel grid convention, without automatically
adopting its optional endpoint pruning. No spatial merging or radius-based
selection. This module has no image, sampler, Torch or inference dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCHEMA = "pn2-rectangular-curve-nodes-audit-v1"
EPS = 1e-8


@dataclass
class CurveNodes:
    xyz: np.ndarray
    radii: np.ndarray
    identities: list
    roles: list
    edges: np.ndarray
    reference_runs: list
    maximum_reference_to_chord_bound: float
    maximum_parametric_deviation_bound: float


def _clip(a, b, low, high):
    entry, leave = 0., 1.
    for axis, delta in enumerate(b-a):
        if abs(delta) < 1e-14:
            if not low[axis] <= a[axis] <= high[axis]:
                return None
        else:
            u, v = sorted(((low[axis]-a[axis])/delta, (high[axis]-a[axis])/delta))
            entry, leave = max(entry, u), min(leave, v)
    return (entry, leave) if entry <= leave else None


def _chain_geometry(morphology):
    children = [[] for _ in morphology.xyz]
    for child, parent in enumerate(morphology.parents):
        if parent >= 0:
            children[parent].append(child)
    critical = {i for i in range(len(children))
                if morphology.parents[i] < 0 or len(children[i]) != 1}
    chains = []
    for start in sorted(critical, key=lambda i: int(morphology.ids[i])):
        for first in sorted(children[start], key=lambda i: int(morphology.ids[i])):
            ids = [start, first]
            while ids[-1] not in critical:
                ids.append(children[ids[-1]][0])
            points = morphology.xyz[ids]
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            chains.append((ids, points, morphology.radii[ids], arc))
    return children, critical, chains


def rectangular_curve_nodes(morphology, start_xyz, stop_xyz, *, step=8., capacity=512):
    """Closed voxel-cell bounds [start-.5,stop-.5]; boundary nodes may be shared.

    The same full-chain phase is used before every crop. Root/leaf/junction IDs
    remain distinct even at identical XYZ. Curve runs are split on leaving and
    reentering the box; no edge spans a positive-length outside interval.
    Input must be a forest already validated by the PN2 SWC reader.
    """
    start, stop = np.asarray(start_xyz, dtype=float), np.asarray(stop_xyz, dtype=float)
    if (start.shape != (3,) or stop.shape != (3,) or not np.isfinite([start, stop]).all()
            or np.any(stop <= start) or not np.isfinite(step) or step <= 0
            or isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1):
        raise ValueError("Invalid rectangular box, step or target capacity")
    low, high = start-.5, stop-.5
    children, critical, chains = _chain_geometry(morphology)
    targets, edges, runs = {}, [], []
    curve_bound = parametric_bound = 0.

    def inside(point):
        return bool(np.all((point >= low-EPS) & (point <= high+EPS)))

    def structural_roles(i):
        result = set()
        if morphology.parents[i] < 0:
            result.add("root")
        if not children[i]:
            result.add("leaf")
        if len(children[i]) > 1:
            result.add("junction")
        return result

    def add(key, point, radius, roles):
        if not inside(point):
            raise ValueError("Target escaped the actual clipped reference domain")
        if key in targets:
            if not np.allclose(targets[key][0], point, atol=EPS, rtol=0):
                raise ValueError("One target identity has inconsistent positions")
            targets[key][2].update(roles)
        else:
            targets[key] = [np.asarray(point, dtype=float), float(radius), set(roles)]
        return key

    for i in sorted(critical, key=lambda j: int(morphology.ids[j])):
        if inside(morphology.xyz[i]):
            add(("node", int(morphology.ids[i])), morphology.xyz[i], morphology.radii[i], structural_roles(i))

    for ids, points, radii, arc in chains:
        length = float(arc[-1])
        if length == 0:
            if inside(points[0]):
                a, b = (("node", int(morphology.ids[i])) for i in (ids[0], ids[-1]))
                edges.append((a, b))  # Preserve distinct identities, including zero-length incidence.
            continue
        intervals = []
        for j in range(len(points)-1):
            clipped = _clip(points[j], points[j+1], low, high)
            if clipped is None:
                continue
            entry, leave = clipped
            lo = arc[j] if entry == 0 else arc[j] + entry*(arc[j+1]-arc[j])
            hi = arc[j+1] if leave == 1 else arc[j] + leave*(arc[j+1]-arc[j])
            if intervals and lo <= intervals[-1][1] + EPS:
                intervals[-1][1] = max(intervals[-1][1], float(hi))
            else:
                intervals.append([float(lo), float(hi)])

        def interpolate(distance):
            if distance == 0:
                return points[0], float(radii[0])
            if distance == length:
                return points[-1], float(radii[-1])
            j = min(int(np.searchsorted(arc, distance, side="right"))-1, len(arc)-2)
            delta = arc[j+1]-arc[j]
            if delta <= 0:
                raise ValueError("No positive interpolation interval")
            fraction = (distance-arc[j])/delta
            return points[j]+fraction*(points[j+1]-points[j]), float(radii[j]+fraction*(radii[j+1]-radii[j]))

        grid = np.arange(step, length-EPS, step)
        for lo, hi in intervals:
            distances = sorted(set([lo, hi] + grid[(grid >= lo-EPS) & (grid <= hi+EPS)].clip(lo, hi).tolist()))
            # Exact repeated arclength events were removed above. Close positions
            # are an audit finding, never grounds for merging target identities.
            events = distances
            keys = []
            for distance in events:
                point, radius = interpolate(distance)
                if distance == 0 or distance == length:
                    i = ids[0] if distance == 0 else ids[-1]
                    key, roles = ("node", int(morphology.ids[i])), structural_roles(i)
                else:
                    key = ("chain_arc", int(morphology.ids[ids[0]]), int(morphology.ids[ids[-1]]), float(distance))
                    roles = set()
                if distance > EPS and distance < length-EPS and abs(distance/step-round(distance/step)) < EPS:
                    roles.add("arc_sample")
                if np.any(np.isclose(point, low, atol=EPS, rtol=0) | np.isclose(point, high, atol=EPS, rtol=0)):
                    roles.add("crop_boundary")
                keys.append(add(key, point, radius, roles))
            run_distances = np.r_[lo, arc[(arc > lo+EPS) & (arc < hi-EPS)], hi]
            run_points = np.asarray([interpolate(d)[0] for d in run_distances])
            if hi > lo:
                runs.append(run_points)
            for a, b, da, db in zip(keys[:-1], keys[1:], events[:-1], events[1:]):
                if a == b:
                    continue
                edges.append((a, b))
                interior = arc[(arc > da+EPS) & (arc < db-EPS)]
                if not len(interior):
                    continue
                actual = np.asarray([interpolate(d)[0] for d in interior])
                first, last = targets[a][0], targets[b][0]
                delta = last-first
                fraction = np.clip((actual-first) @ delta / max(float(delta @ delta), 1e-30), 0., 1.)
                curve_bound = max(curve_bound, float(np.linalg.norm(actual-first-fraction[:, None]*delta, axis=1).max()))
                correspondence = first + ((interior-da)/(db-da))[:, None]*delta
                parametric_bound = max(parametric_bound, float(np.linalg.norm(actual-correspondence, axis=1).max()))
    if len(targets) > capacity:
        raise ValueError(f"Target overflow: {len(targets)} > {capacity}; no truncation")
    keys = list(targets)
    index = {key: i for i, key in enumerate(keys)}
    return CurveNodes(np.asarray([targets[k][0] for k in keys], dtype=float).reshape(-1, 3),
                      np.asarray([targets[k][1] for k in keys], dtype=float), keys,
                      [frozenset(targets[k][2]) for k in keys],
                      np.asarray([(index[a], index[b]) for a, b in edges], dtype=np.int64).reshape(-1, 2),
                      runs, curve_bound, parametric_bound)

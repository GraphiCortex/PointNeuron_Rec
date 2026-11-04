"""Isolated vote scoring/consolidation; no invented edges or graph F1."""
import numpy as np
from scipy.spatial import cKDTree

from .formulation import curve_samples, runs_graph, graph_segments, distances_to_geometry, support


def consolidate(points, scores, *, radius=1., cutoff=None, identities=None, cap=None):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    scores = np.asarray(scores, dtype=float)
    ids = np.arange(len(points)) if identities is None else np.asarray(identities)
    if (scores.shape != (len(points),) or ids.shape != scores.shape
            or len(np.unique(ids)) != len(ids) or not np.isfinite(points).all()
            or not np.isfinite(scores).all() or radius <= 0 or (cap is not None and cap < 0)):
        raise ValueError('Invalid vote consolidation')
    if not len(points) or cap == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((ids, -scores))
    tree = cKDTree(points)
    removed = np.zeros(len(points), dtype=bool)
    kept = []
    for i in order:
        if removed[i] or (cutoff is not None and scores[i] < cutoff):
            continue
        kept.append(int(i))
        removed[tree.query_ball_point(points[i], radius)] = True
        if cap is not None and len(kept) >= cap:
            break
    return np.asarray(kept, dtype=np.int64)


def geometric_fps(points, cap):
    """Score-free classical reduction; lexicographic start, stable input ties."""
    points = np.asarray(points).reshape(-1, 3)
    if len(points) <= cap:
        return np.arange(len(points))
    current = int(np.lexsort(points.T[::-1])[0])
    distance = np.full(len(points), np.inf)
    result = []
    for _ in range(cap):
        result.append(current)
        distance = np.minimum(distance, ((points-points[current])**2).sum(1))
        distance[result] = -1
        current = int(np.argmax(distance))
    return np.asarray(result)


def reference_geometry(reference):
    runs = reference['reference_runs']
    segments = graph_segments(runs_graph(runs))
    samples, weights = curve_samples(runs, .5)
    return segments, samples, weights


def score_points(points, reference, geometry=None, distance_function=None):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    segments, samples, weights = reference_geometry(reference) if geometry is None else geometry
    known = reference['supervision'] == 'reference_present'
    result = {'points': len(points), 'known': known, 'curve_length': float(weights.sum())}
    if not known:
        return result
    distance = (distance_function(points) if distance_function is not None
                else distances_to_geometry(points, segments, reference['target_xyz']))
    coverage, gaps, _ = support(points, samples, weights, (2., 6.))
    result.update(near_2=int((distance <= 2).sum()), near_6=int((distance <= 6).sum()),
                  covered_2=coverage['supported_length']['2.0'],
                  covered_6=coverage['supported_length']['6.0'],
                  precision_2=float((distance <= 2).mean()) if len(points) else 0.,
                  coverage_2=coverage['fraction']['2.0'] or 0.,
                  maximum_sample_gap_distance=float(gaps.max()) if len(gaps) and len(points) else None,
                  roles={})
    for role in ('root', 'leaf', 'junction', 'crop_boundary'):
        target = np.asarray([p for p, roles in zip(reference['target_xyz'], reference['target_roles'])
                             if role in roles]).reshape(-1, 3)
        d = cKDTree(points).query(target)[0] if len(points) and len(target) else np.full(len(target), np.inf)
        result['roles'][role] = {'count': len(target), 'covered': int((d <= 2).sum())}
    return result


def summarize(rows):
    """Pool within neuron before macro averaging. Unknown outputs stay separate."""
    neurons = []
    for cid in sorted({r['case_id'] for r in rows}):
        group = [r for r in rows if r['case_id'] == cid]
        known = [r['metrics'] for r in group if r['metrics']['known']]
        n = sum(r['points'] for r in known)
        length = sum(r['curve_length'] for r in known)
        role_counts = {role: {'count': sum(r['roles'][role]['count'] for r in known),
                              'covered': sum(r['roles'][role]['covered'] for r in known)}
                       for role in ('root', 'leaf', 'junction', 'crop_boundary')}
        neurons.append({'case_id': cid, 'reference_points': n,
            'unknown_points': sum(r['metrics']['points'] for r in group if not r['metrics']['known']),
            'known_boxes': len(known), 'unknown_boxes': len(group)-len(known), 'curve_length': length,
            'precision_2': sum(r['near_2'] for r in known)/n if n else 0.,
            'coverage_2': sum(r['covered_2'] for r in known)/length if length else 0.,
            'precision_6': sum(r['near_6'] for r in known)/n if n else 0.,
            'coverage_6': sum(r['covered_6'] for r in known)/length if length else 0.,
            'roles': role_counts})
    return {'neurons': neurons, 'macro': {k: float(np.mean([r[k] for r in neurons])) if neurons else 0.
            for k in ('precision_2', 'coverage_2', 'precision_6', 'coverage_6')},
            'unknown_points': sum(r['unknown_points'] for r in neurons)}

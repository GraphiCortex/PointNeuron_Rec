"""Geometry diagnostics without learned inference or reference-guided edges.

Point support depends on node spacing. Segment agreement evaluates geometric
support, not topology. A reference-connectivity oracle is labeled separately.
"""
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from .cloud_support import polyline_samples
from .geometry import _CurveIndex, _integration_segments, _sample_arclength


def curve_samples(runs, step=.5):
    values = [polyline_samples(run, step) for run in runs]
    return (np.concatenate([p for p, _ in values]) if values else np.empty((0, 3)),
            np.concatenate([w for _, w in values]) if values else np.empty(0))


def runs_graph(runs):
    xyz, edges, offset = [], [], 0
    for run in runs:
        run = np.asarray(run, dtype=float)
        xyz.extend(run)
        edges.extend((offset+i, offset+i+1) for i in range(len(run)-1))
        offset += len(run)
    return SimpleNamespace(xyz=np.asarray(xyz, dtype=float).reshape(-1, 3),
                           edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2))


def graph_segments(graph):
    xyz = np.asarray(graph.xyz, dtype=float); edges = np.asarray(graph.edges)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or (len(edges) and (edges.min() < 0 or edges.max() >= len(xyz)))):
        raise ValueError('Invalid geometric graph')
    lengths = np.linalg.norm(xyz[edges[:, 1]]-xyz[edges[:, 0]], axis=1)
    return _integration_segments(xyz, edges, lengths)


def distances_to_geometry(points, segments, isolated=None):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all(): raise ValueError('Nonfinite query geometry')
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if not np.isfinite(segments).all(): raise ValueError('Nonfinite reference geometry')
    segments = segments[np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1) > 0]
    result = np.full(len(points), np.inf)
    if len(points) and len(segments): result = _CurveIndex(segments, 4.).distances(points)
    if isolated is not None and len(isolated) and len(points):
        result = np.minimum(result, cKDTree(isolated).query(points)[0])
    return result


def support(points, samples, weights, tolerances=(2., 6., 8.)):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    samples = np.asarray(samples, dtype=float).reshape(-1, 3)
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(samples),) or not np.isfinite(points).all()
            or not np.isfinite(samples).all() or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError('Invalid support samples/weights')
    distance, indices = (cKDTree(points).query(samples) if len(points)
                         else (np.full(len(samples), np.inf), np.zeros(len(samples), dtype=int)))
    length = float(weights.sum())
    return {'curve_length': length,
        'supported_length': {str(t): float(weights[distance <= t].sum()) for t in tolerances},
        'fraction': {str(t): float(weights[distance <= t].sum()/length) if length else None for t in tolerances}}, distance, indices


def point_diagnostics(points, target_nodes, segments, samples, weights):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    target_nodes = np.asarray(target_nodes, dtype=float).reshape(-1, 3)
    result, _, _ = support(points, samples, weights)
    dc = distances_to_geometry(points, segments, target_nodes)
    dn = cKDTree(target_nodes).query(points)[0] if len(target_nodes) else np.full(len(points), np.inf)
    known = bool(len(target_nodes) or len(segments))
    result.update(points=len(points), reference_geometry_present=known,
        point_near_curve={str(t): int((dc <= t).sum()) for t in (2., 6.)} if known else None,
        point_near_grid_2=int((dn <= 2.).sum()) if len(target_nodes) else None,
        on_curve_beyond_grid_2=int(((dc <= 2.) & (dn > 2.)).sum()) if len(target_nodes) else None,
        point_curve_distance_quantiles=(np.quantile(dc, [.25, .5, .9]).tolist()
                                        if known and len(points) else None))
    return result


def segment_agreement(reference, predicted, step=.5, tolerances=(2., 6.)):
    """Two directed arclength integrals; no inferred connectivity/branch labels."""
    reference = np.asarray(reference, dtype=float).reshape(-1, 2, 3)
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2, 3)
    reference = reference[np.linalg.norm(reference[:, 1]-reference[:, 0], axis=1) > 0]
    predicted = predicted[np.linalg.norm(predicted[:, 1]-predicted[:, 0], axis=1) > 0]
    rpoints, rw = _sample_arclength(reference, step)
    ppoints, pw = _sample_arclength(predicted, step)
    rd = distances_to_geometry(rpoints, predicted);pd = distances_to_geometry(ppoints, reference)
    rl, pl = float(rw.sum()), float(pw.sum())
    rows = {}
    for t in tolerances:
        rcovered, pcovered = float(rw[rd <= t].sum()), float(pw[pd <= t].sum())
        p = pcovered/pl if pl and rl else 0. if rl else None
        r = rcovered/rl if rl else None
        rows[str(t)] = {'precision': p, 'recall': r,
            'f1': 2*p*r/(p+r) if p is not None and p+r else 0. if rl else None,
            'supported_reference_length': rcovered, 'supported_prediction_length': pcovered}
    return {'reference_length': rl, 'predicted_length': pl, 'tolerances': rows,
            'reference_has_curve': rl > 0, 'predicted_has_curve': pl > 0}


def resample_runs(runs, spacing=8., phase=0., preserve_vertices=False):
    """Reference-only controls; never changes active target packets."""
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(phase) or not 0 <= phase < spacing:
        raise ValueError('Invalid reference-control spacing/phase')
    result = []
    for run in runs:
        run = np.asarray(run, dtype=float)
        if not len(run): continue
        lengths = np.linalg.norm(np.diff(run, axis=0), axis=1)
        arc = np.r_[0., np.cumsum(lengths)]; total = arc[-1]
        if not total: result.append(run[:1].copy());continue
        positions = np.r_[0., np.arange(phase, total, spacing), total]
        if preserve_vertices: positions = np.r_[positions, arc]
        positions = np.unique(positions)
        out = np.column_stack([np.interp(positions, arc, run[:, j]) for j in range(3)])
        result.append(out)
    return result


def clip_segments(segments, low, high):
    """Exact box clipping of baseline edges, not node snapping or alignment."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    low, high = np.asarray(low), np.asarray(high)
    if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
        raise ValueError('Invalid clipping bounds')
    a = segments[:, 0];delta = segments[:, 1]-a
    lo, hi = np.zeros(len(a)), np.ones(len(a));valid = np.ones(len(a), dtype=bool)
    for axis in range(3):
        moving = delta[:, axis] != 0
        valid &= moving | ((a[:, axis] >= low[axis]) & (a[:, axis] <= high[axis]))
        u, v = np.full(len(a), -np.inf), np.full(len(a), np.inf)
        x = (low[axis]-a[moving, axis])/delta[moving, axis]
        y = (high[axis]-a[moving, axis])/delta[moving, axis]
        u[moving] = np.minimum(x, y);v[moving] = np.maximum(x, y)
        lo = np.maximum(lo, u);hi = np.minimum(hi, v)
    valid &= hi > lo
    return np.stack((a[valid]+lo[valid, None]*delta[valid],
                     a[valid]+hi[valid, None]*delta[valid]), axis=1)

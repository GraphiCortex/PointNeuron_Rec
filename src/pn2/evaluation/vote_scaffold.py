"""CPU saved-vote diagnostics and image-only graph scaffold; not a PN2 stage."""
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .local_curve import consolidate


def xyz_array(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite [N,3] coordinates')
    return p


def neighbor_counts(raw_points, retained_points, radius=2.):
    raw, retained = xyz_array(raw_points), xyz_array(retained_points)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Positive support radius required')
    return (np.asarray(cKDTree(raw).query_ball_point(retained, radius, return_length=True), dtype=np.int64)
            if len(raw) else np.zeros(len(retained), dtype=np.int64))


def support_histogram(counts, near):
    counts, near = np.asarray(counts), np.asarray(near, dtype=bool)
    if counts.ndim != 1 or counts.shape != near.shape or (counts < 0).any():
        raise ValueError('Invalid supporter/precision labels')
    def entry(mask):
        n = int(mask.sum()); good = int(near[mask].sum())
        return {'retained': n, 'within_2': good, 'beyond_2': n-good,
                'precision': good/n if n else None}
    bins = {'1': counts == 1, '2': counts == 2, '3-4': (counts >= 3) & (counts <= 4),
            '5-9': (counts >= 5) & (counts <= 9), '10+': counts >= 10}
    return {'bins': {k: entry(v) for k, v in bins.items()},
            'exact': {str(int(n)): entry(counts == n) for n in np.unique(counts)},
            'isolated': entry(counts < 3), 'supported': entry(counts >= 3)}


def isolation_decision(counts, near, source_distance):
    """Predeclared pooled rule; no free threshold/cost parameters here."""
    counts, near, distance = np.asarray(counts), np.asarray(near, bool), np.asarray(source_distance)
    if counts.shape != near.shape or distance.shape != near.shape:
        raise ValueError('Unaligned source-domain diagnostics')
    low, high, bad = counts < 3, counts >= 3, ~near
    low_p = float(near[low].mean()) if low.any() else None
    high_p = float(near[high].mean()) if high.any() else None
    bad_count = int(bad.sum())
    isolated_bad = int((low & bad).sum())
    far_bad = int(((distance > 6) & bad).sum())
    share = isolated_bad/bad_count if bad_count else 0.
    far_share = far_bad/bad_count if bad_count else 0.
    eligible = bool(low.any() and high.any() and share >= .5 and low_p < high_p and far_share < .5)
    return {'eligible': eligible, 'off_reference_retained': bad_count,
            'isolated_off_reference': isolated_bad, 'isolated_bad_share': share,
            'isolated_precision': low_p, 'supported_precision': high_p,
            'far_source_off_reference': far_bad, 'far_source_bad_share': far_share,
            'far_sources_dominate': far_share >= .5}


def supported_consolidation(points, scores, identities):
    """Sole post-failure variant: .25 / radius 2 / >=3 accepted-raw supporters."""
    p, scores = xyz_array(points), np.asarray(scores)
    raw = np.flatnonzero(scores >= .25)
    counts = neighbor_counts(p[raw], p[raw], 2.)
    eligible = raw[counts >= 3]
    kept = consolidate(p[eligible], scores[eligible], radius=2., identities=np.asarray(identities)[eligible])
    return eligible[kept]


def knn_edges(points, k):
    """Union of directed Euclidean kNN pairs, exact distance/index ties."""
    p = xyz_array(points)
    if type(k) is not int or k < 1:
        raise ValueError('Positive integer k required')
    pairs = set()
    ids = np.arange(len(p))
    for i, point in enumerate(p):
        d2 = ((p-point)**2).sum(1)
        order = np.lexsort((ids, d2))
        for j in order[order != i][:k]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def guide_at(guide, points, maximum):
    p = xyz_array(points)
    if not len(p) or maximum <= 0:
        return np.zeros(len(p))
    values = map_coordinates(guide, p[:, ::-1].T, order=1, mode='constant', cval=0., prefilter=False)
    return np.clip(np.asarray(values, dtype=float)/maximum, 0., 1.)


def image_edge_costs(points, edges, guide, maximum, step=.5, penalty=4.):
    p = xyz_array(points); edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return np.empty(0), np.empty(0)
    if step != .5 or penalty != 4.:
        raise ValueError('Only the fixed scaffold cost is supported')
    delta = p[edges[:, 1]]-p[edges[:, 0]]
    lengths = np.linalg.norm(delta, axis=1)
    counts = np.maximum(1, np.ceil(lengths/step).astype(int))
    samples = np.concatenate([p[a]+((np.arange(n)+.5)/n)[:, None]*v
                              for (a, _), v, n in zip(edges, delta, counts)])
    q = guide_at(guide, samples, maximum)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    means = np.add.reduceat(q, offsets)/counts
    return lengths*(1+penalty*(1-means)), means


def minimum_spanning_forest(points, edges, costs):
    p = xyz_array(points); edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    costs = np.asarray(costs, dtype=float)
    if costs.shape != (len(edges),) or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Invalid image-derived edge costs')
    if len(edges) and (edges.min() < 0 or edges.max() >= len(p) or (edges[:, 0] == edges[:, 1]).any()):
        raise ValueError('Invalid candidate endpoints')
    parent = list(range(len(p)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    selected = []
    for i in np.lexsort((edges[:, 1], edges[:, 0], costs)):
        u, v = map(int, edges[i]); a, b = find(u), find(v)
        if a != b:
            parent[b] = a
            selected.append(int(i))
    return edges[np.asarray(selected, dtype=np.int64)], len({find(i) for i in range(len(p))})


def export_forest(path, points, edges, node_guide):
    """No biological type/radius/root claim. No reference accepted by this API."""
    p = xyz_array(points); edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    q = np.asarray(node_guide)
    if q.shape != (len(p),):
        raise ValueError('Misaligned image-derived root evidence')
    adjacency = [[] for _ in p]
    for u, v in edges:
        adjacency[u].append(int(v)); adjacency[v].append(int(u))
    parent = np.full(len(p), -2, dtype=int)
    for seed in range(len(p)):
        if parent[seed] != -2:
            continue
        component, seen, stack = [], {seed}, [seed]
        while stack:
            u = stack.pop(); component.append(u)
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        root = min(component, key=lambda i: (-q[i], i))
        parent[root] = -1; stack = [root]
        while stack:
            u = stack.pop()
            for v in sorted(adjacency[u], reverse=True):
                if parent[v] == -2:
                    parent[v] = u; stack.append(v)
                elif v != parent[u]:
                    raise ValueError('Export requires a forest, not a cyclic graph')
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write('# Measurement scaffold: type=0, radius=1 UNSUPERVISED; image-guide roots\n')
        for i, ((x, y, z), predecessor) in enumerate(zip(p, parent)):
            stream.write(f'{i+1} 0 {x:.17g} {y:.17g} {z:.17g} 1 {predecessor+1 if predecessor>=0 else -1}\n')
    return {'roots': np.flatnonzero(parent == -1).tolist(), 'nodes': len(p), 'edges': len(edges),
            'isolated_nodes': sum(not a for a in adjacency), 'radius': 'placeholder_1_unsupervised',
            'rooting': 'highest_image_guide_per_component_not_soma_detection'}

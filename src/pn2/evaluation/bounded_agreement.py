"""Exact 2/6-voxel agreement decisions on the frozen midpoint quadrature.

Distances beyond six voxels are censored, never reported as exact distances.
The original general-purpose distance and segment scorers are unchanged.
"""
import numpy as np

from .geometry import _CurveIndex, _sample_arclength


def positive_segments(segments):
    s = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite curve')
    return s[np.linalg.norm(s[:, 1]-s[:, 0], axis=1) > 0]


def bounded_distances(points, index, chunk_size=64):
    """Exact minimum when <=6; infinity otherwise. No changed score tolerance.

    A fragment within six voxels has midpoint distance <=6+half_length.
    Expand only the candidate search for floating-point boundary error, as in
    the original index. Reuse its fragment construction and residual arithmetic.
    Batch all candidate projections instead of looping over every query point.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(p).all() or chunk_size < 1:
        raise ValueError('Invalid queries/chunk size')
    result = np.full(len(p), np.inf)
    if index is None or not len(p):
        return result
    radius = 6.+index.half_length
    target_norm = float(np.linalg.norm(index.tree.data, axis=1).max())
    for offset in range(0, len(p), chunk_size):
        batch = p[offset:offset+chunk_size]
        margin = 128*np.finfo(float).eps*(np.linalg.norm(batch, axis=1)+target_norm+radius+1)
        candidates = index.tree.query_ball_point(batch, np.nextafter(radius+margin, np.inf))
        counts = np.fromiter((len(ids) for ids in candidates), dtype=int, count=len(batch))
        rows = np.flatnonzero(counts)
        if not len(rows):
            continue
        ids = np.concatenate([candidates[i] for i in rows]).astype(np.int64)
        queries = np.repeat(rows, counts[rows])
        vectors = batch[queries]-index.starts[ids]
        fractions = np.clip(np.einsum('ij,ij->i', vectors, index.delta[ids])/index.length2[ids], 0, 1)
        residual = index._residual(vectors, fractions, index.delta[ids])
        d2 = np.einsum('ij,ij->i', residual, residual)
        starts = np.r_[0, np.cumsum(counts[rows])[:-1]]
        distance = np.sqrt(np.minimum.reduceat(d2, starts))
        result[offset+rows] = np.where(distance <= 6., distance, np.inf)
    return result


class BoundedAgreement:
    """Cache only supplied reference geometry; never builds inference graphs."""
    def __init__(self, reference):
        self.reference = positive_segments(reference)
        self.rpoints, self.rw = _sample_arclength(self.reference, .5)
        self.index = _CurveIndex(self.reference, 4.) if len(self.reference) else None

    def score(self, predicted):
        predicted = positive_segments(predicted)
        ppoints, pw = _sample_arclength(predicted, .5)
        target = _CurveIndex(predicted, 4.) if len(predicted) else None
        rd = bounded_distances(self.rpoints, target)
        pd = bounded_distances(ppoints, self.index)
        rl, pl = float(self.rw.sum()), float(pw.sum())
        rows = {}
        for t in (2., 6.):
            rc, pc = float(self.rw[rd <= t].sum()), float(pw[pd <= t].sum())
            precision = pc/pl if pl and rl else 0. if rl else None
            recall = rc/rl if rl else None
            rows[str(t)] = {'precision': precision, 'recall': recall,
                'f1': 2*precision*recall/(precision+recall) if precision is not None and precision+recall else 0. if rl else None,
                'supported_reference_length': rc, 'supported_prediction_length': pc}
        return {'reference_length': rl, 'predicted_length': pl, 'tolerances': rows,
                'reference_has_curve': rl > 0, 'predicted_has_curve': pl > 0}

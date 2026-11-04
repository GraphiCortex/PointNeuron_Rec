"""Reference-only local curve labels; never imported by the model forward path."""
import numpy as np
from scipy.spatial import cKDTree


class ProjectionIndex:
    """Exact projections inside a bounded tube, including near-tie alternatives.

    Segment midpoint distance <= point/segment distance + half segment length.
    Subdivide solely for an admissible bounded KD-tree candidate search. Outside
    the support tube, distances are censored at radius+1, explicitly not exact.
    """

    def __init__(self, segments, isolated=None, fragment_length=4.):
        segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
        if not np.isfinite(segments).all() or fragment_length <= 0:
            raise ValueError('Invalid reference segments')
        if isolated is not None:
            p = np.asarray(isolated, dtype=float).reshape(-1, 3)
            if not np.isfinite(p).all():
                raise ValueError('Invalid isolated geometry')
            segments = np.concatenate((segments, np.stack((p, p), axis=1)))
        self.segments = segments
        self.incident = {}
        pieces, parents = [], []
        for parent, (a, b) in enumerate(segments):
            count = max(1, int(np.ceil(np.linalg.norm(b-a)/fragment_length)))
            v = a + np.arange(count+1)[:, None]/count*(b-a)
            pieces.append(np.stack((v[:-1], v[1:]), axis=1))
            parents.extend([parent]*count)
            self.incident.setdefault(tuple(a), []).append(b-a)
            self.incident.setdefault(tuple(b), []).append(a-b)
        if not pieces:
            raise ValueError('A label index requires supplied geometry')
        fragments = np.concatenate(pieces)
        self.starts = fragments[:, 0]
        self.delta = fragments[:, 1]-fragments[:, 0]
        self.length2 = np.einsum('ij,ij->i', self.delta, self.delta)
        self.half = float(np.sqrt(self.length2.max())/2)
        self.tree = cKDTree(self.starts + self.delta/2)
        self.parents = np.asarray(parents)

    def labels(self, points, *, support_radius=8., offset_radius=6.,
               tie_distance=.25, tie_separation=.5):
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or not 0 < offset_radius < support_radius or tie_distance < 0 or tie_separation < 0):
            raise ValueError('Invalid label coordinates or radii')
        distances = np.full(len(points), support_radius+1.)
        projection = points.copy()
        ambiguous = np.zeros(len(points), dtype=bool)
        # Include every possible alternative within tie_distance of a supported
        # nearest projection, even when that alternative falls outside the tube.
        radius = support_radius + tie_distance + self.half
        margin = 128*np.finfo(float).eps*(np.linalg.norm(points, axis=1)+radius+1)
        candidates = self.tree.query_ball_point(points, radius+margin)
        for i, raw in enumerate(candidates):
            if not raw:
                continue
            ids = np.unique(self.parents[raw])
            starts = self.segments[ids, 0]
            delta = self.segments[ids, 1]-starts
            fractions = np.clip(np.einsum('ij,ij->i', points[i]-starts, delta)
                                / np.maximum((delta*delta).sum(1), 1e-30), 0., 1.)
            q = starts + fractions[:, None]*delta
            d = np.linalg.norm(points[i]-q, axis=1)
            best = int(np.argmin(d))
            if d[best] > support_radius:
                continue
            distances[i] = d[best]
            projection[i] = q[best]
            alternatives = d <= d[best]+tie_distance+1e-12
            # A subdivision endpoint on a continuous branch is not an
            # alternative local minimum if moving along an incident segment
            # reduces distance. Fragment/SWC subdivision must not create ties.
            for j in np.flatnonzero(alternatives):
                if fractions[j] == 0 or fractions[j] == 1:
                    vertex = starts[j] if fractions[j] == 0 else self.segments[ids[j], 1]
                    directions = np.asarray(self.incident[tuple(vertex)])
                    if np.any(directions @ (points[i]-vertex) > 1e-10):
                        alternatives[j] = False
            ambiguous[i] = bool(np.any(np.linalg.norm(q[alternatives]-q[best], axis=1)
                                       > tie_separation+1e-12))
        return {'support': np.maximum(0., 1-distances/support_radius).astype(np.float32),
                'offset': (projection-points).astype(np.float32),
                'offset_mask': (distances <= offset_radius) & ~ambiguous,
                'ambiguous': ambiguous, 'source_distance': distances.astype(np.float32),
                'projection': projection}


def distance_bands(distance):
    """0: [0,2), 1: [2,4), 2: [4,6], 3: (6,8), 4: [8,inf)."""
    d = np.asarray(distance)
    return np.where(d < 2, 0, np.where(d < 4, 1, np.where(d <= 6, 2,
                    np.where(d < 8, 3, 4)))).astype(np.int64)

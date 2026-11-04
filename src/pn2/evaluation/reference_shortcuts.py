"""Reference-only continuous path diagnostic; never used to select graph edges."""
import numpy as np

from pn2.supervision.local_curve import ProjectionIndex


class ReferencePaths:
    def __init__(self, morphology):
        self.xyz = morphology.xyz
        self.edges = morphology.edges
        self.parents = morphology.parents
        self.lengths = np.linalg.norm(self.xyz[self.edges[:, 1]]-self.xyz[self.edges[:, 0]], axis=1)
        self.index = ProjectionIndex(morphology.segments) if len(self.edges) else None
        children = [[] for _ in self.xyz]
        for v, parent in enumerate(self.parents):
            if parent >= 0:
                children[parent].append(v)
        n = len(self.xyz)
        self.depth = np.zeros(n, dtype=int)
        self.arc = np.zeros(n)
        self.component = np.full(n, -1, dtype=int)
        for root in np.flatnonzero(self.parents < 0):
            self.component[root] = root
            stack = [int(root)]
            while stack:
                u = stack.pop()
                for v in children[u]:
                    self.depth[v] = self.depth[u]+1
                    self.arc[v] = self.arc[u]+np.linalg.norm(self.xyz[v]-self.xyz[u])
                    self.component[v] = root
                    stack.append(v)
        self.up = np.empty((max(1, n.bit_length()), n), dtype=int)
        self.up[0] = np.where(self.parents < 0, np.arange(n), self.parents)
        for j in range(1, len(self.up)):
            self.up[j] = self.up[j-1, self.up[j-1]]

    def lca(self, a, b):
        if self.component[a] != self.component[b]:
            return None
        if self.depth[a] < self.depth[b]:
            a, b = b, a
        difference = int(self.depth[a]-self.depth[b])
        for j in range(len(self.up)):
            if difference & (1 << j):
                a = int(self.up[j, a])
        if a == b:
            return a
        for j in range(len(self.up)-1, -1, -1):
            if self.up[j, a] != self.up[j, b]:
                a, b = int(self.up[j, a]), int(self.up[j, b])
        return int(self.up[0, a])

    def route(self, a, b):
        ancestor = self.lca(a, b)
        if ancestor is None:
            return np.inf
        return float(self.arc[a]+self.arc[b]-2*self.arc[ancestor])

    def path(self, a, b):
        """Continuous path length plus endpoint route for box-boundary checks."""
        ea, fa, _ = a
        eb, fb, _ = b
        if ea == eb:
            return abs(fa-fb)*self.lengths[ea], None
        options = []
        for va, da in zip(self.edges[ea], (fa*self.lengths[ea], (1-fa)*self.lengths[ea])):
            for vb, db in zip(self.edges[eb], (fb*self.lengths[eb], (1-fb)*self.lengths[eb])):
                options.append((float(da+db+self.route(va, vb)), int(va), int(vb)))
        best = min(options)
        return best[0], best[1:]

    def projection(self, point):
        if self.index is None:
            return None, 'off_reference'
        idx = self.index
        raw = idx.tree.query_ball_point(point, 2.25+idx.half+1e-9)
        if not raw:
            return None, 'off_reference'
        ids = np.unique(idx.parents[raw])
        starts = idx.segments[ids, 0]
        delta = idx.segments[ids, 1]-starts
        fractions = np.clip(np.einsum('ij,ij->i', point-starts, delta)/np.maximum((delta*delta).sum(1), 1e-30), 0, 1)
        projections = starts+fractions[:, None]*delta
        distances = np.linalg.norm(point-projections, axis=1)
        best = int(np.argmin(distances))
        if distances[best] > 2.+1e-12:
            return None, 'off_reference'
        selected = (int(ids[best]), float(fractions[best]), projections[best])
        for j in np.flatnonzero(distances <= distances[best]+.25+1e-12):
            if fractions[j] in (0., 1.):
                vertex = idx.segments[ids[j], int(fractions[j])]
                directions = np.asarray(idx.incident[tuple(vertex)])
                if np.any(directions @ (point-vertex) > 1e-10):
                    continue
            alternative = (int(ids[j]), float(fractions[j]), projections[j])
            if self.path(selected, alternative)[0] > .5+1e-12:
                return None, 'ambiguous'
        return selected, None

    def in_box(self, a, b, route, low, high):
        def inside(p):
            return bool(np.all((p >= low) & (p <= high)))
        if not inside(a[2]) or not inside(b[2]):
            return False
        if route is None:
            return True
        u, v = route
        ancestor = self.lca(u, v)
        for node in (u, v):
            while node != ancestor:
                if not inside(self.xyz[node]):
                    return False
                node = int(self.parents[node])
        return inside(self.xyz[ancestor])

    def score(self, points, edges, low, high):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        edges = np.asarray(edges, dtype=int).reshape(-1, 2)
        projections = [self.projection(p) for p in points]
        rows = []
        for u, v in edges:
            length = float(np.linalg.norm(points[u]-points[v]))
            a, reason_a = projections[u]
            b, reason_b = projections[v]
            row = {'u': int(u), 'v': int(v), 'length': length}
            if length <= 1e-12:
                reason = 'degenerate'
            elif reason_a or reason_b:
                reason = 'ambiguous' if 'ambiguous' in (reason_a, reason_b) else 'off_reference'
            elif self.component[self.edges[a[0], 0]] != self.component[self.edges[b[0], 0]]:
                reason = 'different_reference_components_unknown'
            else:
                distance, route = self.path(a, b)
                if not self.in_box(a, b, route, np.asarray(low), np.asarray(high)):
                    reason = 'reference_path_leaves_box_unknown'
                else:
                    reason = 'scorable'
                    row.update(reference_path_length=float(distance), path_to_edge_ratio=float(distance/length),
                               excess_over_twice_length=float(distance-2*length), shortcut=bool(distance > 2*length+4.))
            row['status'] = reason
            rows.append(row)
        categories = {name: {'count': sum(r['status'] == name for r in rows),
                            'length': sum(r['length'] for r in rows if r['status'] == name)}
                      for name in ('scorable', 'off_reference', 'ambiguous', 'degenerate',
                                   'different_reference_components_unknown', 'reference_path_leaves_box_unknown')}
        scorable = categories['scorable']['count']
        flags = sum(r.get('shortcut', False) for r in rows)
        return {'categories': categories, 'flagged': flags,
                'flagged_length': sum(r['length'] for r in rows if r.get('shortcut', False)),
                'flag_fraction_of_scorable': flags/scorable if scorable else None,
                'edges': rows, 'criterion': 'reference_path_length > 2*edge_length + 4',
                'biological_wrong_join_rate': None, 'used_for_edge_selection': False}

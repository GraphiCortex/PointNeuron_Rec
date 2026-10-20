"""Experimental training targets with branch-aware, guarded sample removal.

The legacy target generator and every inference path remain unchanged. Decisions
use full reference chains before cropping; distinct nearby branches never merge.
"""
from dataclasses import dataclass

import numpy as np

from pointneuron.data.curve_targets import curve_geometry, patch_targets
from pointneuron.graph.representation_oracle import ReferenceGraph, target_graph, topology_comparison

TARGET_SCHEMA = 'pointneuron2-canonical-curve-targets-v2'


def segment_error(points, start, end):
    delta = end - start
    fraction = np.clip((points - start) @ delta / max(float(delta @ delta), 1e-30), 0., 1.)
    return np.linalg.norm(points - start - fraction[:, None] * delta, axis=1)


def interpolate(xyz, radii, arc, distance):
    index = min(int(np.searchsorted(arc, distance, side='right')) - 1, len(arc) - 2)
    length = arc[index + 1] - arc[index]
    if length <= 0:
        raise ValueError('Invalid interpolation interval')
    fraction = (distance - arc[index]) / length
    return xyz[index] + fraction * (xyz[index + 1] - xyz[index]), float(radii[index] + fraction * (radii[index + 1] - radii[index])), index, fraction


def identity(parent, child, fraction):
    if abs(fraction) < 1e-8:
        return ('node', parent)
    if abs(fraction - 1.) < 1e-8:
        return ('node', child)
    return ('edge', parent, child, round(float(fraction), 8))


def arc_position(key, ids, arc):
    if key[0] == 'node':
        return float(arc[ids.index(key[1])])
    index = next(i for i, pair in enumerate(zip(ids[:-1], ids[1:])) if pair == tuple(key[1:3]))
    return float(arc[index] + key[3] * (arc[index + 1] - arc[index]))


def span_error(chain, lo, hi, sample_step=None):
    """Bound the change to the existing target polyline, without quadrature.

The straight-segment capsule is convex, so checking all polyline vertices
    bounds the entire intervening target polyline. Radius differences are
    piecewise linear in the reference arclength parameter. With sample_step=None
    this instead measures absolute error against every SWC vertex, for auditing.
"""
    _, xyz, radii, arc = chain
    a, ra = (xyz[0], float(radii[0])) if abs(lo) < 1e-8 else interpolate(xyz, radii, arc, lo)[:2]
    # The terminal vertex may follow a zero-length SWC edge.
    b, rb = (xyz[-1], float(radii[-1])) if abs(hi - arc[-1]) < 1e-8 else interpolate(xyz, radii, arc, hi)[:2]
    if sample_step is None:
        mask = (arc >= lo) & (arc <= hi)
        interior_xyz, interior_radius, interior_arc = xyz[mask], radii[mask], arc[mask]
    else:
        interior_arc = np.arange(sample_step, arc[-1] - 1e-8, sample_step)
        interior_arc = interior_arc[(interior_arc > lo + 1e-6) & (interior_arc < hi - 1e-6)]
        samples = [interpolate(xyz, radii, arc, s) for s in interior_arc]
        interior_xyz = np.array([s[0] for s in samples]).reshape(-1, 3)
        interior_radius = np.array([s[1] for s in samples])
    points = np.vstack((a, interior_xyz, b))
    distances = np.r_[lo, interior_arc, hi]
    radius = np.r_[ra, interior_radius, rb]
    expected = ra + (distances - lo) / max(hi - lo, 1e-30) * (rb - ra)
    return float(segment_error(points, a, b).max()), float(np.abs(radius - expected).max())


@dataclass
class CanonicalPatch:
    targets: list
    graph: ReferenceGraph
    removed: list
    certificates: list
    topology: dict


class CanonicalTargetBuilder:
    def __init__(self, tree, *, step=8., endpoint_fraction=.25, maximum_deviation=.25, maximum_radius_deviation=.25):
        values = (step, endpoint_fraction, maximum_deviation, maximum_radius_deviation)
        if not np.isfinite(values).all() or step <= 0 or not 0 < endpoint_fraction < .5 or min(maximum_deviation, maximum_radius_deviation) < 0:
            raise ValueError('Invalid canonical target policy')
        self.tree, self.geometry, self.step = tree, curve_geometry(tree), float(step)
        self.policy = {'schema': TARGET_SCHEMA, 'step': float(step), 'endpoint_fraction': float(endpoint_fraction),
            'maximum_deviation': float(maximum_deviation), 'maximum_radius_deviation': float(maximum_radius_deviation),
            'guard_reference': 'additional_change_to_existing_target_polyline',
            'distance_units': 'existing voxel-coordinate Euclidean units; not calibrated physical distance'}
        self.decisions, self.removed = [], {}
        for chain in self.geometry[3]:
            ids, xyz, radii, arc = chain
            samples = np.arange(step, arc[-1] - 1e-8, step)
            if not len(samples) or arc[-1] - samples[-1] >= step * endpoint_fraction:
                continue
            _, _, index, fraction = interpolate(xyz, radii, arc, samples[-1])
            key = identity(ids[index], ids[index + 1], fraction)
            if key[0] == 'node' and key[1] in self.geometry[2]:
                continue  # Never erase a structural identity, even at coincident coordinates.
            lo = float(samples[-2]) if len(samples) > 1 else 0.
            error, radius_error = span_error(chain, lo, float(arc[-1]), self.step)
            absolute_error, absolute_radius_error = span_error(chain, lo, float(arc[-1]))
            accepted = error <= maximum_deviation + 1e-8 and radius_error <= maximum_radius_deviation + 1e-8
            decision = {'identity': list(key), 'chain_endpoints': [ids[0], ids[-1]],
                'endpoint_arclength_gap': float(arc[-1] - samples[-1]), 'splice_arclength': float(arc[-1] - lo),
                'maximum_curve_deviation': error, 'maximum_radius_deviation': radius_error,
                'absolute_source_curve_deviation': absolute_error, 'absolute_source_radius_deviation': absolute_radius_error,
                'remove': bool(accepted), 'reason': 'redundant_endpoint_sample' if accepted else 'curve_or_radius_guard'}
            self.decisions.append(decision)
            if accepted:
                if key in self.removed:
                    raise ValueError('Optional identity unexpectedly belongs to multiple chains')
                self.removed[key] = chain

    def patch(self, center, radius):
        targets = patch_targets(self.geometry, center, radius, self.step)
        original = target_graph(self.tree, [(center, radius)], self.step)
        if original.identities != [tuple(t['reference_identity']) for t in targets]:
            raise ValueError('Legacy target/graph ordering differs')
        remove = {i for i, t in enumerate(targets) if tuple(t['reference_identity']) in self.removed and t['roles'] == {'arc_sample'}}
        incoming, outgoing = [[] for _ in targets], [[] for _ in targets]
        for u, v in original.edges:
            outgoing[u].append(v); incoming[v].append(u)
        certificates = []
        for i in sorted(remove):
            if len(incoming[i]) != 1 or len(outgoing[i]) != 1:
                raise ValueError('Only directed degree-two optional samples may be removed')
            key = original.identities[i]; chain = self.removed[key]
            lo = arc_position(original.identities[incoming[i][0]], chain[0], chain[3])
            hi = arc_position(original.identities[outgoing[i][0]], chain[0], chain[3])
            error, radius_error = span_error(chain, lo, hi, self.step)
            if error > 2 * self.policy['maximum_deviation'] + 1e-6 or radius_error > 2 * self.policy['maximum_radius_deviation'] + 1e-6:
                raise ValueError('Cropped splice exceeds conservative local geometry/radius bound')
            certificates.append({'identity': list(key), 'maximum_curve_deviation': error, 'maximum_radius_deviation': radius_error})
        keep = [i for i in range(len(targets)) if i not in remove]
        mapping = {i: j for j, i in enumerate(keep)}; edges = []
        for u in keep:
            for v in outgoing[u]:
                while v in remove:
                    v = outgoing[v][0]
                edges.append((mapping[u], mapping[v]))
        graph = ReferenceGraph([original.identities[i] for i in keep], original.xyz[keep], original.radii[keep], sorted(edges))
        comparison = topology_comparison(original, graph)
        if not comparison['passed']:
            raise ValueError('Canonicalization changed structural incidence or crop components')
        return CanonicalPatch([targets[i] for i in keep], graph, [list(original.identities[i]) for i in sorted(remove)], certificates, comparison)

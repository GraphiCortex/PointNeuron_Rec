"""Reference-only point proximity along clipped segments; not graph connectivity."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def segment_probe(segments, spacing=(1., 1., 1.), step=1.):
    """Arc-length weighted midpoint probes, with fixed sampling before evaluation."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    spacing = np.asarray(spacing, dtype=float)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not np.isfinite(segments).all() or not np.isfinite(step) or step <= 0):
        raise ValueError('Invalid segment sampling parameters')
    pieces, weights, slices = [], [], []
    offset = 0
    for start, end in segments:
        length = np.linalg.norm((end - start) * spacing)
        if not length:
            continue
        count = max(1, int(np.ceil(length / step)))
        fractions = (np.arange(count) + .5) / count
        pieces.append((start + fractions[:, None] * (end - start)) * spacing)
        weights.append(np.full(count, length / count))
        slices.append((offset, offset + count)); offset += count
    return {'points': np.concatenate(pieces) if pieces else np.empty((0, 3)),
            'weights': np.concatenate(weights) if weights else np.empty(0), 'slices': slices,
            'spacing': spacing, 'step': step}


def segment_support(probe, xyz, distance=3.):
    """Report proximity gaps; branch edges can be unsupported despite high means."""
    xyz = np.asarray(xyz, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or not np.isfinite(distance) or distance <= 0):
        raise ValueError('Invalid support coordinates or distance')
    if not len(probe['points']):
        return None
    distances = (cKDTree(xyz * probe['spacing']).query(probe['points'], workers=1)[0]
                 if len(xyz) else np.full(len(probe['points']), np.inf))
    supported = distances <= distance
    weights = probe['weights']; worst_gap = 0.; complete = 0
    for start, end in probe['slices']:
        run = 0.
        for present, weight in zip(supported[start:end], weights[start:end]):
            run = 0. if present else run + weight
            worst_gap = max(worst_gap, run)
        complete += int(supported[start:end].all())
    return {'reference_length': float(weights.sum()), 'supported_length': float(weights[supported].sum()),
            'supported_length_fraction': float(weights[supported].sum() / weights.sum()),
            'max_unsupported_run_length': float(worst_gap), 'segments': len(probe['slices']),
            'fully_supported_segments': complete, 'probe_step': probe['step'], 'distance': distance}

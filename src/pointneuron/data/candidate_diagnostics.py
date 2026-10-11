"""Image-only point-pool alternatives for controlled development diagnostics."""
from __future__ import annotations

import numpy as np

from pointneuron.data.manual_review import farthest_indices
from pointneuron.data.medial_sampling import MedialSamplingConfig, medial_distances, medial_probabilities
from pointneuron.data.training_cache import brightest_cell_representatives, indices_to_point_array


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def image_candidate_indices(indices, data, width, height, count, method, seed, cell_size=8, spacing_xyz=(1., 1., 1.)):
    """Use only foreground coordinates/intensities; preserve raw voxels and padding.

    Uniform follows existing cache selection. Spatial uses the brightest voxel
    per cell, farthest-point coverage of those representatives, then fills any
    remaining slots uniformly from other foreground voxels.
    """
    positive_integer(count, 'count'); positive_integer(cell_size, 'cell_size')
    positive_integer(width, 'width'); positive_integer(height, 'height')
    if method not in ('uniform', 'spatial'):
        raise ValueError('Unknown candidate method')
    indices, data = np.asarray(indices), np.asarray(data)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or data.ndim != 1
            or np.any(indices < 0) or np.any(indices >= data.size) or len(np.unique(indices)) != len(indices)):
        raise ValueError('Expected unique in-range foreground indices and flat image data')
    spacing = MedialSamplingConfig(spacing_xyz=spacing_xyz).spacing_xyz
    if not len(indices):
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(data[indices]).all():
        raise ValueError('Nonfinite foreground intensity')
    rng = np.random.default_rng(seed)
    if len(indices) < count:
        return np.sort(np.r_[indices, rng.choice(indices, count - len(indices))])
    if method == 'uniform':
        return np.sort(rng.choice(indices, count, replace=False))
    representatives = brightest_cell_representatives(indices, data, width, height, cell_size)
    xyz = indices_to_point_array(representatives, data, width, height)[:, :3]
    chosen = representatives[farthest_indices(xyz, count, spacing)]
    if len(chosen) < count:
        remaining = np.setdiff1d(indices, chosen)
        chosen = np.r_[chosen, rng.choice(remaining, count - len(chosen), replace=False)]
    return np.sort(chosen)


def select_candidate_rows(points, guide_xyz, count, method, seed, medial=MedialSamplingConfig()):
    """Fixed CPU probes; no optimizer, learned checkpoint or reference input.

    Random and fixed-medial use identical seeded Gumbel noise on the same pool.
    The latter uses the existing NumPy exponential/uniform prior. It implements
    the same sampling distribution, not PyTorch's bit-identical random stream.
    """
    positive_integer(count, 'count')
    if method not in ('random', 'fixed_medial', 'farthest'):
        raise ValueError('Unknown selector')
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError('Expected finite XYZ/intensity candidate rows')
    if not len(points):
        return np.empty(0, dtype=np.int64)
    if count > len(points):
        raise ValueError('Selection budget exceeds pool rows')
    if method == 'farthest':
        return farthest_indices(points[:, :3], count, medial.spacing_xyz)
    probabilities = (medial_probabilities(medial_distances(points[:, :3], guide_xyz, medial.spacing_xyz), medial)
                     if method == 'fixed_medial' else np.full(len(points), 1 / len(points)))
    rng = np.random.default_rng(seed)
    with np.errstate(divide='ignore'):
        keys = np.log(probabilities) + rng.gumbel(size=len(points))
    return np.sort(np.argpartition(keys, -count)[-count:])

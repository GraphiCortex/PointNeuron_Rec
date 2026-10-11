"""Image-only allocation between a bright core and additional permissive voxels."""
from __future__ import annotations

import numpy as np

from pointneuron.data.candidate_diagnostics import positive_integer


def mixed_candidate_indices(bright_indices, permissive_indices, count, seed, bright_fraction=.5):
    """Sample disjoint strata without replacement, then pad only if union is small.

    Strong voxels outside the permissive mask remain eligible. Weak-only means
    permissive minus bright, not verified neuronal signal. Shortages transfer to
    the other stratum; a small union is fully retained before uniform padding.
    No skeleton, SWC, reference metric or intensity transform enters selection.
    """
    positive_integer(count, 'count')
    if not np.isfinite(bright_fraction) or not 0 <= bright_fraction <= 1:
        raise ValueError('bright_fraction must be finite and in [0, 1]')
    bright, permissive = np.asarray(bright_indices), np.asarray(permissive_indices)
    for indices in (bright, permissive):
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or len(np.unique(indices)) != len(indices)):
            raise ValueError('Expected unique nonnegative integer voxel indices')
    weak = np.setdiff1d(permissive, bright)
    union = np.union1d(bright, weak)
    rng = np.random.default_rng(seed)
    if not len(union):
        return np.empty(0, dtype=np.int64)
    if len(union) < count:
        return np.sort(np.r_[union, rng.choice(union, count - len(union), replace=True)])
    quota = int(np.floor(count * bright_fraction + .5))
    n_bright = min(quota, len(bright))
    n_weak = min(count - n_bright, len(weak))
    n_bright += min(count - n_bright - n_weak, len(bright) - n_bright)
    chosen = np.r_[rng.choice(bright, n_bright, replace=False), rng.choice(weak, n_weak, replace=False)]
    return np.sort(chosen.astype(np.int64, copy=False))

"""Isolated image-only Phase 1 selector; production samplers are unchanged."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARMS = ("threshold_random", "expanded_random", "expanded_coverage", "guided_context")


@dataclass(frozen=True)
class FieldSamplingConfig:
    field_threshold: float = .5
    cell_size: int = 8
    guide_fraction: float = .5

    def __post_init__(self):
        if (not np.isfinite(self.field_threshold) or not 0 <= self.field_threshold <= 1
                or isinstance(self.cell_size, bool) or not isinstance(self.cell_size, int) or self.cell_size < 1
                or not np.isfinite(self.guide_fraction) or not 0 <= self.guide_fraction <= 1):
            raise ValueError("Invalid field sampling settings")


@dataclass(frozen=True)
class CandidatePools:
    foreground: np.ndarray
    guide: np.ndarray
    expanded: np.ndarray
    shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class SelectedCloud:
    points: np.ndarray
    voxel_indices: np.ndarray
    field_scores: np.ndarray
    selection_role: np.ndarray  # 1: guide; 0: ordinary/context image point
    requested_count: int
    actual_count: int
    fallback: bool


def local_xyz(indices, shape_zyx):
    return np.column_stack(np.unravel_index(indices, shape_zyx))[:, ::-1].astype(np.int64)


def candidate_pools(raw_image, valid, probability, intensity_threshold, config=FieldSamplingConfig()):
    """Only image evidence and predictions can admit candidates; no reference input."""
    raw_image, valid, probability = np.asarray(raw_image), np.asarray(valid), np.asarray(probability)
    if (raw_image.ndim != 3 or valid.shape != raw_image.shape or probability.shape != raw_image.shape
            or valid.dtype != np.bool_ or not np.isfinite(raw_image).all() or raw_image.min() < 0
            or probability.dtype.kind != "f" or probability.dtype.itemsize < 4
            or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1
            or not np.isfinite(intensity_threshold) or intensity_threshold < 0):
        raise ValueError("Expected aligned valid image and full-precision probability fields")
    foreground = np.flatnonzero(valid & (raw_image > intensity_threshold))
    guide = np.flatnonzero(valid & (probability >= config.field_threshold))
    return CandidatePools(foreground, guide, np.union1d(foreground, guide), raw_image.shape)


def coverage_indices(pool, shape_zyx, count, seed, cell_size=8, scores=None):
    """Distinct points in rounds over spatially ordered occupied cells."""
    pool = np.asarray(pool, dtype=np.int64)
    if count < 0 or count > len(pool) or len(np.unique(pool)) != len(pool):
        raise ValueError("Invalid unique coverage pool/count")
    if not count:
        return np.empty(0, dtype=np.int64)
    xyz = local_xyz(pool, shape_zyx)
    cells, inverse = np.unique(xyz // cell_size, axis=0, return_inverse=True)
    centers = (cells.astype(float) + .5) * cell_size
    rng = np.random.default_rng(seed)
    cell_ties = rng.random(len(cells))
    cell_order = []
    current = int(np.argmin(cell_ties))
    remaining = np.ones(len(cells), dtype=bool)
    minimum_distance2 = np.full(len(cells), np.inf)
    for _ in range(min(count, len(cells))):
        cell_order.append(current)
        remaining[current] = False
        minimum_distance2 = np.minimum(minimum_distance2, ((centers - centers[current])**2).sum(axis=1))
        if not remaining.any():
            break
        maximum = minimum_distance2[remaining].max()
        tied = np.flatnonzero(remaining & (minimum_distance2 == maximum))
        current = int(tied[np.argmin(cell_ties[tied])])
    cell_rank = np.full(len(cells), len(cells), dtype=np.int64)
    cell_rank[cell_order] = np.arange(len(cell_order))
    point_ties = rng.random(len(pool))
    if scores is None:
        within = np.lexsort((point_ties, inverse))
    else:
        scores = np.asarray(scores)
        if scores.shape != pool.shape or not np.isfinite(scores).all():
            raise ValueError("Guide scores must align with the candidate pool")
        within = np.lexsort((point_ties, -scores, inverse))
    counts = np.bincount(inverse, minlength=len(cells))
    starts = np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    within_rank = np.arange(len(pool)) - starts
    traversal = np.lexsort((cell_rank[inverse[within]], within_rank))
    return pool[within[traversal[:count]]]


def select_cloud(raw_image, valid, probability, origin_xyz, intensity_threshold,
                 arm, count, seed, config=FieldSamplingConfig(), *, pools=None):
    if arm not in ARMS or isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Invalid arm or requested point count")
    pools = pools or candidate_pools(raw_image, valid, probability, intensity_threshold, config)
    # Revalidate supplied pool provenance rather than accepting arbitrary/label-selected indices.
    expected = candidate_pools(raw_image, valid, probability, intensity_threshold, config)
    if pools.shape_zyx != expected.shape_zyx or any(not np.array_equal(getattr(pools, k), getattr(expected, k))
                                                  for k in ("foreground", "guide", "expanded")):
        raise ValueError("Candidate pool does not match image-only admission")
    origin = np.asarray(origin_xyz)
    if origin.shape != (3,) or not np.isfinite(origin).all() or not np.equal(origin, np.round(origin)).all():
        raise ValueError("Expected integer native XYZ crop origin")
    pool = pools.foreground if arm == "threshold_random" else pools.expanded
    n = min(count, len(pool))
    roles = np.zeros(n, dtype=np.uint8)
    fallback = False
    if arm in ("threshold_random", "expanded_random"):
        indices = np.random.default_rng(seed).choice(pool, size=n, replace=False)
    elif arm == "expanded_coverage" or not len(pools.guide):
        indices = coverage_indices(pool, pools.shape_zyx, n, seed, config.cell_size)
        fallback = arm == "guided_context" and not len(pools.guide)
    else:
        guide_count = min(int(np.floor(n * config.guide_fraction)), len(pools.guide))
        guide = coverage_indices(pools.guide, pools.shape_zyx, guide_count, seed, config.cell_size,
                                 scores=probability.ravel()[pools.guide])
        remaining = np.setdiff1d(pool, guide, assume_unique=True)
        context = coverage_indices(remaining, pools.shape_zyx, n - guide_count, seed, config.cell_size)
        indices = np.r_[guide, context]
        roles[:guide_count] = 1
    order = np.argsort(indices)
    indices, roles = indices[order], roles[order]
    if len(np.unique(indices)) != n or not np.isin(indices, pool).all():
        raise AssertionError("Selector lost uniqueness or pool membership")
    xyz = local_xyz(indices, pools.shape_zyx) + origin
    points = np.column_stack((xyz, raw_image.ravel()[indices])).astype(np.float32)
    return SelectedCloud(points, indices, probability.ravel()[indices].astype(np.float32), roles,
                         count, n, fallback)

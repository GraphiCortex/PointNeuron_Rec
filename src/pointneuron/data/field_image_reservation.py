"""Isolated image quota control; existing field selectors are unchanged."""
from __future__ import annotations

import numpy as np

from pointneuron.data.field_guided_sampling import (
    FieldSamplingConfig, SelectedCloud, candidate_pools, local_xyz,
)


def image_reserved_cloud(raw_image, valid, probability, origin_xyz, intensity_threshold,
                         count, seed, config=FieldSamplingConfig(), *, pools=None):
    """Uniform half-budget foreground reservation, then disjoint uniform union fill.

    Return (cloud, reserved_mask). This mask denotes image quota provenance,
    not the guide role used by the existing 50/50 guide selector. The fill may
    also select foreground points, so half is a floor, not an exact final ratio.
    No references, labels, score ranking or spatial ordering enter selection.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, int) or count < 0:
        raise ValueError("Expected a nonnegative integer unique-point budget")
    expected = candidate_pools(raw_image, valid, probability, intensity_threshold, config)
    if pools is not None and (pools.shape_zyx != expected.shape_zyx or any(
            not np.array_equal(getattr(pools, k), getattr(expected, k))
            for k in ("foreground", "guide", "expanded"))):
        raise ValueError("Supplied pools differ from image/field admission")
    pools = expected
    origin = np.asarray(origin_xyz)
    if (origin.shape != (3,) or not np.isfinite(origin).all()
            or not np.equal(origin, np.round(origin)).all()):
        raise ValueError("Expected integer native XYZ crop origin")
    n = min(count, len(pools.expanded))
    reserved_n = min(n // 2, len(pools.foreground))
    rng = np.random.default_rng(seed)
    reserved = rng.choice(pools.foreground, reserved_n, replace=False)
    remaining = np.setdiff1d(pools.expanded, reserved, assume_unique=True)
    fill = rng.choice(remaining, n - reserved_n, replace=False)
    indices = np.r_[reserved, fill]
    reserved_mask = np.r_[np.ones(reserved_n, dtype=bool), np.zeros(n-reserved_n, dtype=bool)]
    order = np.argsort(indices)
    indices, reserved_mask = indices[order], reserved_mask[order]
    if len(np.unique(indices)) != n or not np.isin(indices, pools.expanded).all():
        raise AssertionError("Reservation lost its unique budget or candidate provenance")
    points = np.column_stack((local_xyz(indices, pools.shape_zyx) + origin,
                              np.asarray(raw_image).ravel()[indices])).astype(np.float32)
    cloud = SelectedCloud(points, indices, np.asarray(probability).ravel()[indices].astype(np.float32),
                          np.zeros(n, dtype=np.uint8), count, n, False)
    return cloud, reserved_mask

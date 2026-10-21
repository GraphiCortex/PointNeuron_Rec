"""Reference-free CPU sampling probe, not a trained/product input policy."""
from __future__ import annotations

from itertools import product
from math import ceil

import numpy as np


def normalized_image(values):
    """Whole-image affine normalization; no clipping, mask or histogram fit."""
    values = np.asarray(values)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite scalar ZYX image")
    low, high = float(values.min()), float(values.max())
    normalized = values.astype(np.float32) - low
    if high > low:
        normalized /= high - low
    return normalized, {"minimum": low, "maximum": high, "constant": high == low}


def sato_guide(normalized, sigmas):
    """Pinned published operator; native isotropic voxel calculation only."""
    import skimage
    from skimage.filters import sato

    if skimage.__version__ != "0.25.2":
        raise RuntimeError("This probe requires scikit-image 0.25.2; use its isolated runtime")
    values = np.asarray(normalized)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite ZYX image")
    if not sigmas or any(not np.isfinite(x) or x <= 0 for x in sigmas):
        raise ValueError("Guide scales must be finite and positive")
    guide = sato(values.astype(np.float32), sigmas=tuple(sigmas), black_ridges=False,
                 mode="reflect", cval=0.)
    if not np.isfinite(guide).all() or (guide < 0).any():
        raise ValueError("Operator returned invalid weights")
    return guide


def partition_tiles(shape_zyx, side):
    """Disjoint, image-defined boxes; edge boxes are clipped, never padded."""
    if type(side) is not int or side <= 0 or len(shape_zyx) != 3 or min(shape_zyx) <= 0:
        raise ValueError("Expected positive dimensions and integer tile side")
    for start in product(*(range(0, d, side) for d in shape_zyx)):
        yield tuple(slice(a, min(a + side, d)) for a, d in zip(start, shape_zyx))


def selection_count(voxels, cell_side):
    if type(voxels) is not int or voxels <= 0 or type(cell_side) is not int or cell_side <= 0:
        raise ValueError("Expected positive integer voxel count and cell side")
    return min(voxels, ceil(voxels / cell_side**3))


def probabilities(values, uniform_mass):
    """Positive mixture weights; zero-response tile explicitly falls back to uniform.

    uniform_mass is mass in the draw distribution, not a retained-point quota.
    Without-replacement inclusion probabilities differ from these weights.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative weights")
    if not np.isfinite(uniform_mass) or not 0 < uniform_mass <= 1:
        raise ValueError("uniform_mass must be in (0, 1]")
    maximum = float(values.max())
    if maximum == 0:
        return np.full(len(values), 1 / len(values)), True
    scaled = values / maximum
    probability = (1 - uniform_mass) * scaled / scaled.sum() + uniform_mass / len(values)
    return probability, False


def ranked_selection(probability, count, seed, tile_id):
    """Exponential-race PPS sampling without replacement; paired random numbers.

    A larger count uses a prefix of the same ranking. Unique voxel indices are
    returned; no reference, geometry relocation or duplicate padding is involved.
    """
    probability = np.asarray(probability, dtype=np.float64)
    if (probability.ndim != 1 or not len(probability) or not np.isfinite(probability).all()
            or (probability <= 0).any() or not np.isclose(probability.sum(), 1)):
        raise ValueError("Expected a positive probability vector summing to one")
    if type(count) is not int or not 0 < count <= len(probability):
        raise ValueError("Invalid distinct-point count")
    if any(type(x) is not int or x < 0 for x in (seed, tile_id)):
        raise ValueError("Seed and tile_id must be nonnegative integers")
    rng = np.random.default_rng(np.random.SeedSequence([seed, tile_id]))
    # 1-random() is in (0,1], preventing log(0) in an exponential waiting time.
    waiting = -np.log1p(-rng.random(len(probability))) / probability
    ids = np.argpartition(waiting, count - 1)[:count]
    return ids[np.lexsort((ids, waiting[ids]))]


def source_indices(local_ids, slices_zyx, full_shape_zyx):
    shape = tuple(s.stop - s.start for s in slices_zyx)
    local_zyx = np.array(np.unravel_index(local_ids, shape))
    global_zyx = local_zyx + np.array([s.start for s in slices_zyx])[:, None]
    flat = np.ravel_multi_index(tuple(global_zyx), full_shape_zyx)
    return flat, global_zyx[::-1].T

"""Image-only foreground rules for controlled preprocessing diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_propagation, gaussian_filter
from scipy.stats import median_abs_deviation


def fractional_intensity_threshold(datatype, fraction, *, mode='dtype_fraction', data=None):
    """Image-only threshold scaling; uint16 storage does not imply 16-bit signal.

    image_max_fraction is a minimal automatic range correction for the baseline,
    not a neurite segmentation model. Its maximum can be affected by outliers.
    """
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Threshold fraction must be finite and in [0, 1]')
    if datatype not in (1, 2):
        raise NotImplementedError(f'Unsupported Vaa3D datatype {datatype}')
    if mode == 'dtype_fraction':
        scale = 255 if datatype == 1 else 65535
    elif mode == 'image_max_fraction':
        values = np.asarray(data)
        if values.size == 0 or values.dtype.kind not in 'uif' or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Image threshold scaling requires finite nonnegative decoded data')
        scale = float(values.max())
    else:
        raise ValueError(f'Unknown threshold mode {mode}')
    return int(round(scale * fraction))


def difference_noise_scale(image_zyx):
    """Patch-level robust noise proxy; assumes sparse signal and weakly correlated noise.

    Neighbor differences remove constant offsets. For independent Gaussian noise,
    their standard deviation is sqrt(2) times the voxel-noise standard deviation.
    Structure, gradients, clipping and correlated noise can bias this estimate.
    """
    image = np.asarray(image_zyx)
    if image.ndim != 3 or not image.size or not np.isfinite(image).all():
        raise ValueError('Expected a finite nonempty ZYX image')
    values = image.astype(np.float64)
    estimates = [float(median_abs_deviation(np.diff(values, axis=axis).ravel(), scale='normal') / np.sqrt(2))
                 if image.shape[axis] > 1 else None for axis in range(3)]
    available = [x for x in estimates if x is not None]
    estimate = float(np.median(available)) if available else 0.
    # Quantized sparse images may have zero MAD. Never normalize by zero.
    floor = (1. / np.sqrt(12) if np.issubdtype(image.dtype, np.integer) else
             float(np.finfo(np.float32).eps * max(float(np.max(np.abs(values))), float(np.ptp(values)),
                                                  float(np.finfo(np.float32).tiny))))
    return max(estimate, floor), {'difference_sigma_zyx': estimates,
                                  'noise_sigma_before_floor': estimate, 'noise_floor': floor,
                                  'noise_estimate_above_floor': bool(estimate > floor),
                                  'noise_estimator': 'median of axis-wise normal-scaled neighbor-difference MAD / sqrt(2)',
                                  'noise_scope': 'core patch; scalar, not a spatially varying noise map'}


def noise_hysteresis_mask(score_zyx, raw_image_zyx, noise_high=4., noise_low=2.):
    """Apply noise thresholds to an already background-corrected core patch.

    This entry point also permits a pilot to reuse the exact saved halo-filtered
    score instead of approximating it with a crop-only background estimate.
    """
    score = np.asarray(score_zyx)
    if (score.shape != np.asarray(raw_image_zyx).shape or score.ndim != 3 or not score.size
            or not np.isfinite(score).all() or np.any(score < 0)):
        raise ValueError('Expected matching finite nonnegative ZYX score and raw image')
    if not np.isfinite(noise_high) or not np.isfinite(noise_low) or not 0 < noise_low <= noise_high:
        raise ValueError('Require finite 0 < noise_low <= noise_high')
    sigma, metadata = difference_noise_scale(raw_image_zyx)
    threshold, weak_threshold = float(noise_high * sigma), float(noise_low * sigma)
    seeds, allowed = score > threshold, score > weak_threshold
    mask = binary_propagation(seeds, mask=allowed, structure=np.ones((3, 3, 3), dtype=bool))
    metadata.update(noise_sigma=sigma, noise_high=float(noise_high), noise_low=float(noise_low),
                    threshold=threshold, weak_threshold=weak_threshold,
                    strong_seed_voxels=int(seeds.sum()), allowed_weak_voxels=int(allowed.sum()),
                    connectivity=26, propagation_scope='core patch only; no gap filling')
    return mask, metadata


def guarded_noise_hysteresis_mask(score_zyx, raw_image_zyx, noise_high=4., noise_low=2.):
    """Bound noise growth by background Otsu; fall back if MAD hits its floor.

    This conservative diagnostic cannot recover signal below the Otsu threshold.
    It prevents a collapsed noise estimate from admitting almost all texture.
    The historical noise_hysteresis method remains available for reproduction.
    """
    _, metadata = noise_hysteresis_mask(score_zyx, raw_image_zyx, noise_high, noise_low)
    score = np.asarray(score_zyx)
    otsu = otsu_threshold(score)
    above_floor = metadata['noise_estimate_above_floor']
    high = max(otsu, metadata['threshold']) if above_floor else otsu
    low = max(otsu, metadata['weak_threshold']) if above_floor else otsu
    seeds, allowed = score > high, score > low
    mask = binary_propagation(seeds, mask=allowed, structure=np.ones((3, 3, 3), dtype=bool))
    metadata.update(threshold=high, weak_threshold=low, background_otsu_threshold=otsu,
                    fallback='none' if above_floor else 'background_otsu; noise MAD at floor',
                    strong_seed_voxels=int(seeds.sum()), allowed_weak_voxels=int(allowed.sum()),
                    growth_bound='subset of background Otsu foreground')
    return mask, metadata


def otsu_threshold(image, bins=256):
    """Maximize between-class variance; constant images have no separable classes."""
    values = np.asarray(image, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or bins < 2:
        raise ValueError('Expected a nonempty finite image and at least two bins')
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return hi
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    weights = np.cumsum(hist).astype(float)
    moments = np.cumsum(hist * centers)
    denom = weights[:-1] * (weights[-1] - weights[:-1])
    score = np.full(bins - 1, -np.inf)
    valid = denom > 0
    score[valid] = ((moments[-1] * weights[:-1][valid] - moments[:-1][valid] * weights[-1]) ** 2
                    / denom[valid])
    # Bin boundary makes thresholding agree with the two histogram classes.
    return float(edges[int(np.argmax(score)) + 1])


def foreground_mask(image_zyx, method, fixed_threshold=51, background_sigma=8., spacing_xyz=(1., 1., 1.), core_slices=None,
                    noise_high=4., noise_low=2.):
    """Return mask, score image, metadata; never accepts labels or reference traces.

    Background correction subtracts a Gaussian mean and clips negative residuals.
    It is an experimental contrast rule, not a validated neuron segmentation.
    """
    image = np.asarray(image_zyx, dtype=np.float32)
    spacing = np.asarray(spacing_xyz, dtype=float)
    if image.ndim != 3 or not image.size or not np.isfinite(image).all():
        raise ValueError('Expected a finite, nonempty ZYX image')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Expected positive finite XYZ spacing')
    if not np.isfinite(background_sigma) or background_sigma <= 0 or not np.isfinite(fixed_threshold):
        raise ValueError('Invalid foreground parameters')
    core = core_slices if core_slices is not None else (slice(None),) * 3
    metadata = {'method': method, 'constant_score': False}
    if method == 'fixed':
        score, threshold = image[core], float(fixed_threshold)
    elif method == 'raw_otsu':
        score = image[core]
        threshold = otsu_threshold(score)
    elif method in ('background_otsu', 'noise_hysteresis', 'guarded_noise_hysteresis'):
        sigma_zyx = (background_sigma / spacing)[::-1]
        background = gaussian_filter(image, sigma=sigma_zyx, mode='reflect')
        score = np.maximum(image - background, 0.)[core]
        metadata.update(background_sigma=background_sigma, sigma_zyx=sigma_zyx.tolist(),
                        boundary_mode='reflect', negative_residuals='clipped to zero')
        if method == 'background_otsu':
            threshold = otsu_threshold(score)
        else:
            rule = guarded_noise_hysteresis_mask if method == 'guarded_noise_hysteresis' else noise_hysteresis_mask
            mask, noise_metadata = rule(score, np.asarray(image_zyx)[core], noise_high, noise_low)
            threshold = noise_metadata['threshold']
            metadata.update(noise_metadata)
    else:
        raise ValueError(f'Unknown mask method: {method}')
    metadata.update(threshold=threshold, constant_score=bool(score.min() == score.max()),
                    score_range=[float(score.min()), float(score.max())])
    return (mask if method in ('noise_hysteresis', 'guarded_noise_hysteresis') else score > threshold), score, metadata

"""Local image-evidence measurements, not foreground labels or model losses.

Reference-derived centers/tangents define evaluation windows only. Sampling
and inference never call this module. Width is an apparent radial-profile
half-height diameter in native voxel units, not anatomical caliber.
"""
from __future__ import annotations

import numpy as np


def arc_windows(vertices, step=4.):
    """Uniform arc-length midpoint samples with tangents and length weights."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or step <= 0:
        raise ValueError("Invalid polyline/step")
    delta = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 0
    starts, delta, lengths = vertices[:-1][valid], delta[valid], lengths[valid]
    if not len(lengths):
        return []
    total = lengths.sum()
    count = max(1, int(np.ceil(total/step)))
    offsets = (np.arange(count)+.5)*total/count
    cumulative = np.r_[0., np.cumsum(lengths)]
    ids = np.minimum(np.searchsorted(cumulative, offsets, side="right")-1, len(lengths)-1)
    fractions = (offsets-cumulative[ids])/lengths[ids]
    centers = starts[ids]+fractions[:, None]*delta[ids]
    return [{"xyz": p, "tangent": d/l, "weight": float(total/count), "arc": float(s)}
            for p, d, l, s in zip(centers, delta[ids], lengths[ids], offsets)]


def profile_window(image, center, tangent, start_xyz, stop_xyz, radii=(1., 2., 4., 8.),
                   half_length=2., shell=(8., 12.), bin_width=.5):
    """Prepare unique source-voxel windows and image-only profile metadata.

    Background/width use the available source image. Retention denominators
    include only voxels in the declared input box. Clipping is measured, not
    padded. Overlapping windows must be aggregated as local length-weighted
    fractions, not summed as a unique whole-neuron evidence mass.
    """
    image = np.asarray(image)
    center, tangent = np.asarray(center, float), np.asarray(tangent, float)
    start, stop = np.asarray(start_xyz, int), np.asarray(stop_xyz, int)
    radii = np.asarray(radii, float)
    if (image.ndim != 3 or center.shape != (3,) or tangent.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(tangent).all()
            or np.linalg.norm(tangent) == 0 or np.any(stop <= start)
            or np.any(start < 0) or np.any(stop > np.asarray(image.shape[::-1]))
            or half_length <= 0 or bin_width <= 0 or not 0 < shell[0] < shell[1]
            or np.any(radii <= 0) or np.any(radii > shell[0])):
        raise ValueError("Invalid source, frame or measurement scales")
    tangent = tangent/np.linalg.norm(tangent)
    extent = shell[1]+half_length
    lower = np.floor(center-extent).astype(int)
    upper = np.ceil(center+extent).astype(int)+1
    clipped_low = np.maximum(lower, 0)
    clipped_high = np.minimum(upper, image.shape[::-1])
    if np.any(clipped_high <= clipped_low):
        raise ValueError("Window outside image")
    slices = tuple(slice(int(a), int(b)) for a, b in zip(clipped_low[::-1], clipped_high[::-1]))
    values = image[slices].astype(float).ravel()
    coords = np.indices(image[slices].shape).reshape(3, -1)[::-1].T+clipped_low
    offset = coords-center
    axial = offset@tangent
    radial = np.sqrt(np.maximum(0., np.einsum("ij,ij->i", offset, offset)-axial**2))
    slab = np.abs(axial) <= half_length
    shell_mask = slab & (radial >= shell[0]) & (radial < shell[1])
    background = float(np.median(values[shell_mask])) if shell_mask.any() else None
    mad = float(np.median(np.abs(values[shell_mask]-background))) if shell_mask.any() else None
    in_box = ((coords >= start) & (coords < stop)).all(axis=1)
    bin_centers = np.arange(bin_width/2, shell[0], bin_width)
    profile = []
    for r in bin_centers:
        mask = slab & (np.abs(radial-r) <= bin_width/2) & (radial < r+bin_width/2)
        profile.append(float(values[mask].mean()) if mask.any() else None)
    width, contrast, status = None, None, "no_background"
    if background is not None:
        valid = np.array([v is not None for v in profile])
        inner = valid & (bin_centers < 2.)
        if inner.any():
            peak = max(float(profile[i])-background for i in np.flatnonzero(inner))
            contrast = float(peak)
            status = "nonpositive_inner_contrast"
            if peak > 0:
                p = np.array([np.nan if v is None else v-background for v in profile])
                p = np.interp(bin_centers, bin_centers[valid], p[valid])
                peak_id = int(np.argmax(np.where(bin_centers < 2., p, -np.inf)))
                crossing = np.flatnonzero((np.arange(len(p)) > peak_id) & (p <= peak/2))
                status = "no_half_height_crossing"
                if len(crossing):
                    j = int(crossing[0]); fraction = (p[j-1]-peak/2)/(p[j-1]-p[j])
                    width = float(2*(bin_centers[j-1]+fraction*(bin_centers[j]-bin_centers[j-1])))
                    status = "resolved_apparent_profile"
                if np.any(p[bin_centers >= 2.] > peak):
                    width, status = None, "outer_profile_exceeds_inner_peak"
        else:
            status = "no_inner_profile_samples"
    domains = []
    shape = tuple((stop-start)[::-1])
    for r in radii:
        ideal_in_image = slab & (radial <= r)
        mask = ideal_in_image & in_box
        local = (coords[mask]-start).astype(int)
        ids = np.ravel_multi_index(tuple(local[:, ::-1].T), shape)
        mass = np.maximum(values[mask]-background, 0.) if background is not None else None
        domains.append({"radius": float(r), "ids": ids, "mass": mass,
                        "source_voxel_count": int(mask.sum()),
                        "input_box_fraction_of_source_window": float(mask.sum()/ideal_in_image.sum())
                        if ideal_in_image.any() else None})
    return {"metadata": {"xyz": center.tolist(), "tangent": tangent.tolist(), "background": background,
                         "background_MAD": mad, "background_shell_voxels": int(shell_mask.sum()),
                         "source_bounding_cube_clipped": bool(np.any(clipped_low != lower) or np.any(clipped_high != upper)),
                         "apparent_width_voxel": width, "inner_peak_contrast": contrast,
                         "profile_status": status, "profile_radii": bin_centers.tolist(),
                         "profile_mean_raw_intensity": profile}, "domains": domains}


def measure(window, selected_mask):
    mask = np.asarray(selected_mask, dtype=bool).ravel()
    rows = []
    for domain in window["domains"]:
        chosen = mask[domain["ids"]]
        total = float(domain["mass"].sum()) if domain["mass"] is not None else None
        retained = float(domain["mass"][chosen].sum()) if domain["mass"] is not None else None
        rows.append({"radius": domain["radius"], "source_voxel_count": domain["source_voxel_count"],
                     "selected_voxel_count": int(chosen.sum()), "available_positive_contrast_mass": total,
                     "retained_positive_contrast_mass": retained,
                     "retained_mass_fraction": retained/total if total is not None and total > 0 else None,
                     "input_box_fraction_of_source_window": domain["input_box_fraction_of_source_window"]})
    return rows


def source_priority_order(flat_ids, seed=0):
    """Fixed source-ID hash ordering, independent of sampler rank and membership."""
    ids = np.asarray(flat_ids, dtype=np.uint64)
    x = ids+np.uint64(seed)+np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27)))*np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return np.lexsort((ids, x))

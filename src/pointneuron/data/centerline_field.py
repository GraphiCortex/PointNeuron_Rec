"""Native voxel crops and continuous-segment weak supervision (XYZ -> ZYX)."""
from __future__ import annotations

import numpy as np
from pointneuron.data.curve_targets import curve_geometry

SCHEMA = "pointneuron2-centerline-field-v1"


def crop_volume(volume_zyx, center_xyz, radius):
    """Integer voxel centers inside the original inclusive patch box; pad as unknown."""
    volume_zyx = np.asarray(volume_zyx)
    center = np.asarray(center_xyz, dtype=float)
    if volume_zyx.ndim != 3 or center.shape != (3,) or not np.isfinite(center).all() or radius <= 0:
        raise ValueError("Invalid native crop")
    origin = np.ceil(center - radius).astype(int)
    end = np.floor(center + radius).astype(int) + 1
    shape_xyz = end - origin
    if (shape_xyz <= 0).any():
        raise ValueError("Empty crop")
    crop = np.zeros(tuple(shape_xyz[::-1]), dtype=volume_zyx.dtype)
    valid = np.zeros(crop.shape, dtype=bool)
    lower = np.maximum(origin, 0)
    upper = np.minimum(end, np.array(volume_zyx.shape[::-1]))
    if np.all(upper > lower):
        src = tuple(slice(a, b) for a, b in zip(lower[::-1], upper[::-1]))
        dst = tuple(slice(a, b) for a, b in zip((lower-origin)[::-1], (upper-origin)[::-1]))
        crop[dst] = volume_zyx[src]
        valid[dst] = True
    return crop, valid, origin


def segment_distance_field(tree, origin_xyz, shape_zyx, maximum_distance):
    """Exact voxel-center distance to full segments, truncated outside a margin.

    Endpoints need not be inside the crop. Collinear subdivision does not change
    the field. Roots, including isolated roots and zero-length edges, contribute.
    Distances are in acquisition voxels, not physical micrometers.
    """
    nodes, _, _, _ = curve_geometry(tree)  # Validate full forest before rasterizing.
    origin = np.asarray(origin_xyz, dtype=float)
    shape = np.asarray(shape_zyx, dtype=int)
    if origin.shape != (3,) or not np.isfinite(origin).all() or shape.shape != (3,) or (shape <= 0).any():
        raise ValueError("Invalid field grid")
    if not np.isfinite(maximum_distance) or maximum_distance <= 0:
        raise ValueError("Invalid distance margin")
    distance = np.full(tuple(shape), np.inf, dtype=np.float32)
    for node in nodes.values():
        b = np.array([node.x, node.y, node.z]) - origin
        parent = nodes.get(node.parent_id, node)
        a = np.array([parent.x, parent.y, parent.z]) - origin
        lo = np.maximum(np.ceil(np.minimum(a, b) - maximum_distance).astype(int), 0)
        hi = np.minimum(np.floor(np.maximum(a, b) + maximum_distance).astype(int) + 1, shape[::-1])
        if (hi <= lo).any():
            continue
        zz, yy, xx = np.meshgrid(*(np.arange(l, h) for l, h in zip(lo[::-1], hi[::-1])), indexing="ij")
        xyz = np.stack((xx, yy, zz), axis=-1)
        delta = b - a
        length2 = float(delta @ delta)
        t = np.zeros(xx.shape) if length2 == 0 else np.clip(((xyz - a) @ delta) / length2, 0, 1)
        d = np.linalg.norm(xyz - (a + t[..., None] * delta), axis=-1)
        d[d > maximum_distance] = np.inf
        slices = tuple(slice(l, h) for l, h in zip(lo[::-1], hi[::-1]))
        np.minimum(distance[slices], d, out=distance[slices])
    return distance


def weak_labels(raw_image, valid, distance, intensity_threshold, tube_radius=2., uncertainty_distance=6.):
    """Trace tube positive; distant bright image evidence unknown, never negative.

    Dark distant voxels and the near-trace annulus provide weak negative labels.
    This does not establish annotation completeness or biological foreground.
    A crop without a local trace supplies no supervision at all.
    """
    if not (0 < tube_radius < uncertainty_distance) or not np.isfinite(intensity_threshold):
        raise ValueError("Invalid weak-label settings")
    if raw_image.shape != valid.shape or distance.shape != valid.shape:
        raise ValueError("Mismatched voxel arrays")
    target = (distance <= tube_radius) & valid
    unknown_bright = valid & (raw_image > intensity_threshold) & (distance > uncertainty_distance)
    known = valid & ~unknown_bright
    if not np.any(valid & (distance <= 0.9)):
        known[:] = False
    return target, known, unknown_bright

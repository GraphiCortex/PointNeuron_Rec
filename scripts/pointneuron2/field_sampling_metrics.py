"""Reference-only scoring after image-derived input clouds have been frozen."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pointneuron.data.curve_targets import curve_geometry, clip_parameters
from scripts.pointneuron2.audit_field_centerpaths import nearest_segments, distance_summary
from scripts.pointneuron2.reference_gaps import segment_probe

TOLERANCES = (1, 2, 4, 6)


def reference_geometry(tree, origin_xyz, shape_zyx):
    nodes, children, critical, chains = curve_geometry(tree)
    origin = np.asarray(origin_xyz, dtype=float)
    lower, upper = origin, origin + np.asarray(shape_zyx[::-1]) - 1
    full = []
    for node in nodes.values():
        end = np.array([node.x, node.y, node.z])
        if node.parent_id != -1:
            parent = nodes[node.parent_id]
            full.append([np.array([parent.x, parent.y, parent.z]) - origin, end - origin])
        elif not children[node.node_id]:
            full.append([end - origin, end - origin])
    parts = []
    for ids, xyz, _, arc in chains:
        pieces, current = [], []
        for a, b in zip(xyz[:-1], xyz[1:]):
            interval = clip_parameters(a, b, lower, upper)
            if interval is None:
                if current:
                    pieces.append(current)
                    current = []
                continue
            clipped = np.array([a + interval[0]*(b-a), a + interval[1]*(b-a)]) - origin
            if current and not np.allclose(current[-1][1], clipped[0], atol=1e-8, rtol=0):
                pieces.append(current)
                current = []
            current.append(clipped)
        if current:
            pieces.append(current)
        for piece_id, segments in enumerate(pieces):
            probe = segment_probe(segments, step=1.)
            length = float(probe["weights"].sum())
            parts.append({"chain_start_id": ids[0], "chain_end_id": ids[-1], "piece": piece_id,
                          "boundary_clipped": bool(length < arc[-1] - 1e-8),
                          "shorter_than_6_voxels": length < 6., "length_voxels": length,
                          "probe": probe})
    critical_points = np.array([[nodes[i].x, nodes[i].y, nodes[i].z] for i in sorted(critical)]).reshape(-1, 3)
    inside = np.all((critical_points >= lower) & (critical_points <= upper), axis=1)
    return {"full_segments": np.asarray(full).reshape(-1, 2, 3), "chains": parts,
            "critical_points": critical_points[inside] - origin}


def unsupported_run(supported, weights):
    worst, run = 0., 0.
    for present, weight in zip(supported, weights):
        run = 0. if present else run + weight
        worst = max(worst, run)
    return float(worst)


def measure_support(xyz, geometry):
    """No edges are invented between points; distances are nearest-point distances."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite selected point geometry")
    tree = cKDTree(xyz) if len(xyz) else None
    chains, all_distances, all_weights = [], [], []
    for part in geometry["chains"]:
        probe = part["probe"]
        distances = tree.query(probe["points"], workers=1)[0] if tree is not None else np.full(len(probe["points"]), np.inf)
        all_distances.append(distances)
        all_weights.append(probe["weights"])
        row = {k: v for k, v in part.items() if k != "probe"}
        summary = distance_summary(distances, probe["weights"])
        row["reference_to_points"] = summary
        row["longest_unsupported_run_voxels"] = {
            str(t): unsupported_run(distances <= t, probe["weights"]) for t in TOLERANCES}
        chains.append(row)
    distances = np.concatenate(all_distances) if all_distances else np.empty(0)
    weights = np.concatenate(all_weights) if all_weights else np.empty(0)
    summary = distance_summary(distances, weights)
    annotated = [c for c in chains if c["reference_to_points"] is not None]
    critical = geometry["critical_points"]
    critical_distances = tree.query(critical, workers=1)[0] if tree is not None else np.full(len(critical), np.inf)
    return {"reference_length_voxels": float(weights.sum()), "reference_to_points": summary,
            "reference_coverage_2": None if summary is None else summary["within_voxels"]["2"],
            "worst_chain_coverage": {str(t): min((c["reference_to_points"]["within_voxels"][str(t)] for c in annotated), default=None)
                                     for t in TOLERANCES},
            "longest_unsupported_run_voxels": {str(t): max((c["longest_unsupported_run_voxels"][str(t)] for c in annotated), default=None)
                                               for t in TOLERANCES},
            "critical_reference_points": len(critical),
            "critical_reference_coverage": {str(t): float((critical_distances <= t).mean()) if len(critical) else None
                                            for t in TOLERANCES}, "chains": chains}


def point_agreement(distances):
    return distance_summary(np.asarray(distances), np.ones(len(distances)))


def reference_distances(xyz, full_segments):
    """Exact segment distances, with conservative endpoint-bound pruning."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    segments = np.asarray(full_segments, dtype=float).reshape(-1, 2, 3)
    if not len(segments):
        return np.full(len(xyz), np.inf)
    endpoints = cKDTree(segments.reshape(-1, 3))
    minimum, maximum = segments.min(axis=1), segments.max(axis=1)
    result = np.empty(len(xyz))
    for start in range(0, len(xyz), 128):
        points = xyz[start:start+128]
        # Distance to any endpoint is an upper bound on distance to its segment.
        radius = float(endpoints.query(points, workers=1)[0].max()) + 1e-8
        nearby = np.all((maximum >= points.min(axis=0)-radius) & (minimum <= points.max(axis=0)+radius), axis=1)
        result[start:start+len(points)] = nearest_segments(points, segments[nearby])
    return result


def cloud_metrics(cloud, raw_image, origin_xyz, threshold, geometry, union_indices, union_reference_distances):
    indices = cloud.voxel_indices
    positions = np.searchsorted(union_indices, indices)
    if len(indices) and (np.any(positions >= len(union_indices)) or not np.array_equal(union_indices[positions], indices)):
        raise ValueError("Cloud indices escaped measured union")
    distances = union_reference_distances[positions]
    result = measure_support(cloud.points[:, :3].astype(float) - origin_xyz, geometry)
    agreement = point_agreement(distances)
    intensities = raw_image.ravel()[indices]
    result.update(actual_unique_count=len(indices), duplicate_count=len(indices)-len(np.unique(indices)),
                  occupied_cell_count=len(np.unique(np.floor((cloud.points[:, :3]-origin_xyz)/8).astype(np.int64), axis=0)),
                  points_to_reference=agreement,
                  reference_proximity_6=None if agreement is None else agreement["within_voxels"]["6"],
                  below_original_threshold_points=int((intensities <= threshold).sum()),
                  zero_intensity_points=int((intensities == 0).sum()),
                  intensity_quantiles=None if not len(indices) else np.quantile(intensities, [0, .25, .5, .75, 1]).tolist(),
                  field_score_quantiles=None if not len(indices) else np.quantile(cloud.field_scores, [0, .25, .5, .75, 1]).tolist(),
                  field_fallback=cloud.fallback,
                  roles={name: {"count": int(mask.sum()), "points_to_reference": point_agreement(distances[mask])}
                         for name, mask in (("guide", cloud.selection_role == 1), ("context", cloud.selection_role == 0))})
    return result

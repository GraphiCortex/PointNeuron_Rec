"""One-to-one reference-agreement metrics for proposals, not reconstructed edges."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from .cloud_support import longest_run


def one_to_one(reference, predicted, tolerance):
    reference, predicted = np.asarray(reference, dtype=float), np.asarray(predicted, dtype=float)
    if (reference.ndim != 2 or predicted.ndim != 2 or reference.shape[-1] != 3
            or predicted.shape[-1] != 3 or not np.isfinite(reference).all()
            or not np.isfinite(predicted).all() or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Expected finite XYZ sets and nonnegative tolerance")
    matched_targets = np.zeros(len(reference), dtype=bool)
    if len(reference) and len(predicted):
        neighbors = cKDTree(reference).query_ball_tree(cKDTree(predicted), tolerance)
        rows = np.repeat(np.arange(len(reference)), [len(row) for row in neighbors])
        cols = np.asarray([p for row in neighbors for p in row], dtype=int)
        graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(reference), len(predicted)))
        matched_targets = maximum_bipartite_matching(graph, perm_type="column") >= 0
    count = int(matched_targets.sum())
    return {"matched": count, "predicted": len(predicted), "reference": len(reference),
            "precision": count/len(predicted) if len(predicted) else 0.,
            "recall": count/len(reference) if len(reference) else None,
            "f1": 2*count/(len(predicted)+len(reference)) if len(reference) else None,
            "extras": len(predicted)-count, "misses": len(reference)-count}, matched_targets


def proposal_metrics(centers, scores, target_xyz, target_roles, curve_samples, *, cutoff=.5):
    centers, scores = np.asarray(centers), np.asarray(scores)
    if (centers.ndim != 2 or centers.shape[1] != 3 or scores.shape != (len(centers),)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
            or not 0 <= cutoff <= 1 or len(target_roles) != len(target_xyz)):
        raise ValueError("Invalid proposal packet, confidence or target roles")
    accepted = centers[scores >= cutoff]
    result = {"cutoff": cutoff, "accepted_count": len(accepted), "target_count": len(target_xyz),
              "annotation_status": "reference_agreement_not_certified_foreground", "tolerances": {}}
    for tolerance in (2., 6.):
        nodes, target_mask = one_to_one(target_xyz, accepted, tolerance)
        all_nodes, _ = one_to_one(target_xyz, centers, tolerance)
        support = {}
        for role in ("root", "leaf", "junction", "crop_boundary"):
            mask = np.asarray([role in r for r in target_roles], dtype=bool)
            support[role] = {"reference_count": int(mask.sum()), "matched_in_selected_geometric_matching": int((mask & target_mask).sum())}
        total = sum(float(w.sum()) for _, w in curve_samples)
        distances = [cKDTree(accepted).query(points)[0] if len(accepted) else np.full(len(points), np.inf)
                     for points, _ in curve_samples]
        covered = sum(float(w[d <= tolerance].sum()) for d, (_, w) in zip(distances, curve_samples))
        result["tolerances"][str(tolerance)] = {"accepted_nodes": nodes,
            "all_query_node_recall": all_nodes["recall"], "structural_node_support": support,
            "curve_to_accepted_node_support": covered/total if total else None,
            "longest_unsupported_run": max((longest_run(d > tolerance, w) for d, (_, w) in zip(distances, curve_samples)), default=None)}
    result["fixed_cutoff_diagnostics"] = {
        str(t): one_to_one(target_xyz, centers[scores >= t], 2.)[0] for t in (.1, .25, .5, .75, .9)}
    result["structural_support_limitation"] = "One geometric matching, potentially ambiguous near close nodes; not anatomical correspondence or connectivity"
    main = result["tolerances"]["2.0"]["accepted_nodes"]
    result["learnability_gate"] = bool(len(target_xyz) and main["precision"] >= .95 and main["recall"] >= .95)
    return result

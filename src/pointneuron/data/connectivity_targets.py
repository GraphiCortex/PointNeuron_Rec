"""Training-only connectivity labels; never imported by the inference scorer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from pointneuron.data.curve_targets import curve_geometry


def full_reference_components(tree, target_identities):
    """Use full SWC roots, not the disconnected components caused by cropping."""
    nodes, children, _, _ = curve_geometry(tree)
    roots = {}
    for root, node in nodes.items():
        if node.parent_id != -1:
            continue
        pending = [root]
        while pending:
            current = pending.pop()
            roots[current] = root
            pending.extend(children[current])
    return np.array([roots[key[1]] for key in target_identities], dtype=np.int64)


@dataclass(frozen=True)
class EdgeSupervision:
    labels: torch.Tensor  # 1 known adjacency, 0 known nonadjacency, -1 unknown.
    reasons: tuple[str, ...]
    statistics: dict


def assign_edge_labels(packet, candidates, target_xyz, target_edges, target_components,
                       assignment, *, maximum_match_error=2.):
    """Transfer an existing assignment, without changing candidates or features."""
    packet.validate()
    candidates.validate(packet)
    if not np.isfinite(maximum_match_error) or maximum_match_error <= 0:
        raise ValueError("Invalid trusted-placement tolerance")
    xyz = torch.as_tensor(target_xyz).detach().cpu().double().numpy()
    components = torch.as_tensor(target_components).detach().cpu().numpy()
    raw_edges = torch.as_tensor(target_edges).detach().cpu().numpy()
    if raw_edges.size and not np.issubdtype(raw_edges.dtype, np.integer):
        raise ValueError("Target adjacency indices must be integers")
    edges = raw_edges.astype(np.int64).reshape(-1, 2)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or components.shape != (len(xyz),):
        raise ValueError("Invalid target geometry/components")
    if not np.issubdtype(components.dtype, np.integer):
        raise ValueError("Reference component IDs must be integers")
    required = set()
    for u, v in edges:
        if not (0 <= u < len(xyz) and 0 <= v < len(xyz)) or u == v:
            raise ValueError("Invalid target edge")
        if components[u] >= 0 and components[v] >= 0 and components[u] != components[v]:
            raise ValueError("Known reference edge crosses full-reference components")
        key = tuple(sorted((int(u), int(v))))
        if key in required:
            raise ValueError("Duplicate target adjacency")
        required.add(key)
    assignment_arrays = [torch.as_tensor(a).detach().cpu().numpy() for a in assignment]
    if any(a.size and not np.issubdtype(a.dtype, np.integer) for a in assignment_arrays):
        raise ValueError("Assignment indices must be integers")
    pred_indices, target_indices = (a.astype(np.int64) for a in assignment_arrays)
    if pred_indices.ndim != 1 or target_indices.shape != pred_indices.shape:
        raise ValueError("Invalid assignment arrays")
    if len(set(pred_indices)) != len(pred_indices) or len(set(target_indices)) != len(target_indices):
        raise ValueError("Assignment must be one-to-one")
    if (pred_indices < 0).any() or (pred_indices >= len(packet.positions)).any():
        raise ValueError("Assigned proposal out of range")
    if (target_indices < 0).any() or (target_indices >= len(xyz)).any():
        raise ValueError("Assigned target out of range")
    predicted = packet.positions.detach().cpu().double().numpy()
    distances = np.linalg.norm(predicted[pred_indices]-xyz[target_indices], axis=1)
    trusted_pairs = [(int(p), int(t)) for p, t, d in zip(pred_indices, target_indices, distances)
                     if d <= maximum_match_error]
    pred_to_target = dict(trusted_pairs)
    target_to_pred = {t: p for p, t in trusted_pairs}
    array_edges = candidates.edges.detach().cpu().numpy()
    candidate_set = {tuple(sorted(map(int, edge))) for edge in array_edges}
    labels, reasons = [], []
    for u, v in array_edges:
        a, b = pred_to_target.get(int(u)), pred_to_target.get(int(v))
        if a is None or b is None:
            labels.append(-1.)
            reasons.append("unmatched_or_untrusted_endpoint")
        elif tuple(sorted((a, b))) in required:
            labels.append(1.)
            reasons.append("known_reference_adjacency")
        elif components[a] >= 0 and components[a] == components[b]:
            labels.append(0.)
            reasons.append("known_same_tree_nonadjacency")
        else:
            labels.append(-1.)
            reasons.append("different_or_unknown_reference_tree")
    trusted_edges = [(target_to_pred[u], target_to_pred[v]) for u, v in required
                     if u in target_to_pred and v in target_to_pred]
    covered_edges = [edge for edge in trusted_edges if tuple(sorted(edge)) in candidate_set]
    counts = {"positive": labels.count(1.), "negative": labels.count(0.), "unknown": labels.count(-1.),
              "targets": len(xyz), "assigned": len(pred_indices), "trusted_matches": len(trusted_pairs),
              "unmatched_targets": len(xyz)-len(target_indices),
              "poorly_placed_matches": int(np.sum(distances > maximum_match_error)),
              "required_target_edges": len(required), "edges_with_trusted_endpoints": len(trusted_edges),
              "missing_candidate_positive_edges": len(trusted_edges)-len(covered_edges),
              "unrepresented_target_edges": len(required)-len(trusted_edges),
              "attainable_recall_all_target_edges": None if not required else len(covered_edges)/len(required),
              "trusted_match_tolerance": maximum_match_error}
    tensor = torch.tensor(labels, dtype=packet.positions.dtype, device=packet.positions.device)
    return EdgeSupervision(tensor, tuple(reasons), counts)


@dataclass(frozen=True)
class EdgeLoss:
    total: torch.Tensor
    positive_count: int
    negative_count: int
    unknown_count: int
    supervised_crops: int


def masked_edge_loss(logits, labels):
    if logits.ndim != 1 or labels.shape != logits.shape or logits.device != labels.device:
        raise ValueError("Expected same-device edge logits and labels [E]")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("Nonfinite edge loss inputs")
    if not bool(((labels == -1) | (labels == 0) | (labels == 1)).all()):
        raise ValueError("Labels must be -1, 0 or 1")
    positive, negative = labels == 1, labels == 0
    terms = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive]).mean())
    if bool(negative.any()):
        terms.append(F.softplus(logits[negative]).mean())
    total = torch.stack(terms).mean() if terms else logits.sum()*0
    return EdgeLoss(total, int(positive.sum()), int(negative.sum()), int((labels == -1).sum()), int(bool(terms)))


def mean_crop_edge_loss(losses):
    """An all-unknown result requires callers to SKIP the optimizer entirely."""
    if not losses:
        raise ValueError("Expected at least one crop loss")
    supervised = [loss.total for loss in losses if loss.supervised_crops]
    total = torch.stack(supervised).mean() if supervised else sum(loss.total for loss in losses)*0
    return EdgeLoss(total, sum(x.positive_count for x in losses), sum(x.negative_count for x in losses),
                    sum(x.unknown_count for x in losses), len(supervised))

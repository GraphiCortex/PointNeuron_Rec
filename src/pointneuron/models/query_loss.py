"""Per-crop set supervision with explicit reference-empty abstention."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from pointneuron.models.proposal_loss import conservative_skeleton_proposal_loss


@dataclass(frozen=True)
class ProposalComparisonLoss:
    total: torch.Tensor
    objectness: torch.Tensor
    center: torch.Tensor
    radius: torch.Tensor
    supervised_crops: int
    matched_count: int
    assignments: tuple = ()


def query_set_loss(output, nodes, mask, *, positive_class_weight=8., radius_floor=1.,
                   objectness_weight=2., center_weight=3., radius_weight=.2,
                   center_beta=4., match_scale=6.):
    if nodes.ndim != 3 or nodes.shape[-1] != 6 or mask.shape != nodes.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected padded nodes [B,M,6] and boolean mask [B,M]")
    b, q, xyz = output.center_proposals.shape
    if xyz != 3 or b != len(nodes) or output.objectness_logits.shape != (b, q, 2) or output.radius.shape != (b, q, 1):
        raise ValueError("Proposal/target dimensions disagree")
    if min(match_scale, center_beta, positive_class_weight) <= 0:
        raise ValueError("Loss scales must be positive")
    if not all(bool(torch.isfinite(t).all()) for t in (output.center_proposals, output.objectness_logits, output.radius)):
        raise FloatingPointError("Nonfinite query predictions")
    totals, classes, centers, radii, assignments = [], [], [], [], []
    matched = 0
    weights = output.objectness_logits.new_tensor([1., positive_class_weight])
    for i in range(b):
        target = nodes[i, mask[i]]
        count = len(target)
        if count > q:
            raise ValueError(f"Target overflow: {count} targets exceed {q} slots")
        if not count:
            assignments.append(((), ()))
            continue  # Unknown annotation, not verified background.
        if not bool(torch.isfinite(target).all()) or bool((target[:, 4] < 0).any()):
            raise ValueError("Invalid reference targets")
        target_radius = target[:, 4].clamp_min(radius_floor)
        with torch.no_grad():
            cost = (-output.objectness_logits[i].softmax(-1)[:, 1, None]
                    + center_weight * torch.cdist(output.center_proposals[i], target[:, 1:4]) / match_scale
                    + radius_weight * (output.radius[i, :, 0, None] - target_radius[None]).abs() / match_scale)
            pred_np, target_np = linear_sum_assignment(cost.detach().float().cpu().numpy())
            pred_idx = torch.as_tensor(pred_np, dtype=torch.long, device=nodes.device)
            target_idx = torch.as_tensor(target_np, dtype=torch.long, device=nodes.device)
        labels = torch.zeros(q, dtype=torch.long, device=nodes.device)
        labels[pred_idx] = 1
        classification = F.cross_entropy(output.objectness_logits[i], labels, weight=weights)
        distance = torch.linalg.vector_norm(output.center_proposals[i, pred_idx] - target[target_idx, 1:4], dim=-1)
        center = F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=center_beta)
        radius = F.smooth_l1_loss(output.radius[i, pred_idx, 0], target_radius[target_idx], beta=1.)
        totals.append(objectness_weight * classification + center_weight * center + radius_weight * radius)
        classes.append(classification.detach()); centers.append(center.detach()); radii.append(radius.detach())
        assignments.append((tuple(pred_np.tolist()), tuple(target_np.tolist())))
        matched += count
    if not totals:
        zero = (output.center_proposals.sum() + output.objectness_logits.sum() + output.radius.sum()) * 0
        return ProposalComparisonLoss(zero, zero.detach(), zero.detach(), zero.detach(), 0, 0, tuple(assignments))
    return ProposalComparisonLoss(torch.stack(totals).mean(), torch.stack(classes).mean(),
                                  torch.stack(centers).mean(), torch.stack(radii).mean(), len(totals), matched,
                                  tuple(assignments))


def pointwise_comparison_loss(output, points, nodes, mask, recipe):
    losses = []
    for i in range(len(points)):
        if not bool(mask[i].any()):
            continue
        one = type(output)(**{f.name: getattr(output, f.name)[i:i+1] for f in fields(output)})
        losses.append(conservative_skeleton_proposal_loss(
            one, nodes[i:i+1], mask[i:i+1], points[i:i+1],
            positive_distance=recipe["positive_distance"], radius_scale=recipe["radius_scale"],
            positive_class_weight=recipe["positive_class_weight"],
            objectness_weight=recipe["conservative_objectness_weight"],
            center_weight=recipe["conservative_center_weight"], radius_weight=recipe["conservative_radius_weight"],
            offset_regularization_weight=recipe["offset_regularization_weight"],
            non_worsen_weight=recipe["non_worsen_weight"], non_worsen_margin=recipe["non_worsen_margin"],
            center_beta=recipe["conservative_center_beta"], non_worsen_beta=recipe["non_worsen_beta"],
            objectness_radius_floor=recipe["objectness_radius_floor"], radius_target_floor=recipe["radius_target_floor"],
            radius_target_mode=recipe["radius_target_mode"], selection_radius_floor=recipe["selection_radius_floor"],
            selection_radius_ceiling=recipe["selection_radius_ceiling"]))
    if not losses:
        zero = (output.center_proposals.sum() + output.objectness_logits.sum() + output.radius.sum()) * 0
        return ProposalComparisonLoss(zero, zero.detach(), zero.detach(), zero.detach(), 0, 0)
    mean = lambda name: torch.stack([getattr(x, name) for x in losses]).mean()
    return ProposalComparisonLoss(mean("total"), mean("objectness"), mean("center"), mean("radius"),
                                  len(losses), sum(x.positive_count for x in losses))

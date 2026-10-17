"""Experimental per-crop class normalization; original assignment/cost unchanged."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from pointneuron.models.query_loss import ProposalComparisonLoss, query_set_loss

POLICY = "per_crop_equal_positive_negative_class_means_v1"


def loss_from_assignments(output, nodes, mask, assignments, *, objectness_weight=2., center_weight=3.,
                          radius_weight=.2, radius_floor=1., center_beta=4., **unused):
    """Also supports scoring the exact saved GPU assignments on CPU."""
    totals, classes, centers, radii = [], [], [], []
    matched = 0
    for i, (pred, target_indices) in enumerate(assignments):
        target = nodes[i, mask[i]]
        if not len(target):
            if len(pred) or len(target_indices):
                raise ValueError("Empty annotation has no assignment")
            continue  # Unknown annotation, never a background example.
        if len(pred) != len(target) or sorted(target_indices) != list(range(len(target))):
            raise ValueError("Expected a complete one-to-one target assignment")
        pi = torch.as_tensor(pred, dtype=torch.long, device=nodes.device)
        ti = torch.as_tensor(target_indices, dtype=torch.long, device=nodes.device)
        q = output.objectness_logits.shape[1]
        if len(set(pred)) != len(pred) or bool((pi < 0).any()) or bool((pi >= q).any()):
            raise ValueError("Invalid assigned proposal indices")
        positive = torch.zeros(q, dtype=torch.bool, device=nodes.device); positive[pi] = True
        logp = output.objectness_logits[i].log_softmax(-1)
        terms = [-logp[positive, 1].mean()]
        if bool((~positive).any()):
            terms.append(-logp[~positive, 0].mean())
        classification = torch.stack(terms).mean()
        distance = torch.linalg.vector_norm(output.center_proposals[i, pi] - target[ti, 1:4], dim=-1)
        center = F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=center_beta)
        radius = F.smooth_l1_loss(output.radius[i, pi, 0], target[ti, 4].clamp_min(radius_floor), beta=1.)
        totals.append(objectness_weight*classification + center_weight*center + radius_weight*radius)
        classes.append(classification.detach()); centers.append(center.detach()); radii.append(radius.detach())
        matched += len(target)
    if not totals:
        zero = (output.center_proposals.sum() + output.objectness_logits.sum() + output.radius.sum())*0
        return ProposalComparisonLoss(zero, zero.detach(), zero.detach(), zero.detach(), 0, 0, tuple(assignments))
    return ProposalComparisonLoss(torch.stack(totals).mean(), torch.stack(classes).mean(), torch.stack(centers).mean(),
                                  torch.stack(radii).mean(), len(totals), matched, tuple(assignments))


def balanced_query_loss(output, nodes, mask, **recipe):
    # Reuse the original validator and Hungarian implementation, without its CE gradients.
    with torch.no_grad():
        original = query_set_loss(output, nodes, mask, **recipe)
    return loss_from_assignments(output, nodes, mask, original.assignments, **recipe)

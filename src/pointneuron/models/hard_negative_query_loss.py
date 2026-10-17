"""Isolated 50/25/25 matched/all-negative/hard-negative confidence comparison."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from pointneuron.models.query_loss import ProposalComparisonLoss, query_set_loss

POLICY = "half_positive_quarter_all_negative_quarter_top_min_P_N_v1"


def classification_term(logits, positive):
    """Return differentiable CE and detached hard-set indices; ties use query order."""
    count = int(positive.sum())
    if not count:
        raise ValueError("No matched target: unknown annotations must be skipped")
    logp = logits.log_softmax(-1)
    positive_mean = -logp[positive, 1].mean()
    negatives = torch.nonzero(~positive, as_tuple=False).flatten()
    k = min(count, len(negatives))
    if not k:
        return positive_mean, negatives
    negative_mean = -logp[negatives, 0].mean()
    if k == len(negatives):
        # Same reduction and gradients as the balanced comparator, including rounding.
        return torch.stack((positive_mean, negative_mean)).mean(), negatives
    scores = (logits[negatives, 1]-logits[negatives, 0]).detach()
    hard = negatives[torch.argsort(scores, descending=True, stable=True)[:k]]
    hard_mean = -logp[hard, 0].mean()
    return .5*positive_mean + .25*negative_mean + .25*hard_mean, hard


def loss_from_assignments(output, nodes, mask, assignments, *, objectness_weight=2., center_weight=3.,
                          radius_weight=.2, radius_floor=1., center_beta=4., **unused):
    totals, classes, centers, radii = [], [], [], []
    matched = 0
    for i, (pred, target_indices) in enumerate(assignments):
        target = nodes[i, mask[i]]
        if not len(target):
            if len(pred) or len(target_indices):
                raise ValueError("Empty annotation has no assignment")
            continue
        if len(pred) != len(target) or sorted(target_indices) != list(range(len(target))):
            raise ValueError("Expected a complete one-to-one target assignment")
        pi = torch.as_tensor(pred, dtype=torch.long, device=nodes.device)
        ti = torch.as_tensor(target_indices, dtype=torch.long, device=nodes.device)
        q = output.objectness_logits.shape[1]
        if len(set(pred)) != len(pred) or bool((pi < 0).any()) or bool((pi >= q).any()):
            raise ValueError("Invalid assigned proposal indices")
        positive = torch.zeros(q, dtype=torch.bool, device=nodes.device); positive[pi] = True
        classification, _ = classification_term(output.objectness_logits[i], positive)
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


def hard_negative_query_loss(output, nodes, mask, **recipe):
    with torch.no_grad():
        original = query_set_loss(output, nodes, mask, **recipe)
    return loss_from_assignments(output, nodes, mask, original.assignments, **recipe)

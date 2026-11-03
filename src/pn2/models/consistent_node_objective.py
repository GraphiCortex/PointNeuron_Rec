"""Candidate node-set objective with one query measure and exact assignment.

This is a research alternative, not a replacement for the frozen baseline.
Unknown reference-empty boxes abstain. Unused queries mean unused reference
representatives, not certified biological background.
"""
import math

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from .node_objective import NodeLoss


def consistent_node_set_loss(output, target_xyz, *, classification_weight=2.,
                             center_weight=3., center_beta_voxel=4.,
                             match_scale_voxel=6.):
    """Mean over Q slots of CE plus matched, dimensionless spatial penalties.

    L = [wc sum_q CE(q, assigned_label)
         + wg sum_matched Huber(||x-y||/s, beta=b/s)] / Q.

    The all-unused CE is assignment-independent. Assigning query q to target t
    changes it by wc*(log p_unused - log p_used), plus its spatial penalty.
    Hungarian assignment therefore minimizes this SAME loss over all injections.
    No target-count-derived class weights, count head, or additional loss term.
    We retain the old wc/wg/b/s values to avoid selecting new constants here.
    Scaling is a changed objective, not an algebraically equivalent optimizer.
    Ordinary CE does not by itself guarantee calibrated inference probabilities.
    """
    centers, logits = output.center_proposals, output.objectness_logits
    if (centers.ndim != 3 or centers.shape[0] != 1 or centers.shape[-1] != 3
            or centers.shape[1] < 1 or logits.shape != (*centers.shape[:2], 2)
            or target_xyz.ndim != 2 or target_xyz.shape[-1] != 3
            or logits.device != centers.device or target_xyz.device != centers.device
            or any(x.dtype != torch.float32 for x in (centers, logits, target_xyz))):
        raise ValueError('Expected aligned one-box float32 centers/logits and [M,3] targets')
    if any(not math.isfinite(x) or x <= 0 for x in
           (classification_weight, center_weight, center_beta_voxel, match_scale_voxel)):
        raise ValueError('Loss weights/scales must be finite and positive')
    if not all(bool(torch.isfinite(x).all()) for x in (centers, logits, target_xyz)):
        raise FloatingPointError('Nonfinite node predictions or targets')
    count = centers.shape[1]
    if len(target_xyz) > count:
        raise ValueError('Target overflow; no truncation')
    if not len(target_xyz):
        zero = (centers.sum()+logits.sum())*0.
        return NodeLoss(zero, zero.detach(), zero.detach(), (), (), False)
    logp = logits[0].log_softmax(-1)
    with torch.no_grad():
        distances = torch.cdist(centers[0], target_xyz) / match_scale_voxel
        spatial = F.smooth_l1_loss(distances, torch.zeros_like(distances),
                                  beta=center_beta_voxel/match_scale_voxel, reduction='none')
        cost = center_weight*spatial + classification_weight*(logp[:, 0]-logp[:, 1])[:, None]
        pi, ti = linear_sum_assignment(cost.cpu().numpy())
    pred = torch.as_tensor(pi, dtype=torch.long, device=centers.device)
    target = torch.as_tensor(ti, dtype=torch.long, device=centers.device)
    labels = torch.zeros(count, dtype=torch.long, device=centers.device)
    labels[pred] = 1
    classification = F.nll_loss(logp, labels, reduction='mean')
    distance = torch.linalg.vector_norm(centers[0, pred]-target_xyz[target], dim=-1)/match_scale_voxel
    center = F.smooth_l1_loss(distance, torch.zeros_like(distance),
                             beta=center_beta_voxel/match_scale_voxel, reduction='sum')/count
    return NodeLoss(classification_weight*classification+center_weight*center,
                    classification.detach(), center.detach(), tuple(pi.tolist()), tuple(ti.tolist()), True)

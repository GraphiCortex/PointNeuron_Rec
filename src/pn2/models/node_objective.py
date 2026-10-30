"""Node-only form of the reviewed balanced query objective.

No radius matching/loss, topology terms, proximity relabeling or score tuning.
Unused-slot labels describe reference representation, not biological background.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


@dataclass
class NodeLoss:
    total: torch.Tensor
    classification: torch.Tensor
    center: torch.Tensor
    prediction_indices: tuple
    target_indices: tuple
    supervised: bool


def node_set_loss(output, target_xyz, *, classification_weight=2., center_weight=3.,
                  center_beta_voxel=4., match_scale_voxel=6.):
    """One crop, unpadded XYZ targets; empty reference -> abstention.

    Matching: 3*distance/6 - P(used). Loss: 2*balanced CE + 3*Huber(distance).
    Positive and negative CE are separately averaged, then averaged together;
    this avoids allowing 427-505 unused slots to overwhelm the positive group.
    Defaults retain the reviewed geometry/classification scaling, not a claim
    that those values are optimal for this fresh input representation.
    """
    centers, logits = output.center_proposals, output.objectness_logits
    if (centers.ndim != 3 or centers.shape[0] != 1 or centers.shape[-1] != 3
            or logits.shape != (*centers.shape[:2], 2) or target_xyz.ndim != 2
            or target_xyz.shape[-1] != 3 or target_xyz.device != centers.device
            or centers.dtype != torch.float32 or logits.dtype != torch.float32
            or target_xyz.dtype != torch.float32):
        raise ValueError("Expected one float32 crop and unpadded [M,3] targets")
    if any(not math.isfinite(x) or x <= 0 for x in (classification_weight, center_weight, center_beta_voxel, match_scale_voxel)):
        raise ValueError("Loss weights/scales must be positive")
    if not all(bool(torch.isfinite(x).all()) for x in (centers, logits, target_xyz)):
        raise FloatingPointError("Nonfinite node predictions or targets")
    if len(target_xyz) > centers.shape[1]:
        raise ValueError("Target overflow; no truncation")
    if not len(target_xyz):
        zero = (centers.sum()+logits.sum())*0.
        return NodeLoss(zero, zero.detach(), zero.detach(), (), (), False)
    with torch.no_grad():
        cost = center_weight * torch.cdist(centers[0], target_xyz) / match_scale_voxel
        cost -= logits[0].softmax(-1)[:, 1, None]
        pi, ti = linear_sum_assignment(cost.cpu().numpy())
    pred = torch.as_tensor(pi, dtype=torch.long, device=centers.device)
    target = torch.as_tensor(ti, dtype=torch.long, device=centers.device)
    positive = torch.zeros(centers.shape[1], dtype=torch.bool, device=centers.device)
    positive[pred] = True
    logp = logits[0].log_softmax(-1)
    terms = [-logp[positive, 1].mean()]
    if bool((~positive).any()):
        terms.append(-logp[~positive, 0].mean())
    classification = torch.stack(terms).mean()
    distance = torch.linalg.vector_norm(centers[0, pred]-target_xyz[target], dim=-1)
    center = F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=center_beta_voxel)
    return NodeLoss(classification_weight*classification+center_weight*center,
                    classification.detach(), center.detach(), tuple(pi.tolist()), tuple(ti.tolist()), True)

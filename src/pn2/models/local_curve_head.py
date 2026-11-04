"""Experimental dense head: no Transformer, reference access, or graph edges."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CurveVotes:
    centers: torch.Tensor
    support_logits: torch.Tensor
    offsets: torch.Tensor
    features: torch.Tensor


class LocalCurveHead(nn.Module):
    def __init__(self, offset_bound=8.):
        super().__init__()
        self.offset_bound = float(offset_bound)
        self.network = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64),
                                     nn.ReLU(), nn.Linear(64, 4))
        # Begin close to the source cloud, rather than saturated displacements.
        nn.init.normal_(self.network[-1].weight, std=.001)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, points, geometric_features, patch_centers, patch_radius):
        if (points.ndim != 3 or points.shape[-1] != 4
                or geometric_features.shape != (*points.shape[:2], 32)):
            raise ValueError('Misaligned image points/features')
        raw = self.network(geometric_features)
        vector = raw[..., 1:]
        offset = self.offset_bound*vector/torch.sqrt(1+(vector*vector).sum(-1, keepdim=True))
        return CurveVotes(points[..., :3]+offset, raw[..., 0], offset, geometric_features)


def curve_loss(output, support, target_offset, mask, bands):
    """Separate means: predicted support cannot attenuate offset gradients."""
    logits = output.support_logits[0]
    if (support.shape != logits.shape or target_offset.shape != output.offsets[0].shape
            or mask.shape != logits.shape or bands.shape != logits.shape or mask.dtype != torch.bool):
        raise ValueError('Misaligned curve supervision')
    bce = F.binary_cross_entropy_with_logits(logits, support, reduction='none')
    groups = [bce[bands == i].mean() for i in range(5) if bool((bands == i).any())]
    support_loss = torch.stack(groups).mean()
    if bool(mask.any()):
        offset_loss = F.smooth_l1_loss(output.offsets[0][mask]/6., target_offset[mask]/6., beta=1/6.)
    else:
        offset_loss = output.offsets.sum()*0.
    return support_loss+offset_loss, support_loss, offset_loss

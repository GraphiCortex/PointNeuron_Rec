"""Small full-resolution field baseline; no BatchNorm or query assignment."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv3d(inputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(4, outputs), nn.SiLU(),
                         nn.Conv3d(outputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(4, outputs), nn.SiLU())


class CenterlineUNet(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("Base channels must be a positive multiple of four")
        c = base_channels
        self.enc1, self.enc2 = Block(1, c), Block(c, c*2)
        self.bottom = Block(c*2, c*4)
        self.dec2, self.dec1 = Block(c*6, c*2), Block(c*3, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 4:
            raise ValueError("Expected B,1,Z,Y,X native image")
        a = self.enc1(x)
        b = self.enc2(F.max_pool3d(a, 2))
        h = self.bottom(F.max_pool3d(b, 2))
        h = self.dec2(torch.cat((F.interpolate(h, size=b.shape[2:], mode="trilinear", align_corners=False), b), 1))
        h = self.dec1(torch.cat((F.interpolate(h, size=a.shape[2:], mode="trilinear", align_corners=False), a), 1))
        return self.head(h)


def field_loss_terms(logits, target, known, *, negative_aggregation="all", negative_to_positive_ratio=3):
    """Return positive BCE, negative BCE, masked Dice and selected-negative count.

    Unknown voxels have exactly zero direct loss gradient. Skip entirely unknown
    crops before calling. The optional top-k control changes only which known
    negatives contribute to the BCE mean; Dice still uses every known voxel.
    """
    if negative_aggregation not in ("all", "topk"):
        raise ValueError("Unsupported negative aggregation")
    if (isinstance(negative_to_positive_ratio, bool) or not isinstance(negative_to_positive_ratio, int)
            or negative_to_positive_ratio < 1):
        raise ValueError("Negative-to-positive ratio must be a positive integer")
    positive, negative = known & target, known & ~target
    if not positive.any() or not negative.any():
        raise ValueError("Supervised crop requires positive and negative voxels")
    positive_bce = F.softplus(-logits[positive]).mean()
    negative_losses = F.softplus(logits[negative])
    if negative_aggregation == "topk":
        k = min(negative_losses.numel(), negative_to_positive_ratio * int(positive.sum()))
        negative_losses = torch.topk(negative_losses, k, sorted=False).values
    negative_bce = negative_losses.mean()
    p, y = logits.sigmoid()[known], target[known].to(logits.dtype)
    dice = 1 - (2*(p*y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6)
    return positive_bce, negative_bce, dice, negative_losses.numel()


def field_loss(logits, target, known, *, negative_aggregation="all", negative_to_positive_ratio=3):
    """Class-balanced BCE plus masked Dice; default preserves the original loss."""
    positive, negative, dice, _ = field_loss_terms(logits, target, known,
        negative_aggregation=negative_aggregation, negative_to_positive_ratio=negative_to_positive_ratio)
    return .5 * (positive + negative) + dice

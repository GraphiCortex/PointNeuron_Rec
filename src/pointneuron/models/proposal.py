from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SkeletonProposalOutput:
    offsets: torch.Tensor
    objectness_logits: torch.Tensor
    radius: torch.Tensor
    center_proposals: torch.Tensor
    raw: torch.Tensor


class SkeletonProposalHead(nn.Module):
    def __init__(self, in_channels: int = 1216, hidden_channels: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for hidden in hidden_channels:
            layers.extend(
                [
                    nn.Linear(current_channels, hidden, bias=False),
                    nn.BatchNorm1d(hidden),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            )
            current_channels = hidden
        layers.append(nn.Linear(current_channels, 6))
        self.mlp = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, geometric_features: torch.Tensor) -> SkeletonProposalOutput:
        if points.ndim != 3 or points.shape[-1] != 4:
            raise ValueError(f"Expected points with shape [B, N, 4], got {tuple(points.shape)}")
        if geometric_features.ndim != 3:
            raise ValueError(f"Expected geometric features with shape [B, N, F], got {tuple(geometric_features.shape)}")
        if points.shape[:2] != geometric_features.shape[:2]:
            raise ValueError("Point and feature tensors must have matching batch and point dimensions")

        batch_size, point_count, channels = geometric_features.shape
        flat_features = geometric_features.reshape(batch_size * point_count, channels)
        raw = self.mlp(flat_features).reshape(batch_size, point_count, 6)
        objectness_logits = raw[..., :2]
        radius = F.softplus(raw[..., 2:3])
        offsets = raw[..., 3:6]
        center_proposals = points[..., :3] + offsets
        return SkeletonProposalOutput(
            offsets=offsets,
            objectness_logits=objectness_logits,
            radius=radius,
            center_proposals=center_proposals,
            raw=raw,
        )

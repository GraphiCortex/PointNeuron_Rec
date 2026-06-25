from __future__ import annotations

from dataclasses import dataclass
import math

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
    def __init__(
        self,
        in_channels: int = 1219,
        hidden_channels: tuple[int, ...] = (512, 256, 128),
        include_xyz: bool = True,
        coordinate_mode: str = "raw",
    ):
        super().__init__()
        if coordinate_mode not in {"raw", "normalized"}:
            raise ValueError(f"Unknown coordinate_mode {coordinate_mode!r}; expected 'raw' or 'normalized'")
        self.include_xyz = include_xyz
        self.coordinate_mode = coordinate_mode
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

        xyz = points[..., :3]
        normalized_xyz, coordinate_scale = normalize_proposal_coordinates(xyz)
        feature_xyz = xyz if self.coordinate_mode == "raw" else normalized_xyz

        if self.include_xyz:
            proposal_features = torch.cat([feature_xyz, geometric_features], dim=-1)
        else:
            proposal_features = geometric_features

        batch_size, point_count, channels = proposal_features.shape
        flat_features = proposal_features.reshape(batch_size * point_count, channels)
        raw = self.mlp(flat_features).reshape(batch_size, point_count, 6)
        objectness_logits = raw[..., :2]
        raw_radius = F.softplus(raw[..., 2:3])
        raw_offsets = raw[..., 3:6]
        radius = raw_radius
        if self.coordinate_mode == "normalized":
            offsets = raw_offsets * coordinate_scale
        else:
            offsets = raw_offsets
        center_proposals = xyz + offsets
        return SkeletonProposalOutput(
            offsets=offsets,
            objectness_logits=objectness_logits,
            radius=radius,
            center_proposals=center_proposals,
            raw=raw,
        )

    def initialize_conservative(self, radius: float = 1.0, background_logit: float = 0.5) -> None:
        """Initialize the proposal head close to identity: q ~= xyz."""
        final = self.mlp[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected the final proposal layer to be nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        final.bias.data[0] = float(background_logit)
        final.bias.data[1] = -float(background_logit)
        final.bias.data[2] = inverse_softplus(float(radius))
        final.bias.data[3:6].zero_()


def normalize_proposal_coordinates(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = coords.mean(dim=1, keepdim=True)
    centered = coords - center
    scale = centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return centered / scale, scale


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("Softplus inverse requires a positive value")
    return math.log(math.expm1(value))

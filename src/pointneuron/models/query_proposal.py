"""Native query proposals; local coordinates are decoded into voxel units."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QueryDecoderConfig:
    in_channels: int = 1216
    query_count: int = 512
    model_dim: int = 128
    layers: int = 3
    heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("in_channels", "query_count", "model_dim", "layers", "heads", "feedforward_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_dim % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Invalid attention dimensions or dropout")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QueryProposalOutput:
    center_proposals: torch.Tensor
    objectness_logits: torch.Tensor
    radius: torch.Tensor
    local_centers: torch.Tensor
    decoder_features: torch.Tensor


class QueryProposalHead(nn.Module):
    def __init__(self, config: QueryDecoderConfig = QueryDecoderConfig()):
        super().__init__()
        self.config = config
        self.memory_projection = nn.Linear(config.in_channels, config.model_dim)
        self.position = nn.Sequential(nn.Linear(3, config.model_dim), nn.ReLU(),
                                      nn.Linear(config.model_dim, config.model_dim))
        self.queries = nn.Embedding(config.query_count, config.model_dim)
        layer = nn.TransformerDecoderLayer(config.model_dim, config.heads, config.feedforward_dim,
                                           config.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.layers)
        # Decoder layers are cloned by torch; give their matrices independent draws.
        for block in self.decoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.prediction = nn.Sequential(nn.Linear(config.model_dim, config.model_dim), nn.ReLU(),
                                        nn.Linear(config.model_dim, 6))
        nn.init.normal_(self.queries.weight, std=0.02)
        final = self.prediction[-1]
        nn.init.normal_(final.weight, std=0.002)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[:2] = torch.tensor([0.5, -0.5])
            final.bias[2] = math.log(math.expm1(1.0))

    def forward(self, points, geometric_features, patch_centers, patch_radius):
        if points.ndim != 3 or points.shape[-1] != 4 or points.shape[1] < 2:
            raise ValueError("Expected points [B,N>=2,4]")
        if geometric_features.shape != (*points.shape[:2], self.config.in_channels):
            raise ValueError("Feature dimensions disagree with decoder configuration")
        if patch_centers.shape != (len(points), 3):
            raise ValueError("Expected patch centers [B,3]")
        scale = torch.as_tensor(patch_radius, dtype=points.dtype, device=points.device)
        if scale.numel() != 1 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
            raise ValueError("Patch radius must be finite and positive")
        if not all(bool(torch.isfinite(t).all()) for t in (points, geometric_features, patch_centers)):
            raise ValueError("Nonfinite model input")
        local_xyz = (points[..., :3] - patch_centers[:, None, :]) / scale
        memory = self.memory_projection(geometric_features) + self.position(local_xyz)
        queries = self.queries.weight[None].expand(len(points), -1, -1)
        decoded = self.decoder(queries, memory)
        raw = self.prediction(decoded)
        local = raw[..., 3:6]  # Unbounded: shared rotations can move targets outside the original cube.
        return QueryProposalOutput(patch_centers[:, None, :] + local * scale,
                                   raw[..., :2], F.softplus(raw[..., 2:3]), local, decoded)

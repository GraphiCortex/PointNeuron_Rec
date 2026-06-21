from __future__ import annotations

import torch
from torch import nn


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, mlp_channels: tuple[int, ...], k: int = 20):
        super().__init__()
        self.k = k
        layers: list[nn.Module] = []
        current_channels = in_channels * 2
        for out_channels in mlp_channels:
            layers.extend(
                [
                    nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            )
            current_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
        graph_features = gather_edge_features(features, knn_indices)
        graph_features = graph_features.permute(0, 3, 1, 2).contiguous()
        edge_features = self.mlp(graph_features)
        return edge_features.max(dim=-1).values.transpose(1, 2).contiguous()


class DGCNNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        edge_mlp_channels: tuple[tuple[int, ...], ...] = ((64, 64), (64, 64), (64,)),
        global_feature_dim: int = 1024,
        feature_dim: int | None = None,
        k: int = 20,
    ):
        super().__init__()
        if input_channels != 4:
            raise ValueError("DGCNNEncoder currently expects point features (x, y, z, intensity)")
        self.k = k
        self.feature_dim = feature_dim
        self.edge_blocks = nn.ModuleList()
        current_channels = input_channels
        local_dims: list[int] = []
        for mlp_channels in edge_mlp_channels:
            self.edge_blocks.append(EdgeConvBlock(current_channels, mlp_channels, k=k))
            current_channels = mlp_channels[-1]
            local_dims.append(current_channels)

        local_feature_dim = sum(local_dims)
        self.global_mlp = nn.Sequential(
            nn.Linear(local_feature_dim, global_feature_dim, bias=False),
            nn.BatchNorm1d(global_feature_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        geometric_feature_dim = local_feature_dim + global_feature_dim
        self.geometric_feature_dim = geometric_feature_dim
        if feature_dim is None:
            self.projection = nn.Identity()
            self.output_dim = geometric_feature_dim
        else:
            self.projection = nn.Sequential(
                nn.Linear(geometric_feature_dim, feature_dim, bias=False),
                nn.BatchNorm1d(feature_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.output_dim = feature_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 4:
            raise ValueError(f"Expected points with shape [B, N, 4], got {tuple(points.shape)}")

        features = normalize_point_features(points)
        block_outputs: list[torch.Tensor] = []
        for block in self.edge_blocks:
            knn_indices = knn(features[..., :3].detach(), block.k)
            features = block(features, knn_indices)
            block_outputs.append(features)

        local_features = torch.cat(block_outputs, dim=-1)
        batch_size, point_count, channels = local_features.shape

        pooled = local_features.max(dim=1).values
        global_features = self.global_mlp(pooled)
        repeated_global = global_features.unsqueeze(1).expand(-1, point_count, -1)
        geometric_features = torch.cat([local_features, repeated_global], dim=-1)

        flattened = geometric_features.reshape(batch_size * point_count, geometric_features.shape[-1])
        projected = self.projection(flattened)
        return projected.reshape(batch_size, point_count, self.output_dim)


def normalize_point_features(points: torch.Tensor) -> torch.Tensor:
    coords = points[..., :3]
    raw_intensity = points[..., 3:4]
    intensity_scale = raw_intensity.amax(dim=1, keepdim=True).clamp_min(255.0)
    intensity = raw_intensity / intensity_scale

    center = coords.mean(dim=1, keepdim=True)
    centered = coords - center
    scale = centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    normalized_coords = centered / scale
    return torch.cat([normalized_coords, intensity], dim=-1)


def knn(coords: torch.Tensor, k: int) -> torch.Tensor:
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords with shape [B, N, 3], got {tuple(coords.shape)}")
    point_count = coords.shape[1]
    if point_count < 2:
        raise ValueError("kNN requires at least two points")

    neighbor_count = min(k + 1, point_count)
    distances = torch.cdist(coords, coords)
    indices = distances.topk(k=neighbor_count, dim=-1, largest=False).indices
    return indices[..., 1:]


def gather_edge_features(features: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
    batch_size, point_count, channels = features.shape
    neighbor_count = knn_indices.shape[-1]

    batch_offsets = torch.arange(batch_size, device=features.device).view(batch_size, 1, 1) * point_count
    flat_indices = (knn_indices + batch_offsets).reshape(-1)
    flat_features = features.reshape(batch_size * point_count, channels)
    neighbors = flat_features[flat_indices].reshape(batch_size, point_count, neighbor_count, channels)
    central = features.unsqueeze(2).expand(-1, -1, neighbor_count, -1)
    return torch.cat([central, neighbors - central], dim=-1)

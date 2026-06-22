from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ConnectivityOutput:
    latent: torch.Tensor
    adjacency_logits: torch.Tensor


class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        normalized = normalize_adjacency(adjacency)
        aggregated = normalized @ features
        output = self.linear(aggregated)
        output = self.norm(output)
        return F.relu(output, inplace=True)


class ConnectivityGAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1220,
        hidden_channels: tuple[int, ...] = (32, 32, 48, 64, 64, 80, 96, 96, 102, 128, 128, 144),
        stem_channels: int = 16,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_channels, stem_channels, bias=False),
            nn.LayerNorm(stem_channels),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        current_channels = stem_channels
        for channels in hidden_channels:
            layers.append(GraphConvolution(current_channels, channels))
            current_channels = channels
        self.graph_layers = nn.ModuleList(layers)
        self.decoder_bias = nn.Parameter(torch.tensor(-4.0))
        self.output_dim = current_channels

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> ConnectivityOutput:
        if node_features.ndim != 2:
            raise ValueError(f"Expected node features with shape [N, F], got {tuple(node_features.shape)}")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Expected adjacency with shape [N, N], got {tuple(adjacency.shape)}")
        if node_features.shape[0] != adjacency.shape[0]:
            raise ValueError("Node features and adjacency must have the same node count")

        latent = self.stem(node_features)
        for layer in self.graph_layers:
            latent = layer(latent, adjacency)
        scale = float(latent.shape[-1]) ** 0.5
        adjacency_logits = (latent @ latent.transpose(0, 1)) / scale + self.decoder_bias
        return ConnectivityOutput(latent=latent, adjacency_logits=adjacency_logits)


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = adjacency.float()
    count = adjacency.shape[0]
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    degree_inv_sqrt = degree.rsqrt()
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


def adjacency_reconstruction_loss(
    logits: torch.Tensor,
    target_adjacency: torch.Tensor,
    positive_weight: float | None = None,
) -> torch.Tensor:
    if logits.shape != target_adjacency.shape:
        raise ValueError("Logits and target adjacency must have the same shape")
    target = target_adjacency.float()
    mask = torch.triu(torch.ones_like(target, dtype=torch.bool), diagonal=1)
    logits_flat = logits[mask]
    target_flat = target[mask]
    if positive_weight is None:
        positives = target_flat.sum().clamp_min(1.0)
        negatives = (target_flat.numel() - target_flat.sum()).clamp_min(1.0)
        positive_weight_tensor = negatives / positives
    else:
        positive_weight_tensor = logits_flat.new_tensor(float(positive_weight))
    return F.binary_cross_entropy_with_logits(
        logits_flat,
        target_flat,
        pos_weight=positive_weight_tensor,
    )

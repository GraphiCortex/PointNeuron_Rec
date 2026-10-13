"""PN2 native-operator PointNeXt-S segmentation-feature prototype.

Architecture adapted from Guocheng Qian's OpenPoints (MIT); see
docs/pointneuron2/pointnext_reference/LICENSE.openpoints and reference_manifest.json.
This implements the S recipe, not larger InvResMLP variants or benchmark tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import torch
from torch import nn

from pointneuron.models.pointnext_ops import farthest_point_indices, ball_query_indices, gather_points, interpolate_features


@dataclass(frozen=True)
class PointNextNativeConfig:
    width: int = 32
    radius: float = .1
    neighbors: int = 32
    chunk_size: int = 256
    minimum_points: int = 256
    kind: str = "pointnext_s_native_v1"

    def __post_init__(self):
        if asdict(self) != {"width": 32, "radius": .1, "neighbors": 32, "chunk_size": 256,
                          "minimum_points": 256, "kind": "pointnext_s_native_v1"}:
            raise ValueError("Only the declared native PointNeXt-S prototype recipe is supported")


def _conv(inputs, outputs, dimensions, *, norm=True, activation=True):
    conv = nn.Conv1d if dimensions == 1 else nn.Conv2d
    layers = [conv(inputs, outputs, 1, bias=not norm)]
    if norm:
        layers.append((nn.BatchNorm1d if dimensions == 1 else nn.BatchNorm2d)(outputs))
    if activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResidualAbstraction(nn.Module):
    def __init__(self, inputs, outputs, radius, config):
        super().__init__()
        self.radius, self.config = radius, config
        self.local = nn.Sequential(_conv(inputs+3, outputs//2, 2), _conv(outputs//2, outputs, 2, activation=False))
        self.skip = _conv(inputs, outputs, 1, norm=False, activation=False)

    def forward(self, xyz, features):
        indices = farthest_point_indices(xyz, xyz.shape[1]//4)
        query = gather_points(xyz, indices)
        neighbors = ball_query_indices(xyz, query, self.radius, self.config.neighbors, self.config.chunk_size)
        relative = (gather_points(xyz, neighbors) - query[:, :, None]) / self.radius
        grouped = gather_points(features, neighbors)
        local = torch.cat((relative, grouped), -1).permute(0, 3, 1, 2).contiguous()
        aggregated = self.local(local).max(-1).values
        residual = self.skip(gather_points(features, indices).transpose(1, 2).contiguous())
        return query, torch.relu(aggregated+residual).transpose(1, 2).contiguous()


class FeatureRestoration(nn.Module):
    def __init__(self, coarse_channels, fine_channels, config):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.mlp = nn.Sequential(_conv(coarse_channels+fine_channels, fine_channels, 1),
                                 _conv(fine_channels, fine_channels, 1))

    def forward(self, fine_xyz, fine_features, coarse_xyz, coarse_features):
        restored = interpolate_features(fine_xyz, coarse_xyz, coarse_features, self.chunk_size)
        return self.mlp(torch.cat((fine_features, restored), -1).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()


class PointNextNativeEncoder(nn.Module):
    def __init__(self, config=PointNextNativeConfig()):
        super().__init__()
        self.config, self.output_dim = config, config.width
        self.stem = _conv(4, config.width, 1, norm=False, activation=False)
        widths = [config.width * 2**i for i in range(5)]
        self.abstraction = nn.ModuleList([ResidualAbstraction(widths[i], widths[i+1], config.radius*2**i, config) for i in range(4)])
        self.restoration = nn.ModuleList([FeatureRestoration(widths[i+1], widths[i], config) for i in range(4)])

    def forward(self, local_xyz, attributes):
        if (local_xyz.ndim != 3 or local_xyz.shape[-1] != 3 or local_xyz.shape[1] < self.config.minimum_points
                or attributes.shape != (*local_xyz.shape[:2], 4) or attributes.device != local_xyz.device
                or local_xyz.dtype != torch.float32 or attributes.dtype != torch.float32
                or not bool(torch.isfinite(attributes).all()) or not bool(torch.isfinite(local_xyz).all())):
            raise ValueError("Native S hierarchy needs aligned float32 [B,N>=256,3/4]; no input padding or stage changes")
        positions = [local_xyz]
        features = [self.stem(attributes.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()]
        for layer in self.abstraction:
            p, f = layer(positions[-1], features[-1]); positions.append(p); features.append(f)
        out = features[-1]
        for i in range(3, -1, -1):
            out = self.restoration[i](positions[i], features[i], positions[i+1], out)
        return out


def normalize_native_inputs(points, patch_centers, image_scales, patch_radius):
    if points.ndim != 3 or points.shape[-1] != 4 or patch_centers.shape != (len(points), 3):
        raise ValueError("Expected native points [B,N,4] and patch centers [B,3]")
    scales = torch.as_tensor(image_scales, device=points.device, dtype=torch.float32).reshape(-1)
    radius = torch.as_tensor(patch_radius, device=points.device, dtype=torch.float32)
    if (scales.shape != (len(points),) or radius.numel() != 1 or not bool(torch.isfinite(radius)) or not bool(radius > 0)
            or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any())
            or points.dtype != torch.float32 or patch_centers.dtype != torch.float32
            or not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(patch_centers).all())):
        raise ValueError("Invalid full-image intensity scale, coordinates or crop radius")
    intensity = points[..., 3:4] / scales[:, None, None]
    if bool((intensity < 0).any()) or bool((intensity > 1+1e-6).any()):
        raise ValueError("Raw intensity escapes the declared full-image scale")
    local = (points[..., :3] - patch_centers[:, None]) / radius
    return local, torch.cat((local, intensity), -1)

"""Image-conditioned geometric queries using the pinned native head's modules.

FPS supplies starting locations only. Full point memory remains in attention;
centers are unbounded offsets from starts, with crop-relative public locals.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GeometricQueryHead(nn.Module):
    """Reuse a freshly constructed head without adding parameters or RNG draws."""

    def __init__(self, fresh_head):
        super().__init__()
        self.config = fresh_head.config
        for name in ("memory_projection", "position", "queries", "decoder", "prediction"):
            self.add_module(name, getattr(fresh_head, name))
        self.train(fresh_head.training)

    def starting_indices(self, local_xyz):
        # The imported operator is hash-checked by the experiment factory.
        from pointneuron.models.pointnext_ops import farthest_point_indices
        return farthest_point_indices(local_xyz, min(self.config.query_count, local_xyz.shape[1]))

    def forward(self, points, geometric_features, patch_centers, patch_radius):
        from pointneuron.models.query_proposal import QueryProposalOutput
        from pointneuron.models.pointnext_ops import gather_points
        if (points.ndim != 3 or points.shape[-1] != 4 or points.shape[1] < 2
                or points.dtype != torch.float32):
            raise ValueError("Expected float32 points [B,N>=2,4]")
        if (geometric_features.shape != (*points.shape[:2], self.config.in_channels)
                or geometric_features.dtype != points.dtype or geometric_features.device != points.device
                or patch_centers.shape != (len(points), 3) or patch_centers.dtype != points.dtype
                or patch_centers.device != points.device):
            raise ValueError("Misaligned features or crop centers")
        scale = torch.as_tensor(patch_radius, dtype=points.dtype, device=points.device)
        if scale.numel() != 1 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
            raise ValueError("Patch radius must be finite and positive")
        if not all(bool(torch.isfinite(x).all()) for x in (points, geometric_features, patch_centers)):
            raise ValueError("Nonfinite model input")
        local_xyz = (points[..., :3] - patch_centers[:, None, :]) / scale
        memory = self.memory_projection(geometric_features) + self.position(local_xyz)
        indices = self.starting_indices(local_xyz)
        starts = gather_points(points[..., :3], indices)
        query = self.queries.weight[None, :indices.shape[1]] + gather_points(memory, indices)
        decoded = self.decoder(query, memory)
        raw = self.prediction(decoded)
        centers = starts + raw[..., 3:6] * scale
        local_centers = (centers - patch_centers[:, None, :]) / scale
        return QueryProposalOutput(centers, raw[..., :2], F.softplus(raw[..., 2:3]), local_centers, decoded)

"""Experimental straight-through relaxation of image-guided point selection."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from pointneuron.models.guide_distances import nearest_distances


@dataclass
class AdaptiveSample:
    points: torch.Tensor
    indices: torch.Tensor
    probabilities: torch.Tensor


class AdaptiveMedialSampler(nn.Module):
    """Learn distance decay through a biased straight-through estimator.

    Training uses sequential masked Gumbel-softmax. The forward cloud contains
    exact, unique candidates; the backward pass uses soft assignment gradients.
    Evaluation uses seeded Gumbel top-k, preserving the exploration mixture.
    Extraction, masks and hard indices are
    discrete. This is a surrogate gradient, not exact end-to-end differentiation.
    Candidate pools remain bounded; full guides can use blocked distance
    reductions. Relaxed selection still scales with B*K*N during training.
    """
    def __init__(self, initial_alpha: float = 0.25, uniform_fraction: float = 0.25,
                 temperature: float = 1.0, spacing_xyz=(1.0, 1.0, 1.0),
                 distance_chunk_size: int | None = None):
        super().__init__()
        if not math.isfinite(initial_alpha) or initial_alpha <= 0:
            raise ValueError("initial_alpha must be finite and positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= uniform_fraction <= 1:
            raise ValueError("uniform_fraction must be in [0, 1]")
        if len(spacing_xyz) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing_xyz):
            raise ValueError("spacing_xyz must contain three finite positive values")
        if distance_chunk_size is not None and (isinstance(distance_chunk_size, bool) or
                not isinstance(distance_chunk_size, int) or distance_chunk_size < 1):
            raise ValueError('distance_chunk_size must be a positive integer or None')
        self.log_alpha = nn.Parameter(torch.tensor(initial_alpha + math.log(-math.expm1(-initial_alpha))))
        self.register_buffer("spacing_xyz", torch.tensor(spacing_xyz, dtype=torch.float32))
        self.uniform_fraction = uniform_fraction
        self.temperature = temperature
        self.distance_chunk_size = distance_chunk_size

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)

    def forward(self, points: torch.Tensor, image_guide_xyz: torch.Tensor, count: int,
                guide_mask: torch.Tensor | None = None, seed: int = 0) -> AdaptiveSample:
        if points.ndim != 3 or points.shape[-1] != 4 or image_guide_xyz.ndim != 3 or image_guide_xyz.shape[-1] != 3:
            raise ValueError("Expected points [B,N,4] and image guide [B,M,3]")
        if points.shape[0] != image_guide_xyz.shape[0] or not 0 < count <= points.shape[1]:
            raise ValueError("Batch sizes must agree and count must be in [1,N]")
        if not torch.isfinite(points).all() or not torch.isfinite(image_guide_xyz).all():
            raise ValueError("Points and image guide must be finite")
        if not points.is_floating_point() or not image_guide_xyz.is_floating_point():
            raise ValueError("Points and image guide must be floating point")
        n = points.shape[1]
        if image_guide_xyz.shape[1]:
            if guide_mask is None:
                guide_mask = torch.ones(image_guide_xyz.shape[:2], dtype=torch.bool, device=points.device)
            if guide_mask.shape != image_guide_xyz.shape[:2] or guide_mask.dtype != torch.bool:
                raise ValueError("guide_mask must be boolean [B,M]")
            if self.distance_chunk_size is None:
                pair_distances = torch.cdist(points[..., :3] * self.spacing_xyz, image_guide_xyz * self.spacing_xyz)
                distances = pair_distances.masked_fill(~guide_mask[:, None, :], torch.inf).amin(dim=-1)
            else:
                distances = nearest_distances(points[..., :3] * self.spacing_xyz,
                                              image_guide_xyz * self.spacing_xyz, guide_mask,
                                              self.distance_chunk_size)
            # An empty image skeleton contributes no prior. Zero distances give
            # a uniform distribution and zero decay gradient for that record.
            distances = torch.where(guide_mask.any(dim=-1)[:, None], distances, torch.zeros_like(distances))
            guided = (-self.alpha * distances).softmax(dim=-1)
        else:
            guided = points.new_full((points.shape[0], n), 1 / n)
        probabilities = (1 - self.uniform_fraction) * guided + self.uniform_fraction / n
        logits = probabilities.clamp_min(torch.finfo(points.dtype).tiny).log()
        if not self.training:
            generator = torch.Generator(device=points.device).manual_seed(seed)
            uniform = torch.rand(logits.shape, device=points.device, dtype=points.dtype, generator=generator)
            uniform = uniform.clamp(torch.finfo(points.dtype).eps, 1 - torch.finfo(points.dtype).eps)
            noise = -torch.log(-torch.log(uniform))
            indices = (logits + noise).topk(count, dim=-1).indices
            return AdaptiveSample(points.gather(1, indices[..., None].expand(-1, -1, 4)), indices, probabilities)
        available = torch.ones_like(logits, dtype=torch.bool)
        selected, hard_indices = [], []
        for _ in range(count):
            # Clone the mask: subsequent updates must not mutate saved autograd state.
            masked_logits = logits.masked_fill(~available.clone(), -torch.inf)
            assignment = F.gumbel_softmax(masked_logits, tau=self.temperature, hard=True, dim=-1)
            index = assignment.argmax(dim=-1)
            selected.append(torch.einsum("bn,bnc->bc", assignment, points))
            hard_indices.append(index)
            available.scatter_(1, index[:, None], False)
        return AdaptiveSample(torch.stack(selected, dim=1), torch.stack(hard_indices, dim=1), probabilities)


def symmetric_chamfer(selected_xyz: torch.Tensor, image_guide_xyz: torch.Tensor,
                      distance_chunk_size: int | None = None) -> torch.Tensor:
    """Mean symmetric squared distance; inputs must use the same physical units.

    This geometric objective alone can favor noisy skeletons. It is intended as
    an auxiliary loss alongside downstream reconstruction supervision.
    """
    if (selected_xyz.ndim != 3 or image_guide_xyz.ndim != 3 or
            selected_xyz.shape[-1] != 3 or image_guide_xyz.shape[-1] != 3 or
            selected_xyz.shape[0] != image_guide_xyz.shape[0] or
            not selected_xyz.shape[1] or not image_guide_xyz.shape[1]):
        raise ValueError("Chamfer requires nonempty [B,N,3] and [B,M,3] point sets")
    if distance_chunk_size is None:
        distances = torch.cdist(selected_xyz, image_guide_xyz).square()
        return distances.amin(dim=-1).mean() + distances.amin(dim=-2).mean()
    return (nearest_distances(selected_xyz, image_guide_xyz, chunk_size=distance_chunk_size, squared=True).mean()
            + nearest_distances(image_guide_xyz, selected_xyz, chunk_size=distance_chunk_size, squared=True).mean())


def masked_chamfer(selected_xyz: torch.Tensor, guide_xyz: torch.Tensor, guide_mask: torch.Tensor,
                   distance_chunk_size: int | None = None) -> torch.Tensor:
    """Average per-record Chamfer; empty extracted guides have zero auxiliary loss."""
    if guide_mask.shape != guide_xyz.shape[:2] or guide_mask.dtype != torch.bool:
        raise ValueError("guide_mask must be boolean [B,M]")
    losses = []
    for b in range(selected_xyz.shape[0]):
        valid = guide_xyz[b, guide_mask[b]]
        losses.append(symmetric_chamfer(selected_xyz[b:b + 1], valid[None], distance_chunk_size) if len(valid)
                      else selected_xyz[b].sum() * 0)
    return torch.stack(losses).mean()

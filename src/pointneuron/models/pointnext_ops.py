"""Native PyTorch point operations for the documented PN2 PointNeXt prototype.

Reference operation conventions: OpenPoints bc778ec20c51fa9a72861c0ffe32718815367b6a.
No compiled extension or claim of kernel/throughput equivalence. Coordinates
are fixed inputs; sampling/interpolation weights do not backpropagate geometry.
"""
from __future__ import annotations

import math
import torch


def _xyz(x):
    if x.ndim != 3 or x.shape[-1] != 3 or min(x.shape[:2]) < 1 or x.dtype != torch.float32:
        raise ValueError("Expected nonempty float32 coordinates [B,N,3]")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite point coordinates")


def gather_points(values, indices):
    """Gather [B,N,C] in the supplied index order, including grouped indices."""
    if values.ndim != 3 or indices.ndim < 2 or indices.shape[0] != values.shape[0] or indices.dtype != torch.long:
        raise ValueError("Invalid batched gather shapes/indices")
    if indices.device != values.device or bool((indices < 0).any()) or bool((indices >= values.shape[1]).any()):
        raise ValueError("Gather indices outside point support")
    batch = torch.arange(len(values), device=values.device).reshape(-1, *([1] * (indices.ndim-1)))
    return values[batch, indices]


@torch.no_grad()
def farthest_point_indices(xyz, count):
    """Start at index 0; first-index tie break; distinct indices, even on ties."""
    _xyz(xyz)
    b, n, _ = xyz.shape
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= n:
        raise ValueError("FPS requires an integer count within unique support")
    output = torch.empty(b, count, dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), float("inf"), device=xyz.device)
    used = torch.zeros(b, n, dtype=torch.bool, device=xyz.device)
    selected = torch.zeros(b, dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(count):
        output[:, i] = selected
        used[batch, selected] = True
        delta = xyz - xyz[batch, selected, None]
        distance = torch.minimum(distance, delta.square().sum(-1))
        selected = distance.masked_fill(used, -1).argmax(-1)
    return output


@torch.no_grad()
def ball_query_indices(support, query, radius, neighbors, chunk_size=256):
    """Strict radius; support-index order; repeat first member to fill K.

    Repeated neighborhood indices are the reference grouping convention, not
    duplicate input-cloud padding. Empty external query balls fail explicitly.
    """
    _xyz(support); _xyz(query)
    if support.shape[0] != query.shape[0] or support.device != query.device:
        raise ValueError("Ball query supports must share batch/device")
    if not math.isfinite(radius) or radius <= 0 or isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
        raise ValueError("Invalid ball radius/neighbor count")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Invalid distance chunk size")
    n = support.shape[1]
    order = torch.arange(n, device=support.device)[None, None]
    chunks = []
    for start in range(0, query.shape[1], chunk_size):
        delta = query[:, start:start+chunk_size, None] - support[:, None]
        candidates = order.expand(delta.shape[:-1]).masked_fill(delta.square().sum(-1) >= radius**2, n)
        indices = candidates.topk(min(neighbors, n), dim=-1, largest=False, sorted=True).values
        first = indices[..., :1]
        if bool((first == n).any()):
            raise ValueError("Empty ball query; no implicit outside-radius neighbor")
        indices = torch.where(indices == n, first, indices)
        if indices.shape[-1] < neighbors:
            indices = torch.cat((indices, first.expand(*first.shape[:-1], neighbors-indices.shape[-1])), -1)
        chunks.append(indices)
    return torch.cat(chunks, 1)


def interpolate_features(query, support, features, chunk_size=256):
    """Inverse-Euclidean-distance three-neighbor interpolation; feature gradients.

    Use 1/(distance+1e-8), stable support-index tie order. One/two-point supports
    use their actual available neighbors rather than synthetic far neighbors.
    """
    _xyz(query); _xyz(support)
    if (query.shape[0] != support.shape[0] or query.device != support.device
            or features.shape[:2] != support.shape[:2] or features.ndim != 3
            or features.device != support.device or features.dtype != torch.float32
            or not bool(torch.isfinite(features).all())):
        raise ValueError("Interpolation requires aligned float32 support features")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Invalid distance chunk size")
    chunks = []
    for start in range(0, query.shape[1], chunk_size):
        with torch.no_grad():
            distance = torch.cdist(query[:, start:start+chunk_size], support, compute_mode="donot_use_mm_for_euclid_dist")
            indices = distance.argsort(dim=-1, stable=True)[..., :min(3, support.shape[1])]
            reciprocal = (distance.gather(-1, indices) + 1e-8).reciprocal()
            weights = reciprocal / reciprocal.sum(-1, keepdim=True)
        chunks.append((gather_points(features, indices) * weights[..., None]).sum(-2))
    return torch.cat(chunks, 1)

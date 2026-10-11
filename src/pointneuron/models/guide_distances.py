"""Exact nearest-guide reductions with bounded pairwise storage and first-order gradients."""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


class _NearestDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, reference, mask, chunk_size, squared):
        nearest = query.new_full(query.shape[:2], torch.inf)
        ties = torch.zeros(query.shape[:2], dtype=torch.long, device=query.device)
        for q in range(0, query.shape[1], chunk_size):
            qs = slice(q, q + chunk_size)
            best, counts = nearest[:, qs], ties[:, qs]
            for r in range(0, reference.shape[1], chunk_size):
                rs = slice(r, r + chunk_size)
                distances = torch.cdist(query[:, qs], reference[:, rs], compute_mode='donot_use_mm_for_euclid_dist')
                distances.masked_fill_(~mask[:, None, rs], torch.inf)
                minimum = distances.amin(dim=-1)
                count = ((distances == minimum[..., None]) & torch.isfinite(distances)).sum(dim=-1)
                counts = torch.where(minimum < best, count,
                                     torch.where(minimum == best, counts + count, counts))
                best = torch.minimum(best, minimum)
            nearest[:, qs], ties[:, qs] = best, counts
        ctx.save_for_backward(query, reference, mask, nearest, ties)
        ctx.chunk_size, ctx.squared = chunk_size, squared
        return nearest.square() if squared else nearest

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, reference, mask, nearest, ties = ctx.saved_tensors
        grad_query = torch.zeros_like(query) if ctx.needs_input_grad[0] else None
        grad_reference = torch.zeros_like(reference) if ctx.needs_input_grad[1] else None
        c = ctx.chunk_size
        for q in range(0, query.shape[1], c):
            qs = slice(q, q + c)
            for r in range(0, reference.shape[1], c):
                rs = slice(r, r + c)
                distances = torch.cdist(query[:, qs], reference[:, rs], compute_mode='donot_use_mm_for_euclid_dist')
                matched = (distances == nearest[:, qs, None]) & mask[:, None, rs] & torch.isfinite(distances)
                weight = matched * (grad_output[:, qs] / ties[:, qs].clamp_min(1))[..., None]
                if ctx.squared:
                    weight = weight * 2
                else:
                    weight = weight / torch.where(distances > 0, distances, torch.ones_like(distances))
                delta = query[:, qs, None, :] - reference[:, None, rs, :]
                derivative = delta * weight[..., None]
                if grad_query is not None:
                    grad_query[:, qs] += derivative.sum(dim=2)
                if grad_reference is not None:
                    grad_reference[:, rs] -= derivative.sum(dim=1)
        return grad_query, grad_reference, None, None, None


def nearest_distances(query, reference, reference_mask=None, chunk_size=256, squared=False):
    """Reduce all guide vertices without keeping dense pair matrices for backward.

    Exact distance ties share gradients as torch.amin does. Coincident points
    have zero distance gradient. First-order derivatives only; no guide truncation.
    Pairwise workspace scales with batch * chunk_size**2, plus input/output storage.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if (query.ndim != 3 or reference.ndim != 3 or query.shape[-1] != 3 or reference.shape[-1] != 3
            or query.shape[0] != reference.shape[0] or query.device != reference.device or query.dtype != reference.dtype
            or not query.is_floating_point() or not torch.isfinite(query).all() or not torch.isfinite(reference).all()):
        raise ValueError('Expected finite matching [B,N,3] and [B,M,3] tensors')
    if reference_mask is None:
        reference_mask = torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
    if reference_mask.shape != reference.shape[:2] or reference_mask.dtype != torch.bool or reference_mask.device != reference.device:
        raise ValueError('reference_mask must be boolean [B,M] on the reference device')
    return _NearestDistance.apply(query, reference, reference_mask, chunk_size, squared)

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from pointneuron.models.proposal import SkeletonProposalOutput


@dataclass(frozen=True)
class SkeletonProposalTargets:
    objectness_labels: torch.Tensor
    matched_centers: torch.Tensor
    matched_radius: torch.Tensor
    matched_distance: torch.Tensor
    positive_mask: torch.Tensor


@dataclass(frozen=True)
class SkeletonProposalLoss:
    total: torch.Tensor
    objectness: torch.Tensor
    center: torch.Tensor
    radius: torch.Tensor
    positive_count: int
    total_count: int


def build_skeleton_proposal_targets(
    points: torch.Tensor,
    skeleton_nodes: torch.Tensor,
    skeleton_mask: torch.Tensor,
    positive_distance: float = 6.0,
    radius_scale: float = 1.5,
    chunk_size: int = 1024,
) -> SkeletonProposalTargets:
    if points.ndim != 3 or points.shape[-1] != 4:
        raise ValueError(f"Expected points with shape [B, N, 4], got {tuple(points.shape)}")
    if skeleton_nodes.ndim != 3 or skeleton_nodes.shape[-1] < 5:
        raise ValueError(f"Expected skeleton_nodes with shape [B, M, >=5], got {tuple(skeleton_nodes.shape)}")
    if skeleton_mask.shape != skeleton_nodes.shape[:2]:
        raise ValueError("skeleton_mask must match the first two skeleton_nodes dimensions")
    if points.shape[0] != skeleton_nodes.shape[0]:
        raise ValueError("points and skeleton_nodes must have the same batch size")

    device = points.device
    batch_size, point_count, _ = points.shape
    point_xyz = points[..., :3]

    labels = torch.zeros((batch_size, point_count), dtype=torch.long, device=device)
    matched_centers = torch.zeros((batch_size, point_count, 3), dtype=points.dtype, device=device)
    matched_radius = torch.zeros((batch_size, point_count, 1), dtype=points.dtype, device=device)
    matched_distance = torch.full((batch_size, point_count), float("inf"), dtype=points.dtype, device=device)
    positive_mask = torch.zeros((batch_size, point_count), dtype=torch.bool, device=device)

    for batch_index in range(batch_size):
        valid_nodes = skeleton_nodes[batch_index, skeleton_mask[batch_index]]
        if valid_nodes.numel() == 0:
            continue

        node_xyz = valid_nodes[:, 1:4].to(dtype=points.dtype, device=device)
        node_radius = valid_nodes[:, 4:5].to(dtype=points.dtype, device=device).clamp_min(0.0)

        nearest_indices = []
        nearest_distances = []
        for start in range(0, point_count, chunk_size):
            end = min(start + chunk_size, point_count)
            distances = torch.cdist(point_xyz[batch_index, start:end], node_xyz)
            chunk_distance, chunk_index = distances.min(dim=1)
            nearest_distances.append(chunk_distance)
            nearest_indices.append(chunk_index)

        sample_distances = torch.cat(nearest_distances, dim=0)
        sample_indices = torch.cat(nearest_indices, dim=0)
        sample_centers = node_xyz[sample_indices]
        sample_radius = node_radius[sample_indices]
        positive_threshold = torch.maximum(
            torch.full_like(sample_radius, float(positive_distance)),
            sample_radius * float(radius_scale),
        ).squeeze(-1)
        sample_positive = sample_distances <= positive_threshold

        labels[batch_index] = sample_positive.to(dtype=torch.long)
        matched_centers[batch_index] = sample_centers
        matched_radius[batch_index] = sample_radius
        matched_distance[batch_index] = sample_distances
        positive_mask[batch_index] = sample_positive

    return SkeletonProposalTargets(
        objectness_labels=labels,
        matched_centers=matched_centers,
        matched_radius=matched_radius,
        matched_distance=matched_distance,
        positive_mask=positive_mask,
    )


def skeleton_proposal_loss(
    output: SkeletonProposalOutput,
    targets: SkeletonProposalTargets,
    points: torch.Tensor,
    objectness_weight: float = 1.0,
    center_weight: float = 1.0,
    radius_weight: float = 0.2,
    positive_class_weight: float = 8.0,
) -> SkeletonProposalLoss:
    labels = targets.objectness_labels
    positive_mask = targets.positive_mask
    class_weight = torch.tensor([1.0, float(positive_class_weight)], dtype=output.objectness_logits.dtype, device=output.objectness_logits.device)
    objectness = F.cross_entropy(output.objectness_logits.reshape(-1, 2), labels.reshape(-1), weight=class_weight)

    if bool(positive_mask.any()):
        coord_scale = coordinate_loss_scale(points)
        center = F.smooth_l1_loss(
            output.center_proposals[positive_mask] / coord_scale,
            targets.matched_centers[positive_mask] / coord_scale,
        )
        radius = F.smooth_l1_loss(output.radius[positive_mask], targets.matched_radius[positive_mask])
    else:
        center = output.center_proposals.sum() * 0.0
        radius = output.radius.sum() * 0.0

    total = objectness_weight * objectness + center_weight * center + radius_weight * radius
    return SkeletonProposalLoss(
        total=total,
        objectness=objectness.detach(),
        center=center.detach(),
        radius=radius.detach(),
        positive_count=int(positive_mask.sum().item()),
        total_count=int(positive_mask.numel()),
    )


def coordinate_loss_scale(points: torch.Tensor) -> torch.Tensor:
    xyz = points[..., :3]
    extent = xyz.amax(dim=(0, 1)) - xyz.amin(dim=(0, 1))
    return extent.max().clamp_min(1.0)

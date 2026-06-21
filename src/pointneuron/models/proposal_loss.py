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


@dataclass(frozen=True)
class PaperSkeletonProposalLoss:
    total: torch.Tensor
    offsets: torch.Tensor
    objectness: torch.Tensor
    radius: torch.Tensor
    positive_count: int
    total_count: int


def build_skeleton_proposal_targets(
    points: torch.Tensor,
    skeleton_nodes: torch.Tensor,
    skeleton_mask: torch.Tensor,
    positive_distance: float = 6.0,
    radius_scale: float = 1.5,
    target_radius_floor: float = 0.0,
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
        node_radius = valid_nodes[:, 4:5].to(dtype=points.dtype, device=device).clamp_min(float(target_radius_floor))

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


def paper_skeleton_proposal_loss(
    output: SkeletonProposalOutput,
    skeleton_nodes: torch.Tensor,
    skeleton_mask: torch.Tensor,
    points: torch.Tensor,
    offset_weight: float = 1.0,
    objectness_weight: float = 10.0,
    radius_weight: float = 1.0,
    positive_class_weight: float = 8.0,
    target_radius_floor: float = 0.0,
    chunk_size: int = 512,
) -> PaperSkeletonProposalLoss:
    if skeleton_nodes.ndim != 3 or skeleton_nodes.shape[-1] < 5:
        raise ValueError(f"Expected skeleton_nodes with shape [B, M, >=5], got {tuple(skeleton_nodes.shape)}")
    if skeleton_mask.shape != skeleton_nodes.shape[:2]:
        raise ValueError("skeleton_mask must match the first two skeleton_nodes dimensions")
    if output.center_proposals.shape[:2] != points.shape[:2]:
        raise ValueError("Output proposals and points must have matching batch and point dimensions")

    batch_size, point_count, _ = output.center_proposals.shape
    offset_losses = []
    objectness_losses = []
    radius_losses = []
    positive_count = 0
    total_count = batch_size * point_count

    for batch_index in range(batch_size):
        valid_nodes = skeleton_nodes[batch_index, skeleton_mask[batch_index]]
        if valid_nodes.numel() == 0:
            continue

        gt_centers = valid_nodes[:, 1:4].to(dtype=points.dtype, device=points.device)
        gt_radius = valid_nodes[:, 4:5].to(dtype=points.dtype, device=points.device).clamp_min(float(target_radius_floor))
        proposals = output.center_proposals[batch_index]
        predicted_radius = output.radius[batch_index]
        logits = output.objectness_logits[batch_index]

        proposal_to_gt_distance, nearest_indices = nearest_distances(proposals, gt_centers, chunk_size=chunk_size)
        gt_to_proposal_distance, _ = nearest_distances(gt_centers, proposals, chunk_size=chunk_size)

        scale = coordinate_scale(points[batch_index : batch_index + 1]).to(dtype=points.dtype, device=points.device)
        offset_loss = (proposal_to_gt_distance.square().mean() + gt_to_proposal_distance.square().mean()) / scale.square()
        offset_losses.append(offset_loss)

        nearest_radius = gt_radius[nearest_indices].squeeze(-1)
        labels = (proposal_to_gt_distance <= nearest_radius).to(dtype=torch.long)
        positive_count += int(labels.sum().item())
        class_weight = torch.tensor(
            [1.0, float(positive_class_weight)],
            dtype=logits.dtype,
            device=logits.device,
        )
        objectness_losses.append(F.cross_entropy(logits, labels, weight=class_weight))

        if bool(labels.any()):
            radius_losses.append(F.l1_loss(predicted_radius[labels.bool()], nearest_radius[labels.bool()].unsqueeze(-1)))
        else:
            radius_losses.append(predicted_radius.sum() * 0.0)

    if not offset_losses:
        zero = output.center_proposals.sum() * 0.0
        return PaperSkeletonProposalLoss(
            total=zero,
            offsets=zero.detach(),
            objectness=zero.detach(),
            radius=zero.detach(),
            positive_count=0,
            total_count=total_count,
        )

    offsets = torch.stack(offset_losses).mean()
    objectness = torch.stack(objectness_losses).mean()
    radius = torch.stack(radius_losses).mean()
    total = offset_weight * offsets + objectness_weight * objectness + radius_weight * radius
    return PaperSkeletonProposalLoss(
        total=total,
        offsets=offsets.detach(),
        objectness=objectness.detach(),
        radius=radius.detach(),
        positive_count=positive_count,
        total_count=total_count,
    )


def nearest_distances(query: torch.Tensor, reference: torch.Tensor, chunk_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    distances = []
    indices = []
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        chunk_distances = torch.cdist(query[start:end], reference)
        chunk_distance, chunk_index = chunk_distances.min(dim=1)
        distances.append(chunk_distance)
        indices.append(chunk_index)
    return torch.cat(distances, dim=0), torch.cat(indices, dim=0)


def coordinate_scale(points: torch.Tensor) -> torch.Tensor:
    xyz = points[..., :3]
    extent = xyz.amax(dim=(0, 1)) - xyz.amin(dim=(0, 1))
    return extent.max().clamp_min(1.0)


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
        center = F.smooth_l1_loss(
            output.center_proposals[positive_mask],
            targets.matched_centers[positive_mask],
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

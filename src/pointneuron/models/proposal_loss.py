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


@dataclass(frozen=True)
class ConservativeSkeletonProposalLoss:
    total: torch.Tensor
    objectness: torch.Tensor
    center: torch.Tensor
    radius: torch.Tensor
    offset_regularization: torch.Tensor
    non_worsen: torch.Tensor
    positive_count: int
    total_count: int


def build_skeleton_proposal_targets(
    points: torch.Tensor,
    skeleton_nodes: torch.Tensor,
    skeleton_mask: torch.Tensor,
    positive_distance: float = 6.0,
    radius_scale: float = 1.5,
    target_radius_floor: float = 0.0,
    objectness_radius_floor: float | None = None,
    radius_target_floor: float | None = None,
    radius_target_mode: str = "physical",
    selection_radius_floor: float = 8.0,
    selection_radius_ceiling: float = 0.0,
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
        raw_radius = valid_nodes[:, 4:5].to(dtype=points.dtype, device=device).clamp_min(0.0)
        objectness_floor = target_radius_floor if objectness_radius_floor is None else objectness_radius_floor
        radius_floor = target_radius_floor if radius_target_floor is None else radius_target_floor
        objectness_radius = raw_radius.clamp_min(float(objectness_floor))
        node_radius = raw_radius.clamp_min(float(radius_floor))

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
        sample_radius = proposal_radius_targets(
            physical_radius=node_radius,
            matched_indices=sample_indices,
            input_distance=sample_distances,
            radius_target_mode=radius_target_mode,
            selection_radius_floor=selection_radius_floor,
            selection_radius_ceiling=selection_radius_ceiling,
        )
        positive_threshold = torch.maximum(
            torch.full_like(sample_radius, float(positive_distance)),
            objectness_radius[sample_indices] * float(radius_scale),
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


def conservative_skeleton_proposal_loss(
    output: SkeletonProposalOutput,
    skeleton_nodes: torch.Tensor,
    skeleton_mask: torch.Tensor,
    points: torch.Tensor,
    positive_distance: float = 6.0,
    radius_scale: float = 1.5,
    objectness_weight: float = 2.0,
    center_weight: float = 1.0,
    radius_weight: float = 0.2,
    offset_regularization_weight: float = 0.05,
    non_worsen_weight: float = 1.0,
    non_worsen_margin: float = 0.0,
    center_beta: float = 4.0,
    non_worsen_beta: float = 4.0,
    positive_class_weight: float = 8.0,
    target_radius_floor: float = 0.0,
    objectness_radius_floor: float | None = None,
    radius_target_floor: float | None = None,
    radius_target_mode: str = "physical",
    selection_radius_floor: float = 8.0,
    selection_radius_ceiling: float = 0.0,
    chunk_size: int = 1024,
) -> ConservativeSkeletonProposalLoss:
    targets = build_skeleton_proposal_targets(
        points=points,
        skeleton_nodes=skeleton_nodes,
        skeleton_mask=skeleton_mask,
        positive_distance=positive_distance,
        radius_scale=radius_scale,
        target_radius_floor=target_radius_floor,
        objectness_radius_floor=objectness_radius_floor,
        radius_target_floor=radius_target_floor,
        radius_target_mode=radius_target_mode,
        selection_radius_floor=selection_radius_floor,
        selection_radius_ceiling=selection_radius_ceiling,
        chunk_size=chunk_size,
    )
    labels = targets.objectness_labels
    positive_mask = targets.positive_mask
    class_weight = torch.tensor(
        [1.0, float(positive_class_weight)],
        dtype=output.objectness_logits.dtype,
        device=output.objectness_logits.device,
    )
    objectness = F.cross_entropy(output.objectness_logits.reshape(-1, 2), labels.reshape(-1), weight=class_weight)

    if bool(positive_mask.any()):
        center_distance = torch.linalg.norm(
            output.center_proposals[positive_mask] - targets.matched_centers[positive_mask],
            dim=-1,
        )
        center = F.smooth_l1_loss(
            center_distance,
            torch.zeros_like(center_distance),
            beta=float(center_beta),
        )
        radius = F.smooth_l1_loss(output.radius[positive_mask], targets.matched_radius[positive_mask])
    else:
        center = output.center_proposals.sum() * 0.0
        radius = output.radius.sum() * 0.0

    offset_norm = output.offsets.norm(dim=-1)
    offset_regularization = offset_norm.square().mean()
    input_distance = targets.matched_distance
    output_distance = torch.linalg.norm(output.center_proposals - targets.matched_centers, dim=-1)
    worsening = F.relu(output_distance - input_distance - float(non_worsen_margin))
    non_worsen = F.smooth_l1_loss(
        worsening,
        torch.zeros_like(worsening),
        beta=float(non_worsen_beta),
    )
    total = (
        objectness_weight * objectness
        + center_weight * center
        + radius_weight * radius
        + offset_regularization_weight * offset_regularization
        + non_worsen_weight * non_worsen
    )
    return ConservativeSkeletonProposalLoss(
        total=total,
        objectness=objectness.detach(),
        center=center.detach(),
        radius=radius.detach(),
        offset_regularization=offset_regularization.detach(),
        non_worsen=non_worsen.detach(),
        positive_count=int(positive_mask.sum().item()),
        total_count=int(positive_mask.numel()),
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
    objectness_radius_floor: float | None = None,
    radius_target_floor: float | None = None,
    radius_target_mode: str = "physical",
    selection_radius_floor: float = 8.0,
    selection_radius_ceiling: float = 0.0,
    endpoint_weight: float = 1.0,
    branch_weight: float = 1.0,
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
        raw_radius = valid_nodes[:, 4:5].to(dtype=points.dtype, device=points.device).clamp_min(0.0)
        objectness_floor = target_radius_floor if objectness_radius_floor is None else objectness_radius_floor
        radius_floor = target_radius_floor if radius_target_floor is None else radius_target_floor
        objectness_radius = raw_radius.clamp_min(float(objectness_floor))
        gt_radius = raw_radius.clamp_min(float(radius_floor))
        gt_weights = skeleton_role_weights(valid_nodes, endpoint_weight=endpoint_weight, branch_weight=branch_weight).to(
            dtype=points.dtype,
            device=points.device,
        )
        proposals = output.center_proposals[batch_index]
        predicted_radius = output.radius[batch_index]
        logits = output.objectness_logits[batch_index]

        proposal_to_gt_distance, nearest_indices = nearest_distances(proposals, gt_centers, chunk_size=chunk_size)
        gt_to_proposal_distance, _ = nearest_distances(gt_centers, proposals, chunk_size=chunk_size)
        input_to_gt_distance, input_nearest_indices = nearest_distances(
            points[batch_index, :, :3],
            gt_centers,
            chunk_size=chunk_size,
        )

        scale = coordinate_scale(points[batch_index : batch_index + 1]).to(dtype=points.dtype, device=points.device)
        proposal_weights = gt_weights[nearest_indices]
        offset_loss = (
            weighted_mean(proposal_to_gt_distance.square(), proposal_weights)
            + weighted_mean(gt_to_proposal_distance.square(), gt_weights)
        ) / scale.square()
        offset_losses.append(offset_loss)

        nearest_radius = proposal_radius_targets(
            physical_radius=gt_radius,
            matched_indices=input_nearest_indices,
            input_distance=input_to_gt_distance,
            radius_target_mode=radius_target_mode,
            selection_radius_floor=selection_radius_floor,
            selection_radius_ceiling=selection_radius_ceiling,
        ).squeeze(-1)
        nearest_objectness_radius = objectness_radius[nearest_indices].squeeze(-1)
        labels = (proposal_to_gt_distance <= nearest_objectness_radius).to(dtype=torch.long)
        positive_count += int(labels.sum().item())
        class_weight = torch.tensor(
            [1.0, float(positive_class_weight)],
            dtype=logits.dtype,
            device=logits.device,
        )
        objectness = F.cross_entropy(logits, labels, weight=class_weight, reduction="none")
        objectness_losses.append(weighted_mean(objectness, proposal_weights))

        if bool(labels.any()):
            positive_mask = labels.bool()
            radius_error = F.l1_loss(predicted_radius[positive_mask], nearest_radius[positive_mask].unsqueeze(-1), reduction="none").squeeze(-1)
            radius_losses.append(weighted_mean(radius_error, proposal_weights[positive_mask]))
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


def proposal_radius_targets(
    physical_radius: torch.Tensor,
    matched_indices: torch.Tensor,
    input_distance: torch.Tensor,
    radius_target_mode: str = "physical",
    selection_radius_floor: float = 8.0,
    selection_radius_ceiling: float = 0.0,
) -> torch.Tensor:
    """Return the radius target for each source point/proposal.

    ``physical`` preserves the SWC node radius target. ``input_distance`` and
    ``selection`` train the proposal radius as a support sphere for NMS: the
    radius must at least cover the source point's distance to its matched
    skeleton center, and ``selection`` additionally applies a floor so nearby
    redundant proposals can suppress each other.
    """
    if radius_target_mode not in {"physical", "input_distance", "selection"}:
        raise ValueError(
            f"Unknown radius_target_mode {radius_target_mode!r}; "
            "expected 'physical', 'input_distance', or 'selection'"
        )

    matched_physical_radius = physical_radius[matched_indices]
    if radius_target_mode == "physical":
        return matched_physical_radius

    target = torch.maximum(input_distance.unsqueeze(-1), matched_physical_radius)
    if radius_target_mode == "selection":
        target = target.clamp_min(float(selection_radius_floor))
    if selection_radius_ceiling and selection_radius_ceiling > 0.0:
        target = target.clamp_max(float(selection_radius_ceiling))
    return target


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=values.dtype, device=values.device)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def skeleton_role_weights(skeleton_nodes: torch.Tensor, endpoint_weight: float = 1.0, branch_weight: float = 1.0) -> torch.Tensor:
    weights = torch.ones((skeleton_nodes.shape[0],), dtype=skeleton_nodes.dtype, device=skeleton_nodes.device)
    if skeleton_nodes.shape[0] == 0 or (float(endpoint_weight) == 1.0 and float(branch_weight) == 1.0):
        return weights

    node_ids = skeleton_nodes[:, 0].to(dtype=torch.long)
    parent_ids = skeleton_nodes[:, 5].to(dtype=torch.long)
    child_counts = torch.zeros((skeleton_nodes.shape[0],), dtype=torch.long, device=skeleton_nodes.device)
    for index, node_id in enumerate(node_ids.tolist()):
        child_counts[index] = (parent_ids == int(node_id)).sum()

    endpoint_mask = (child_counts == 0) & (parent_ids >= 0)
    branch_mask = child_counts > 1
    weights = torch.where(endpoint_mask, weights.new_full(weights.shape, float(endpoint_weight)), weights)
    weights = torch.where(branch_mask, weights.new_full(weights.shape, float(branch_weight)), weights)
    return weights


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

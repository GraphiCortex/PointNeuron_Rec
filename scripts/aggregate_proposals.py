from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.point_cloud import swc_to_skeleton_records, volume_to_point_cloud
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import indices_to_point_array, patch_foreground_indices, sample_indices_fixed, volume_data_array
from pointneuron.data.vaa3d_raw import read_header, read_volume
from scripts.visualize_proposals import render_html, select_proposals


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate PointNeuron center proposals over overlapping full-sample patches.")
    parser.add_argument("--root", default="data/gold166", help="Path to Gold166 root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index in the scanned Gold166 samples.")
    parser.add_argument("--checkpoint", required=True, help="Trained proposal checkpoint.")
    parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold; ignored when --threshold-fraction is set.")
    parser.add_argument("--threshold-fraction", type=float, default=0.2, help="Normalized foreground threshold in [0, 1].")
    parser.add_argument("--patch-radius", type=int, default=96, help="Half-width of cubic inference patches.")
    parser.add_argument("--stride", type=int, default=96, help="Sliding patch center stride in voxels.")
    parser.add_argument("--max-patches", type=int, default=256, help="Maximum patches to evaluate after foreground filtering.")
    parser.add_argument("--patch-selection", default="coverage", choices=["coverage", "density"], help="How to choose patches when --max-patches truncates the sliding grid.")
    parser.add_argument("--max-points", type=int, default=2048, help="Maximum foreground points per patch.")
    parser.add_argument("--min-points", type=int, default=256, help="Minimum foreground points required to evaluate a patch.")
    parser.add_argument("--batch-size", type=int, default=4, help="Patch inference batch size.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Per-patch objectness threshold before local NMS.")
    parser.add_argument("--top-proposals-per-patch", type=int, default=512, help="Maximum proposals retained from each patch.")
    parser.add_argument("--min-candidates", type=int, default=0, help="Adaptive per-patch pre-NMS candidate floor.")
    parser.add_argument("--candidate-score-floor", type=float, default=0.0, help="Minimum score used by adaptive candidate floors.")
    parser.add_argument("--local-nms-mode", default="sphere", choices=["sphere", "distance"], help="Per-patch proposal downsampling mode.")
    parser.add_argument("--global-nms-mode", default="sphere", choices=["sphere", "distance"], help="Final aggregated proposal downsampling mode.")
    parser.add_argument("--local-nms-radius", type=float, default=8.0, help="Per-patch distance NMS radius in voxels.")
    parser.add_argument("--sparse-local-nms-radius", type=float, help="Optional smaller local distance NMS radius for sparse patches.")
    parser.add_argument("--sparse-foreground-threshold", type=int, default=4096, help="Patch foreground count below which sparse local NMS settings apply.")
    parser.add_argument("--global-nms-radius", type=float, default=12.0, help="Final distance NMS radius in voxels.")
    parser.add_argument("--local-iou-threshold", type=float, default=0.1, help="Per-patch spherical NMS IoU threshold.")
    parser.add_argument("--global-iou-threshold", type=float, default=0.1, help="Global spherical NMS IoU threshold.")
    parser.add_argument("--global-top-proposals", type=int, default=4096, help="Maximum final aggregated skeleton proposals.")
    parser.add_argument("--max-center-foreground-distance", type=float, default=0.0, help="Discard local proposals farther than this distance from any patch foreground point. Set 0 to disable.")
    parser.add_argument("--foreground-support-radius", type=float, default=0.0, help="Radius used to count nearby foreground support around each local proposal. Set 0 to disable.")
    parser.add_argument("--min-foreground-support", type=int, default=0, help="Discard local proposals with fewer foreground points inside --foreground-support-radius.")
    parser.add_argument("--skip-support-for-sparse-patches", action="store_true", help="Do not apply foreground-support filtering to sparse patches.")
    parser.add_argument("--match-distance", type=float, default=6.0, help="Proposal-to-SWC distance counted as a hit.")
    parser.add_argument("--coverage-distance", type=float, default=8.0, help="SWC node distance counted as covered.")
    parser.add_argument("--render-points", type=int, default=12000, help="Foreground points to render in HTML.")
    parser.add_argument("--terminal-report", help="Optional CSV report for terminal and branch-node coverage diagnostics.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic patch point sampling.")
    parser.add_argument("--output", default="tmp/aggregations/proposals.npz", help="Output .npz path.")
    parser.add_argument("--html-output", default="tmp/visualizations/aggregated_proposals.html", help="Output HTML visualization path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before aggregating proposals.")
        return 2

    from pointneuron.models.dgcnn import DGCNNEncoder
    from pointneuron.models.proposal import SkeletonProposalHead

    samples = scan_gold166(args.root)
    sample = samples[args.sample_index]
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    header = read_header(sample.volume_path)
    swc = parse_swc(sample.swc_path)
    report = check_swc_in_volume(swc, header)
    if not report.is_aligned:
        print(f"sample_id: {sample.sample_id}")
        print(f"out_of_bounds_nodes: {len(report.out_of_bounds_node_ids)}")
        return 2

    volume = read_volume(sample.volume_path)
    threshold = effective_threshold(header.datatype, args.threshold, args.threshold_fraction)
    data = volume_data_array(volume)
    width, height, depth, channels = volume.dimensions
    if channels != 1:
        raise NotImplementedError(f"Expected a single-channel volume, got {channels} channels")

    foreground_indices = np.flatnonzero(data > threshold)
    if foreground_indices.size == 0:
        print(f"No foreground voxels above threshold {threshold}.")
        return 2

    patch_specs = build_patch_specs(
        foreground_indices=foreground_indices,
        data=data,
        width=width,
        height=height,
        depth=depth,
        patch_radius=args.patch_radius,
        stride=args.stride,
        threshold=threshold,
        min_points=args.min_points,
        max_patches=args.max_patches,
        patch_selection=args.patch_selection,
    )
    if not patch_specs:
        print("No inference patches survived foreground filtering.")
        return 2

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is false.")
        return 2

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    k = int(checkpoint_args.get("k", 20))
    proposal_coordinate_mode = checkpoint_args.get("proposal_coordinate_mode", "raw")
    encoder = DGCNNEncoder(k=k).to(device)
    proposal_in_channels = int(checkpoint["proposal"]["mlp.0.weight"].shape[1])
    proposal = SkeletonProposalHead(
        in_channels=proposal_in_channels,
        include_xyz=(proposal_in_channels == encoder.output_dim + 3),
        coordinate_mode=proposal_coordinate_mode,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    proposal.load_state_dict(checkpoint["proposal"])
    encoder.eval()
    proposal.eval()

    rng = np.random.default_rng(args.seed)
    all_centers = []
    all_radii = []
    all_scores = []
    all_features = []
    all_patch_indices = []
    evaluated_patches = 0

    with torch.no_grad():
        for batch_start in range(0, len(patch_specs), args.batch_size):
            batch_specs = patch_specs[batch_start : batch_start + args.batch_size]
            patch_points = []
            for patch_index, spec in batch_specs:
                selected_indices = sample_indices_fixed(spec["indices"], args.max_points, rng)
                patch_points.append(indices_to_point_array(selected_indices, data, width, height))

            points = torch.from_numpy(np.stack(patch_points).astype(np.float32, copy=False)).to(device)
            features = encoder(points)
            output = proposal(points, features)

            for batch_offset, (patch_index, spec) in enumerate(batch_specs):
                scores = output.objectness_logits[batch_offset].softmax(dim=-1)[:, 1]
                centers = output.center_proposals[batch_offset]
                radii = output.radius[batch_offset, :, 0]
                local_nms_radius = args.local_nms_radius
                if args.sparse_local_nms_radius is not None and spec["foreground"] <= args.sparse_foreground_threshold:
                    local_nms_radius = args.sparse_local_nms_radius
                selected = select_proposals(
                    centers=centers,
                    radii=radii,
                    scores=scores,
                    score_threshold=args.score_threshold,
                    min_candidates=args.min_candidates,
                    candidate_score_floor=args.candidate_score_floor,
                    top_proposals=args.top_proposals_per_patch,
                    nms_mode=args.local_nms_mode,
                    iou_threshold=args.local_iou_threshold,
                    nms_radius=local_nms_radius,
                )
                selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
                if not (args.skip_support_for_sparse_patches and spec["foreground"] <= args.sparse_foreground_threshold):
                    selected_tensor = filter_by_foreground_support(
                        selected_tensor=selected_tensor,
                        centers=centers,
                        patch_points=points[batch_offset, :, :3],
                        max_center_foreground_distance=args.max_center_foreground_distance,
                        foreground_support_radius=args.foreground_support_radius,
                        min_foreground_support=args.min_foreground_support,
                    )
                if selected_tensor.numel() > 0:
                    all_centers.append(centers[selected_tensor].detach().float().cpu())
                    all_radii.append(radii[selected_tensor].detach().float().cpu())
                    all_scores.append(scores[selected_tensor].detach().float().cpu())
                    all_features.append(features[batch_offset, selected_tensor].detach().float().cpu())
                    all_patch_indices.extend([patch_index] * int(selected_tensor.numel()))
                evaluated_patches += 1

            print(f"evaluated_patches: {evaluated_patches}/{len(patch_specs)}", flush=True)

    if all_centers:
        centers_tensor = torch.cat(all_centers, dim=0)
        radii_tensor = torch.cat(all_radii, dim=0)
        scores_tensor = torch.cat(all_scores, dim=0)
        features_tensor = torch.cat(all_features, dim=0)
    else:
        centers_tensor = torch.zeros((0, 3), dtype=torch.float32)
        radii_tensor = torch.zeros((0,), dtype=torch.float32)
        scores_tensor = torch.zeros((0,), dtype=torch.float32)
        features_tensor = torch.zeros((0, encoder.output_dim), dtype=torch.float32)

    final_indices = select_proposals(
        centers=centers_tensor,
        radii=radii_tensor,
        scores=scores_tensor,
        score_threshold=0.0,
        top_proposals=args.global_top_proposals,
        nms_mode=args.global_nms_mode,
        iou_threshold=args.global_iou_threshold,
        nms_radius=args.global_nms_radius,
    )

    final_centers = centers_tensor[final_indices] if final_indices else centers_tensor[:0]
    final_radii = radii_tensor[final_indices] if final_indices else radii_tensor[:0]
    final_scores = scores_tensor[final_indices] if final_indices else scores_tensor[:0]
    final_features = features_tensor[final_indices] if final_indices else features_tensor[:0]
    final_patch_indices = np.array([all_patch_indices[index] for index in final_indices], dtype=np.int32)

    skeleton_records = swc_to_skeleton_records(swc)
    skeleton_array = np.array([[node.node_id, node.x, node.y, node.z, node.radius, node.parent_id] for node in skeleton_records], dtype=np.float32)
    metrics = proposal_metrics(
        final_centers,
        centers_tensor,
        patch_specs,
        args.patch_radius,
        skeleton_array,
        args.match_distance,
        args.coverage_distance,
        torch,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        centers=final_centers.numpy(),
        radii=final_radii.numpy(),
        scores=final_scores.numpy(),
        features=final_features.numpy(),
        patch_indices=final_patch_indices,
        all_centers=centers_tensor.numpy(),
        all_radii=radii_tensor.numpy(),
        all_scores=scores_tensor.numpy(),
        metadata=np.array(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "sample_index": args.sample_index,
                    "checkpoint": args.checkpoint,
                    "threshold": threshold,
                    "threshold_fraction": args.threshold_fraction,
                    "patch_radius": args.patch_radius,
                    "stride": args.stride,
                    "patches": len(patch_specs),
                    "patch_selection": args.patch_selection,
                    "local_score_threshold": args.score_threshold,
                    "local_top_proposals": args.top_proposals_per_patch,
                    "global_top_proposals": args.global_top_proposals,
                    "local_nms_mode": args.local_nms_mode,
                    "global_nms_mode": args.global_nms_mode,
                    "local_nms_radius": args.local_nms_radius,
                    "sparse_local_nms_radius": args.sparse_local_nms_radius,
                    "sparse_foreground_threshold": args.sparse_foreground_threshold,
                    "global_nms_radius": args.global_nms_radius,
                    "max_center_foreground_distance": args.max_center_foreground_distance,
                    "foreground_support_radius": args.foreground_support_radius,
                    "min_foreground_support": args.min_foreground_support,
                    "skip_support_for_sparse_patches": args.skip_support_for_sparse_patches,
                    "local_iou_threshold": args.local_iou_threshold,
                    "global_iou_threshold": args.global_iou_threshold,
                    "metrics": metrics,
                }
            ),
            dtype=np.str_,
        ),
    )

    write_html(
        args=args,
        sample=sample,
        volume=volume,
        swc_records=skeleton_records,
        centers=final_centers,
        radii=final_radii,
        scores=final_scores,
        threshold=threshold,
    )
    if args.terminal_report:
        write_terminal_report(
            path=Path(args.terminal_report),
            skeleton_array=skeleton_array,
            final_centers=final_centers,
            local_centers=centers_tensor,
            patch_specs=patch_specs,
            patch_radius=args.patch_radius,
            coverage_distance=args.coverage_distance,
        )

    print(f"device: {device}")
    print(f"proposal_coordinate_mode: {proposal_coordinate_mode}")
    print(f"sample_id: {sample.sample_id}")
    print(f"volume_dimensions: {volume.dimensions}")
    print(f"foreground_voxels: {foreground_indices.size}")
    print(f"patches: {len(patch_specs)}")
    print(f"local_proposals: {centers_tensor.shape[0]}")
    print(f"final_proposals: {final_centers.shape[0]}")
    print(f"precision@{args.match_distance:g}: {metrics['precision']:.4f} ({metrics['hits']}/{metrics['selected']})")
    print(f"coverage@{args.coverage_distance:g}: {metrics['coverage']:.4f} ({metrics['covered']}/{metrics['nodes']})")
    print(
        f"terminal_coverage@{args.coverage_distance:g}: "
        f"{metrics['terminal_coverage']:.4f} ({metrics['terminal_covered']}/{metrics['terminals']})"
    )
    print(f"local_terminal_coverage@{args.coverage_distance:g}: {metrics['local_terminal_coverage']:.4f}")
    print(f"terminal_patch_coverage: {metrics['terminal_patch_coverage']:.4f}")
    print(
        f"branch_coverage@{args.coverage_distance:g}: "
        f"{metrics['branch_coverage']:.4f} ({metrics['branch_covered']}/{metrics['branch_nodes']})"
    )
    print(f"mean_selected_distance: {metrics['mean_distance']:.4f}")
    print(f"output: {output_path}")
    print(f"html_output: {args.html_output}")
    return 0


def effective_threshold(datatype: int, threshold: int, threshold_fraction: float | None) -> int:
    if threshold_fraction is None:
        return threshold
    if threshold_fraction < 0.0 or threshold_fraction > 1.0:
        raise ValueError(f"--threshold-fraction must be in [0, 1], got {threshold_fraction}")
    if datatype == 1:
        return int(round(255 * threshold_fraction))
    if datatype == 2:
        return int(round(65535 * threshold_fraction))
    raise NotImplementedError(f"Vaa3D datatype {datatype} is not supported yet")


def build_patch_specs(
    foreground_indices,
    data,
    width,
    height,
    depth,
    patch_radius,
    stride,
    threshold,
    min_points,
    max_patches,
    patch_selection,
):
    plane_size = width * height
    z = foreground_indices // plane_size
    offset = foreground_indices - z * plane_size
    y = offset // width
    x = offset - y * width
    mins = np.array([x.min(), y.min(), z.min()], dtype=np.float32)
    maxs = np.array([x.max(), y.max(), z.max()], dtype=np.float32)
    axes = [np.arange(mins[axis], maxs[axis] + 1, stride, dtype=np.float32) for axis in range(3)]
    if any(axis.size == 0 for axis in axes):
        return []

    specs = []
    for cx in axes[0]:
        for cy in axes[1]:
            for cz in axes[2]:
                center = np.array([cx, cy, cz], dtype=np.float32)
                indices = patch_foreground_indices(
                    data=data,
                    width=width,
                    height=height,
                    depth=depth,
                    center_xyz=center,
                    radius=patch_radius,
                    threshold=threshold,
                )
                if indices.shape[0] >= min_points:
                    specs.append((len(specs), {"center": center, "indices": indices, "foreground": int(indices.shape[0])}))

    if max_patches is not None and len(specs) > max_patches:
        if patch_selection == "density":
            specs.sort(key=lambda item: item[1]["foreground"], reverse=True)
            specs = specs[:max_patches]
        else:
            specs = select_spatially_diverse_patches(specs, max_patches, np.array([width, height, depth], dtype=np.float32))
    return [(index, spec) for index, (_original_spec_index, spec) in enumerate(specs)]


def select_spatially_diverse_patches(specs, max_patches, volume_shape):
    centers = np.stack([spec["center"] for _index, spec in specs]).astype(np.float32, copy=False)
    normalized = centers / np.maximum(volume_shape.reshape(1, 3), 1.0)
    foreground_counts = np.array([spec["foreground"] for _index, spec in specs], dtype=np.float32)
    selected = [int(np.argmax(foreground_counts))]
    remaining = np.ones(len(specs), dtype=bool)
    remaining[selected[0]] = False
    min_distances = np.linalg.norm(normalized - normalized[selected[0]], axis=1)

    while len(selected) < max_patches and bool(remaining.any()):
        score = min_distances + 0.05 * (foreground_counts / max(float(foreground_counts.max()), 1.0))
        score[~remaining] = -1.0
        next_index = int(np.argmax(score))
        selected.append(next_index)
        remaining[next_index] = False
        distances = np.linalg.norm(normalized - normalized[next_index], axis=1)
        min_distances = np.minimum(min_distances, distances)

    selected_specs = [specs[index] for index in selected]
    selected_specs.sort(key=lambda item: tuple(item[1]["center"].tolist()))
    return selected_specs


def proposal_metrics(centers, local_centers, patch_specs, patch_radius, skeleton_array, match_distance, coverage_distance, torch):
    skeleton_xyz = torch.from_numpy(skeleton_array[:, 1:4].astype(np.float32, copy=False))
    endpoint_mask, branch_mask = skeleton_role_masks(skeleton_array)
    if centers.shape[0] == 0:
        local_swc_to_selected = torch.cdist(skeleton_xyz, local_centers).min(dim=1).values if local_centers.shape[0] else None
        in_patch_mask = nodes_in_any_patch(skeleton_array[:, 1:4], patch_specs, patch_radius)
        return {
            "selected": 0,
            "hits": 0,
            "nodes": int(skeleton_xyz.shape[0]),
            "covered": 0,
            "terminals": int(endpoint_mask.sum()),
            "terminal_covered": 0,
            "branch_nodes": int(branch_mask.sum()),
            "branch_covered": 0,
            "precision": 0.0,
            "coverage": 0.0,
            "terminal_coverage": 0.0,
            "branch_coverage": 0.0,
            "local_terminal_coverage": (
                int((local_swc_to_selected[torch.from_numpy(endpoint_mask)] <= coverage_distance).sum().item()) / int(endpoint_mask.sum())
                if local_swc_to_selected is not None and int(endpoint_mask.sum()) > 0
                else 0.0
            ),
            "terminal_patch_coverage": int((endpoint_mask & in_patch_mask).sum()) / int(endpoint_mask.sum()) if int(endpoint_mask.sum()) else 0.0,
            "mean_distance": float("inf"),
            "median_distance": float("inf"),
        }
    selected_to_swc = torch.cdist(centers, skeleton_xyz).min(dim=1).values
    swc_to_selected = torch.cdist(skeleton_xyz, centers).min(dim=1).values
    local_swc_to_selected = torch.cdist(skeleton_xyz, local_centers).min(dim=1).values if local_centers.shape[0] else None
    in_patch_mask = nodes_in_any_patch(skeleton_array[:, 1:4], patch_specs, patch_radius)
    hits = int((selected_to_swc <= match_distance).sum().item())
    covered_mask = swc_to_selected <= coverage_distance
    covered = int(covered_mask.sum().item())
    endpoint_tensor = torch.from_numpy(endpoint_mask)
    branch_tensor = torch.from_numpy(branch_mask)
    terminal_covered = int(covered_mask[endpoint_tensor].sum().item()) if bool(endpoint_mask.any()) else 0
    branch_covered = int(covered_mask[branch_tensor].sum().item()) if bool(branch_mask.any()) else 0
    terminals = int(endpoint_mask.sum())
    branch_nodes = int(branch_mask.sum())
    return {
        "selected": int(centers.shape[0]),
        "hits": hits,
        "nodes": int(skeleton_xyz.shape[0]),
        "covered": covered,
        "terminals": terminals,
        "terminal_covered": terminal_covered,
        "branch_nodes": branch_nodes,
        "branch_covered": branch_covered,
        "precision": hits / int(centers.shape[0]),
        "coverage": covered / int(skeleton_xyz.shape[0]) if skeleton_xyz.shape[0] else 0.0,
        "terminal_coverage": terminal_covered / terminals if terminals else 0.0,
        "branch_coverage": branch_covered / branch_nodes if branch_nodes else 0.0,
        "local_terminal_coverage": (
            int((local_swc_to_selected[endpoint_tensor] <= coverage_distance).sum().item()) / terminals
            if local_swc_to_selected is not None and terminals
            else 0.0
        ),
        "terminal_patch_coverage": int((endpoint_mask & in_patch_mask).sum()) / terminals if terminals else 0.0,
        "mean_distance": float(selected_to_swc.mean().item()),
        "median_distance": float(selected_to_swc.median().item()),
    }


def filter_by_foreground_support(
    selected_tensor,
    centers,
    patch_points,
    max_center_foreground_distance,
    foreground_support_radius,
    min_foreground_support,
):
    if selected_tensor.numel() == 0:
        return selected_tensor
    if max_center_foreground_distance <= 0.0 and (foreground_support_radius <= 0.0 or min_foreground_support <= 0):
        return selected_tensor

    selected_centers = centers[selected_tensor]
    distances = selected_centers.new_empty((selected_centers.shape[0], patch_points.shape[0]))
    for start in range(0, selected_centers.shape[0], 256):
        end = min(start + 256, selected_centers.shape[0])
        distances[start:end] = (selected_centers[start:end, None, :] - patch_points[None, :, :]).square().sum(dim=-1).sqrt()

    keep = selected_tensor.new_ones((selected_tensor.numel(),), dtype=bool)
    if max_center_foreground_distance > 0.0:
        keep &= distances.min(dim=1).values <= float(max_center_foreground_distance)
    if foreground_support_radius > 0.0 and min_foreground_support > 0:
        support = (distances <= float(foreground_support_radius)).sum(dim=1)
        keep &= support >= int(min_foreground_support)
    return selected_tensor[keep]


def nodes_in_any_patch(node_xyz: np.ndarray, patch_specs, patch_radius: float) -> np.ndarray:
    if len(patch_specs) == 0:
        return np.zeros((node_xyz.shape[0],), dtype=bool)
    mask = np.zeros((node_xyz.shape[0],), dtype=bool)
    for _patch_index, spec in patch_specs:
        center = spec["center"].reshape(1, 3)
        # Patch membership is cubic, matching patch_foreground_indices.
        mask |= np.max(np.abs(node_xyz - center), axis=1) <= float(patch_radius)
    return mask


def write_terminal_report(path: Path, skeleton_array, final_centers, local_centers, patch_specs, patch_radius: float, coverage_distance: float) -> None:
    import csv
    import torch

    endpoint_mask, branch_mask = skeleton_role_masks(skeleton_array)
    role = np.full((skeleton_array.shape[0],), "internal", dtype=object)
    role[endpoint_mask] = "terminal"
    role[branch_mask] = "branch"
    final_distances = nearest_node_distances(skeleton_array[:, 1:4], final_centers, torch)
    local_distances = nearest_node_distances(skeleton_array[:, 1:4], local_centers, torch)
    in_patch_mask = nodes_in_any_patch(skeleton_array[:, 1:4], patch_specs, patch_radius)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "node_id",
                "role",
                "x",
                "y",
                "z",
                "radius",
                "in_evaluated_patch",
                "local_distance",
                "final_distance",
                "covered",
            ],
        )
        writer.writeheader()
        for index, node in enumerate(skeleton_array):
            if role[index] == "internal":
                continue
            writer.writerow(
                {
                    "node_id": int(node[0]),
                    "role": role[index],
                    "x": float(node[1]),
                    "y": float(node[2]),
                    "z": float(node[3]),
                    "radius": float(node[4]),
                    "in_evaluated_patch": bool(in_patch_mask[index]),
                    "local_distance": float(local_distances[index]),
                    "final_distance": float(final_distances[index]),
                    "covered": bool(final_distances[index] <= coverage_distance),
                }
            )


def nearest_node_distances(node_xyz: np.ndarray, centers, torch) -> np.ndarray:
    if centers.shape[0] == 0:
        return np.full((node_xyz.shape[0],), float("inf"), dtype=np.float32)
    node_tensor = torch.from_numpy(node_xyz.astype(np.float32, copy=False))
    return torch.cdist(node_tensor, centers).min(dim=1).values.numpy()


def skeleton_role_masks(skeleton_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_ids = skeleton_array[:, 0].astype(np.int64, copy=False)
    parent_ids = skeleton_array[:, 5].astype(np.int64, copy=False)
    child_counts = {int(node_id): 0 for node_id in node_ids}
    for parent_id in parent_ids:
        if int(parent_id) in child_counts:
            child_counts[int(parent_id)] += 1
    endpoint_mask = np.array(
        [child_counts[int(node_id)] == 0 and int(parent_id) >= 0 for node_id, parent_id in zip(node_ids, parent_ids)],
        dtype=bool,
    )
    branch_mask = np.array([child_counts[int(node_id)] > 1 for node_id in node_ids], dtype=bool)
    return endpoint_mask, branch_mask


def write_html(args, sample, volume, swc_records, centers, radii, scores, threshold):
    if not args.html_output:
        return
    point_cloud = volume_to_point_cloud(volume, threshold=threshold, max_points=args.render_points, seed=args.seed)
    proposals = [
        [
            float(centers[index, 0].item()),
            float(centers[index, 1].item()),
            float(centers[index, 2].item()),
            float(radii[index].item()),
            float(scores[index].item()),
            0.0,
        ]
        for index in range(centers.shape[0])
    ]
    html = render_html(
        sample_id=sample.sample_id,
        volume_dimensions=point_cloud.volume_dimensions,
        total_foreground_count=point_cloud.total_foreground_count,
        points=[[point.x, point.y, point.z, point.intensity] for point in point_cloud.points],
        skeleton=[[node.node_id, node.x, node.y, node.z, node.radius, node.parent_id] for node in swc_records],
        proposals=proposals,
        split="full-sample",
        checkpoint=args.checkpoint,
    )
    output = Path(args.html_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
